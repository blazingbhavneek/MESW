import os
import json
from pathlib import Path
from typing import List, Dict
import multiprocessing as mp
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import asyncio
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from tqdm import tqdm
import neo4j
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.llm.types import LLMResponse
import httpx

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "kaitchup/Phi-4-AutoRound-GPTQ-4bit"
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
MAX_MODEL_LEN = 8192
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100  # tokens
BATCH_SIZE = 10
# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"  # Change if needed
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # Change this!
# Paths
MARKDOWN_FOLDER = "./ja-wikipedia/wikipedia_dataset_small"  # Change this
OUTPUT_TERMS = "./all_terms.json"


# ----------------------------
# CUSTOM LLM FOR NEO4J_GRAPHRAG (Modified for vLLM Server API)
# ----------------------------
class VLLMLLM(LLMInterface):
    """Wrapper to use vLLM Server API with neo4j-graphrag"""
    
    def __init__(self, api_base_url: str, api_key: str = "token-abc123"): # Use actual API key if required
        self.api_base_url = api_base_url.rstrip('/') # e.g., "http://localhost:8000/v1"
        self.api_key = api_key
        # Use httpx.AsyncClient for async calls, httpx.Client for sync calls
        # Using AsyncClient here for both sync and async methods for potentially better performance
        # Or use separate clients if preferred.
        # Timeout should be reasonably high for LLM generation
        self.async_client = httpx.AsyncClient(timeout=120.0) 
        self.sync_client = httpx.Client(timeout=120.0)

    def _prepare_request_payload(self, input_text: str) -> dict:
         # Prepare the payload according to vLLM's OpenAI-compatible API schema
         # Adjust parameters like temperature, top_p, max_tokens as needed
         return {
             "model": "", # Often optional or can be empty for single-model servers
             "messages": [{"role": "user", "content": input_text}],
             "temperature": 0.0,
             "top_p": 0.1,
             "max_tokens": 1500,
             # "stop": [self.tokenizer.eos_token_id], # vLLM API might not support stop_token_ids directly like this
             # Check vLLM API docs for stop string support if needed
             "stop": ["</s>"] # Example using a stop string if applicable
         }

    def invoke(self, input: str, **kwargs) -> LLMResponse:
        """Single generation via sync HTTP request."""
        payload = self._prepare_request_payload(input)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = self.sync_client.post(f"{self.api_base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            # Extract text from the response (structure depends on the API)
            # Example for OpenAI-compatible API:
            generated_text = result['choices'][0]['message']['content'].strip()
            return LLMResponse(content=generated_text)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            raise
        except KeyError as e:
            print(f"Key error - possibly malformed response: {e}, Response: {result}")
            raise
        # Ensure client is closed if needed (though usually handled by context or script end)
        # self.sync_client.close() # Only call this when completely done with the client instance

    async def ainvoke(self, input: str, **kwargs) -> LLMResponse:
        """Async generation via async HTTP request."""
        payload = self._prepare_request_payload(input)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = await self.async_client.post(f"{self.api_base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            # Extract text from the response
            generated_text = result['choices'][0]['message']['content'].strip()
            return LLMResponse(content=generated_text)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            raise
        except KeyError as e:
            print(f"Key error - possibly malformed response: {e}, Response: {result}")
            raise
        # Ensure client is closed if needed (though usually handled by context or script end)
        # await self.async_client.aclose() # Only call this when completely done with the client instance


# ----------------------------
# CUSTOM EMBEDDER FOR NEO4J_GRAPHRAG
# ----------------------------
class HFEmbedder(Embedder):
    def __init__(self, model_name: str, local_dir: str = "./models/embedding/"):
        # if local_dir exists, use that, else fallback to model_name
        src = local_dir if (local_dir and os.path.isdir(local_dir)) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True, local_files_only=bool(local_dir))
        self.model = AutoModel.from_pretrained(src, trust_remote_code=True, local_files_only=bool(local_dir))
    
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden_states = self.model(**inputs).last_hidden_state
            emb = hidden_states.mean(dim=1)[0].cpu().tolist()
        return emb

    async def aembed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)
# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text by token count with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += (chunk_size - overlap)
    
    return chunks
def load_markdown_files(folder_path: str) -> List[tuple]:
    """Load all markdown files from folder"""
    files = []
    folder = Path(folder_path)
    for md_file in folder.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            files.append((md_file.name, content))
    return files
# ----------------------------
# CUSTOM SCHEMA FOR WIKIPEDIA
# ----------------------------
node_types = [
    {
        "label": "Term",
        "description": "A term, concept, person, place, or entity from Wikipedia",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"}
        ]
    }
]

rel_types = [
    {"label": "PREREQUISITE_OF", "description": "Source is prerequisite for target"},
    {"label": "RELATED_TO", "description": "Semantic relationship between terms"},
    {"label": "SIMILAR_TO", "description": "High semantic similarity between terms"}
]

patterns = [
    ("Term", "PREREQUISITE_OF", "Term"),
    ("Term", "RELATED_TO", "Term"),
    ("Term", "SIMILAR_TO", "Term")
]
# Custom extraction prompt
extraction_prompt = """
あなたは知識抽出の専門家です。以下のテキストから重要な用語、エンティティを抽出し、それらの関係を特定してください。
テキスト:
{text}
Use the following schema:
{schema}
以下の形式でJSON出力してください:
{{
  "nodes": [
    {{
      "id": "unique_id",
      "label": "Term",
      "properties": {{
        "name": "用語名",
        "definition": "用語の定義または説明"
      }}
    }}
  ],
  "relationships": [
    {{
      "type": "PREREQUISITE_OF",
      "start_node_id": "source_id",
      "end_node_id": "target_id",
      "properties": {{}}
    }},
    {{
      "type": "RELATED_TO",
      "start_node_id": "source_id",
      "end_node_id": "target_id",
      "properties": {{}}
    }},
    {{
      "type": "SIMILAR_TO",
      "start_node_id": "source_id",
      "end_node_id": "target_id",
      "properties": {{}}
    }}
  ]
}}
注意:
- 単一の単語と複合語の両方を抽出してください
- definitionには用語の完全な説明を含めてください
- PREREQUISITE_OF関係は、理解の階層を示します
- RELATED_TO関係は、一般的な関連性を示します
- SIMILAR_TO関係は、類似性を示します
Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.
Do not return any additional information other than the JSON.
"""
# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True) # Likely not needed anymore, can remove
    print("=" * 60)
    print("Wikipedia GraphRAG with Neo4j (vLLM via API)")
    print("=" * 60)

    # 1. Load markdown files
    print(f"\n[1/6] Loading markdown files from {MARKDOWN_FOLDER}...")
    md_files = load_markdown_files(MARKDOWN_FOLDER)
    print(f"Loaded {len(md_files)} files")

    # For testing, limit to first 2 files
    # md_files = md_files[:1]

    # 2. Initialize components (models are now on the server)
    print("\n[2/6] Initializing components (LLM via API)...")
    # DO NOT initialize the local LLM object here
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # llm = LLM( ... )
    
    # Initialize your API-based LLM wrapper
    VLLM_API_BASE = "http://localhost:8000/v1" # Adjust to your server's address and port
    vllm_model = VLLMLLM(api_base_url=VLLM_API_BASE)
    # No concurrency limit needed on the wrapper itself if using API,
    # but the server handles its own concurrency. You might still want
    # to limit asyncio.gather tasks depending on server capacity and stability.
    # Consider adding a semaphore here too if the server struggles with many concurrent requests.
    # Example:
    # import asyncio
    # llm_semaphore = asyncio.Semaphore(5) # Adjust limit
    # You'd need to integrate this semaphore into the ainvoke call if added here.

    embedder = HFEmbedder(EMBEDDING_MODEL) # Embedder runs locally

    # 3. Initialize Neo4j Driver
    print("\n[3/6] Connecting to Neo4j...")
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("Neo4j Driver initialized")

    # Initialize SimpleKGPipeline (with API-based LLM)
    text_splitter = FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    schema = {
        "node_types": node_types,
        "relationship_types": rel_types,
        "patterns": patterns
    }
    kg_builder = SimpleKGPipeline(
        llm=vllm_model, # Use the API-based wrapper
        driver=driver,
        embedder=embedder,
        text_splitter=text_splitter,
        schema=schema,
        prompt_template=extraction_prompt,
        from_pdf=False
    )

    # 4. Process and ingest documents...
    print("\n[4/6] Processing and ingesting documents (via API)...")
    async def ingest_documents(md_files):
        for filename, content in tqdm(md_files, desc="Processing files"):
            print(f"Starting ingestion for file: {filename}")
            try:
                await kg_builder.run_async(text=content)
                print(f"Completed ingestion for file: {filename}")
            except Exception as e:
                print(f"Error ingesting file {filename}: {e}")
                # Handle error as needed

    try:
        asyncio.run(ingest_documents(md_files))
        print("Document ingestion complete")
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user.")
    finally:
        # Ensure clients are closed
        try:
            # Access the client instance from the vllm_model wrapper
            asyncio.run(vllm_model.async_client.aclose())
        except Exception as e:
            print(f"Error closing async client: {e}")
        try:
            vllm_model.sync_client.close()
        except Exception as e:
            print(f"Error closing sync client: {e}")

        print("\n[Finalizing] Closing Neo4j driver...")
        driver.close()
        print("Neo4j driver closed.")

    # 5. Extract all terms for fuzzy matching
    print("\n[5/6] Creating vector index and computing embeddings (if nodes exist)...")
    try:
        # Re-initialize the driver temporarily
        temp_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with temp_driver.session() as session:
            session.run("""
                CREATE VECTOR INDEX term_vector IF NOT EXISTS
                FOR (t:Term) ON (t.embedding)
                OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
            """)
                
            # Check if any Term nodes exist before attempting embedding
            count_result = session.run("MATCH (t:Term) RETURN count(t) AS term_count")
            term_count = count_result.single()["term_count"]
            print(f"Found {term_count} Term nodes to process for embeddings.")
            
            if term_count > 0:
                # Fetch nodes without embeddings and compute them
                # Use tqdm if you expect many nodes needing embeddings
                result = session.run("MATCH (t:Term) WHERE t.embedding IS NULL RETURN t.name as name, t.definition as definition")
                for record in result: # Removed tqdm here for simplicity, add back if needed
                    name = record["name"]
                    definition = record["definition"] or ""
                    emb = embedder.embed_query(definition)
                    session.run("MATCH (t:Term {name: $name}) SET t.embedding = $emb", name=name, emb=emb)
            else:
                print("No Term nodes found. Skipping embedding computation.")
    except Exception as e:
        print(f"Error during index creation or embedding: {e}")
    finally:
        try:
            temp_driver.close()
        except:
            pass # Ignore error if temp_driver wasn't created due to exception above

    # 6. Extract all terms for fuzzy matching (if nodes exist) - Now using the original driver (if still open) or re-open
    print("\n[6/6] Extracting all terms (if nodes exist)...")
    try:
        # Use the original driver, but check if it's still open. If not, re-open temporarily.
        # Since driver.close() was called in the finally block, we need to re-open here too.
        temp_driver2 = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with temp_driver2.session() as session:
            result = session.run("MATCH (t:Term) RETURN t.name as name")
            all_terms = [record["name"] for record in result]
        
        with open(OUTPUT_TERMS, 'w', encoding='utf-8') as f:
            json.dump(all_terms, f, ensure_ascii=False, indent=2)
        
        print(f"Extracted {len(all_terms)} unique terms")
        print(f"Saved to {OUTPUT_TERMS}")
    except Exception as e:
        print(f"Error extracting terms: {e}")
    finally:
        try:
            temp_driver2.close()
        except:
            pass # Ignore error if temp_driver2 wasn't created due to exception above

    # 7. Example Queries - Now initialize the retrievers AFTER the index exists
    print("\n[7/6] Running example queries...")
    print("=" * 60)

    try:
        # Initialize retrievers - This should now work as the index 'term_vector' exists
        retrieval_query = """
        OPTIONAL MATCH (prereq:Term)-[:PREREQUISITE_OF]->(node)
        OPTIONAL MATCH (node)-[:RELATED_TO]->(related:Term)
        RETURN node.name as term, 
            node.definition as definition,
            collect(DISTINCT prereq.name) as prerequisites,
            collect(DISTINCT related.name) as related_terms
        """
        vector_cypher_retriever = VectorCypherRetriever(
            driver=driver, # Use the original driver instance (which is now closed, so this will fail too!)
            index_name="term_vector",
            retrieval_query=retrieval_query,
            embedder=embedder
        )

        # text2cypher_retriever initialization would go here if needed for examples

        # Example 1: VectorCypher retrieval
        print("\n--- Example 1: VectorCypher Retrieval ---")
        query1 = "電力"
        results1 = vector_cypher_retriever.search(query_text=query1, top_k=3)
        print(f"\nQuery: {query1}")
        print(f"Raw RAG Output (VectorCypher):")
        out = {
            "items": [
                {
                    "content": item.content
                }
                for item in results1.items
            ]
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        # print(json.dumps(results1, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Error during example queries: {e}")
        print("This might be because the 'driver' object used for the retriever was closed earlier.")
        print("Re-initializing the driver for the examples...")
        example_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            vector_cypher_retriever = VectorCypherRetriever(
                driver=example_driver,
                index_name="term_vector",
                retrieval_query=retrieval_query,
                embedder=embedder
            )
            # Example 1: VectorCypher retrieval
            print("\n--- Example 1: VectorCypher Retrieval (with new driver) ---")
            query1 = "電力"
            results1 = vector_cypher_retriever.search(query_text=query1, top_k=3)
            print(f"\nQuery: {query1}")
            print(f"Raw RAG Output (VectorCypher):")
            out = {
                "items": [
                    {
                        "content": item.content
                    }
                    for item in results1.items
                ]
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            # print(json.dumps(results1, ensure_ascii=False, indent=2))
        except Exception as e2:
            print(f"Error running example with new driver: {e2}")
        finally:
            example_driver.close()

    finally:
        # Close the original driver again if it was somehow reopened (unlikely after finally block)
        # Or just ensure everything is cleaned up if managing differently
        print("\n" + "=" * 60)
        print("GraphRAG setup attempted!")
        print(f"Neo4j database: {NEO4J_URI}")
        print("Check Neo4j Browser for ingested data.")
        print("You can now query the graph using:")
        print("  - VectorCypherRetriever")
        print("  - Text2CypherRetriever")
        print("  - Custom Cypher queries")
        print("=" * 60)

        # The interactive loop part can be commented out or run in a separate script
        # after confirming the ingestion and index creation steps ran fully.
        # For now, just print the summary.
        print("\nScript execution finished.")
