import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import json_repair
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Configuration
MAX_CONCURRENT_DEFINITIONS = 15
WORKING_DIR = "./graph_rag_cache"
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKEN = 2000
DEFINITION_SIMILARITY_THRESHOLD = 0.7  # For semantic similarity in graph

# Initialize embedding model
try:
    EMBED_MODEL = SentenceTransformer("models/embeddings", local_files_only=True)
    print("Local embedding model loaded")
except:
    try:
        EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("Fallback embedding model loaded")
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        sys.exit(1)

# vLLM client
vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=VLLM_HOST + "/v1",
)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def vllm_complete(prompt, system_prompt=None, **kwargs) -> str:
    """General vLLM completion function"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that provides clear, concise responses. Respond directly without any reasoning, explanations, or extra text. Output only the requested information in the exact format specified."
    else:
        system_prompt += " Respond directly without any reasoning, explanations, or extra text. Output only the requested information in the exact format specified."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKEN,
            temperature=0.1,
            extra_body={
                "repetition_penalty": 1.1,
                "top_p": 0.9,
            },
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"vLLM request failed: {e}")
        return ""


# ============================================================================
# TERM EXTRACTOR - PLACEHOLDER
# ============================================================================
def extract_technical_terms(text: str, **kwargs) -> Dict[str, str]:
    """
    Placeholder function to extract technical terms.
    CRITICAL: Later this will be replaced with a hardcoded list of terms.
    For now, it returns a dictionary of terms with empty definitions.
    """
    terms = {}
    patterns = [
        r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b",  # hyphenated terms
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # proper noun phrases
        r"\b[a-z]+[A-Z][a-z]*\b",  # camelCase terms
        r"\b[A-Z]{2,}\b",  # acronyms
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            term = match.lower().strip()
            if len(term) > 3:
                terms[term] = ""

    print(f"Extracted {len(terms)} placeholder technical terms")
    return terms


# ============================================================================
# BOOK CHUNKER
# ============================================================================
class BookChunker:
    """Split book into manageable chunks for processing"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            import tiktoken

            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except:
            self.tokenizer = None
            self.use_tiktoken = False
            print("tiktoken not available, using approximate word-based chunking")

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from the book text"""
        print(f"Creating chunks from book ({len(text)} characters)...")
        if self.use_tiktoken:
            return self._create_token_chunks(text)
        else:
            return self._create_word_chunks(text)

    def _create_token_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks based on token count"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(
                {
                    "id": len(chunks),
                    "text": chunk_text,
                    "start_token": i,
                    "end_token": i + len(chunk_tokens),
                    "token_count": len(chunk_tokens),
                }
            )
        print(f"Created {len(chunks)} token-based chunks")
        return chunks

    def _create_word_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks based on word count (fallback)"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                {
                    "id": len(chunks),
                    "text": chunk_text,
                    "start_word": i,
                    "end_word": i + len(chunk_words),
                    "word_count": len(chunk_words),
                }
            )
        print(f"Created {len(chunks)} word-based chunks")
        return chunks


# ============================================================================
# CHUNK PROCESSOR
# ============================================================================
class ChunkProcessor:
    """Process individual chunks to find terms and generate definitions"""

    def __init__(self):
        self.chunk_terms = {}
        self.chunk_definitions = {}
        self.global_terms = set()

    async def process_chunk(
        self, chunk: Dict[str, Any], given_terms: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process a single chunk to find terms and generate definitions"""
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
        print(f"Processing chunk {chunk_id}")

        # Find which given terms are present in this chunk
        present_terms = []
        for term in given_terms.keys():
            if term.lower() in chunk_text.lower():
                present_terms.append(term)

        print(f"  Found {len(present_terms)} given terms in chunk")

        # Generate definitions for terms in this chunk
        definitions = {}
        if present_terms:
            definitions = await self._generate_chunk_definitions(
                chunk_text, present_terms, chunk_id
            )

        self.chunk_terms[chunk_id] = present_terms
        self.chunk_definitions[chunk_id] = definitions
        self.global_terms.update(present_terms)

        return {
            "chunk_id": chunk_id,
            "terms_found": len(present_terms),
            "definitions_generated": len(definitions),
            "terms": present_terms,
            "definitions": definitions,
        }

    async def _generate_chunk_definitions(
        self, chunk_text: str, terms: List[str], chunk_id: int
    ) -> Dict[str, str]:
        """Generate definitions for terms found in a specific chunk"""
        definitions = {}
        batch_size = 5
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i : i + batch_size]
            prompt = self._create_definition_prompt(chunk_text, batch_terms)
            try:
                response = await vllm_complete(
                    prompt, self._get_definition_system_prompt()
                )
                batch_definitions = self._parse_definitions_response(
                    response, batch_terms
                )
                definitions.update(batch_definitions)
                print(
                    f"    Chunk {chunk_id}: Generated definitions for batch {i//batch_size + 1}"
                )
            except Exception as e:
                print(
                    f"    Failed to generate definitions for chunk {chunk_id}, batch {i//batch_size + 1}: {e}"
                )
                for term in batch_terms:
                    definitions[term] = (
                        f"Term from chunk {chunk_id} context: {chunk_text[:200]}..."
                    )
        return definitions

    def _create_definition_prompt(self, chunk_text: str, terms: List[str]) -> str:
        """Create prompt for generating definitions from chunk context"""
        term_list = ", ".join(terms)
        return f"""Based on the following text chunk, provide clear, concise definitions for these technical terms:
Terms to define: {term_list}
Text chunk context:
{chunk_text}
For each term, provide a definition in this exact format:
TERM: [term name]
DEFINITION: [1-2 sentence definition based on the context]
Only define terms that actually appear in the context. If a term doesn't appear, skip it.
Do not include any reasoning, explanations, or text outside the format."""

    def _get_definition_system_prompt(self) -> str:
        """Get system prompt for definition generation"""
        return """You are an expert at creating clear, contextual definitions for technical terms found in books. 
Based on the provided text chunk, create precise definitions for the requested terms. 
Use ONLY the context provided. Keep definitions to 1-2 sentences.
Follow the exact format requested: TERM: [name] followed by DEFINITION: [definition].
Do not include any reasoning, explanations, or additional text outside the format."""

    def _parse_definitions_response(
        self, response: str, expected_terms: List[str]
    ) -> Dict[str, str]:
        """Parse the LLM response to extract definitions"""
        definitions = {}
        sections = re.split(r"\n(?=TERM:)", response)
        for section in sections:
            term_match = re.search(r"TERM:\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
            def_match = re.search(
                r"DEFINITION:\s*(.+?)(?=\nTERM:|\n|$)",
                section,
                re.IGNORECASE | re.DOTALL,
            )
            if term_match and def_match:
                term = term_match.group(1).strip().lower()
                definition = def_match.group(1).strip()
                definition = re.sub(r"\s+", " ", definition)
                if definition and not definition.endswith((".", "!", "?")):
                    definition += "."
                definitions[term] = definition

        missing_terms = [t for t in expected_terms if t.lower() not in definitions]
        if missing_terms:
            print(f"    Missing definitions for: {missing_terms}")

        return definitions


# ============================================================================
# DICTIONARY MANAGER
# ============================================================================
class DictionaryManager:
    """Manages the term dictionary with character-based lookup"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.dictionary = {}  # term -> definition
        os.makedirs(working_dir, exist_ok=True)

    def add_terms(self, terms_with_definitions: Dict[str, str]) -> None:
        """Add or update terms in dictionary"""
        print(f"Adding {len(terms_with_definitions)} terms to dictionary...")
        for term, definition in terms_with_definitions.items():
            if term in self.dictionary:
                # Term exists - merge definitions if different
                existing_def = self.dictionary[term]
                similarity = SequenceMatcher(
                    None, definition.lower(), existing_def.lower()
                ).ratio()
                if similarity < 0.7:
                    # Definitions are different, merge them
                    self.dictionary[term] = f"{existing_def} {definition}"
            else:
                # New term
                self.dictionary[term] = definition
        print(f"Dictionary now contains {len(self.dictionary)} terms")

    def lookup_exact(self, term: str) -> Optional[str]:
        """Exact character match lookup"""
        term_normalized = term.lower().strip()
        return self.dictionary.get(term_normalized)

    def lookup_similar(self, term: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Character similarity based lookup using fuzzy matching"""
        term_normalized = term.lower().strip()

        # Calculate similarity for all terms
        similarities = []
        for dict_term in self.dictionary.keys():
            similarity = SequenceMatcher(None, term_normalized, dict_term).ratio()
            similarities.append(
                {
                    "term": dict_term,
                    "similarity": similarity,
                    "definition": self.dictionary[dict_term],
                }
            )

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    def save(self) -> None:
        """Save dictionary to disk"""
        dict_path = os.path.join(self.working_dir, "dictionary.json")
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(self.dictionary, f, indent=2, ensure_ascii=False)
        print(f"Dictionary saved to {dict_path}")

    def load(self) -> bool:
        """Load dictionary from disk"""
        dict_path = os.path.join(self.working_dir, "dictionary.json")
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
            print(f"Loaded {len(self.dictionary)} terms from dictionary")
            return True
        return False


# ============================================================================
# GRAPH MANAGER
# ============================================================================
class GraphManager:
    """Manages the GraphRAG knowledge graph for semantic similarity"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.terms_definitions = {}  # term -> definition
        self.term_embeddings = {}  # term -> embedding vector
        os.makedirs(working_dir, exist_ok=True)

    def add_terms(self, terms_with_definitions: Dict[str, str]) -> None:
        """Add terms to graph - definitions will be used for semantic similarity"""
        print(f"Adding {len(terms_with_definitions)} terms to graph...")
        self.terms_definitions.update(terms_with_definitions)
        print(f"Graph now contains {len(self.terms_definitions)} terms")

    def create_graph_chunks(self) -> List[str]:
        """
        Create chunks for GraphRAG where:
        - Each chunk is: TERM + DEFINITION
        - GraphRAG will create nodes from these
        - Relationships will be based on:
          1. Terms appearing in other term definitions
          2. Semantic similarity between definitions
        """
        chunks = []
        for term, definition in self.terms_definitions.items():
            # Format: Just term and definition, nothing else
            chunk_text = f"{term}: {definition}"
            chunks.append(chunk_text)

        print(f"Created {len(chunks)} graph chunks")
        return chunks

    def compute_embeddings(self) -> None:
        """Pre-compute embeddings for all definitions"""
        print("Computing definition embeddings...")
        terms = list(self.terms_definitions.keys())
        definitions = [self.terms_definitions[t] for t in terms]

        embeddings = EMBED_MODEL.encode(definitions, normalize_embeddings=True)

        for term, embedding in zip(terms, embeddings):
            self.term_embeddings[term] = embedding

        print(f"Computed embeddings for {len(self.term_embeddings)} terms")

    def find_semantically_similar(
        self, term: str, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Find terms with semantically similar definitions"""
        if term not in self.term_embeddings:
            return []

        query_embedding = self.term_embeddings[term]
        similarities = []

        for other_term, other_embedding in self.term_embeddings.items():
            if other_term == term:
                continue

            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(query_embedding, other_embedding))

            similarities.append(
                {
                    "term": other_term,
                    "definition": self.terms_definitions[other_term],
                    "similarity": similarity,
                }
            )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    def find_related_by_definition(self, term: str) -> List[str]:
        """Find terms that appear in this term's definition or vice versa"""
        if term not in self.terms_definitions:
            return []

        definition = self.terms_definitions[term].lower()
        related = []

        for other_term in self.terms_definitions.keys():
            if other_term == term:
                continue

            # Check if other_term appears in this definition
            if other_term in definition:
                related.append(other_term)
                continue

            # Check if this term appears in other's definition
            other_def = self.terms_definitions[other_term].lower()
            if term in other_def:
                related.append(other_term)

        return related

    def save(self) -> None:
        """Save graph data"""
        terms_path = os.path.join(self.working_dir, "graph_terms.json")
        with open(terms_path, "w", encoding="utf-8") as f:
            json.dump(self.terms_definitions, f, indent=2, ensure_ascii=False)

        # Save embeddings as numpy array
        if self.term_embeddings:
            embeddings_path = os.path.join(self.working_dir, "term_embeddings.npz")
            terms_list = list(self.term_embeddings.keys())
            embeddings_array = np.array([self.term_embeddings[t] for t in terms_list])
            np.savez(embeddings_path, terms=terms_list, embeddings=embeddings_array)
            print(f"Graph data saved to {self.working_dir}")

    def load(self) -> bool:
        """Load graph data"""
        terms_path = os.path.join(self.working_dir, "graph_terms.json")
        if os.path.exists(terms_path):
            with open(terms_path, "r", encoding="utf-8") as f:
                self.terms_definitions = json.load(f)

            # Load embeddings
            embeddings_path = os.path.join(self.working_dir, "term_embeddings.npz")
            if os.path.exists(embeddings_path):
                data = np.load(embeddings_path, allow_pickle=True)
                terms_list = data["terms"]
                embeddings_array = data["embeddings"]
                self.term_embeddings = {
                    term: embedding
                    for term, embedding in zip(terms_list, embeddings_array)
                }

            print(f"Loaded {len(self.terms_definitions)} terms from graph")
            return True
        return False


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
class GraphRAGBuilder:
    """Main class that builds the dictionary and GraphRAG"""

    def __init__(self):
        self.dictionary_manager = DictionaryManager()
        self.graph_manager = GraphManager()
        self.processed_chunks = []
        self.terms_with_definitions = {}

    async def build_graph(self, book_text: str, term_extractor=extract_technical_terms):
        """Build the dictionary and GraphRAG from book text"""
        print("Starting GraphRAG build process")
        print("=" * 60)

        # Step 1: Extract technical terms
        print("\nSTEP 1: EXTRACTING TECHNICAL TERMS")
        print("-" * 30)
        technical_terms = term_extractor(book_text)
        print(f"Extracted {len(technical_terms)} technical terms")

        if not technical_terms:
            print("No technical terms found. Cannot build graph without terms.")
            return False

        # Step 2: Create chunks from the book
        print("\nSTEP 2: CREATING BOOK CHUNKS")
        print("-" * 30)
        chunker = BookChunker(chunk_size=1000, overlap=200)
        chunks = chunker.create_chunks(book_text)

        # Step 3: Process chunks to find terms and generate definitions
        print("\nSTEP 3: PROCESSING CHUNKS")
        print("-" * 30)
        chunk_processor = ChunkProcessor()

        if MAX_CONCURRENT_DEFINITIONS <= 1:
            for i, chunk in enumerate(chunks):
                print(f"\nChunk {i+1}/{len(chunks)}:")
                result = await chunk_processor.process_chunk(chunk, technical_terms)
                self.processed_chunks.append(result)
        else:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEFINITIONS)

            async def process_chunk_with_semaphore(chunk, idx):
                async with semaphore:
                    print(f"\nChunk {idx+1}/{len(chunks)}:")
                    result = await chunk_processor.process_chunk(chunk, technical_terms)
                    return result

            tasks = [
                process_chunk_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)
            ]
            self.processed_chunks = await asyncio.gather(*tasks)

        # Step 4: Merge definitions from all chunks
        print("\nSTEP 4: MERGING DEFINITIONS")
        print("-" * 30)
        self._merge_definitions(chunk_processor)

        # Step 5: Add to dictionary
        print("\nSTEP 5: ADDING TO DICTIONARY")
        print("-" * 30)
        self.dictionary_manager.add_terms(self.terms_with_definitions)

        # Step 6: Add to graph and compute embeddings
        print("\nSTEP 6: BUILDING GRAPH")
        print("-" * 30)
        self.graph_manager.add_terms(self.terms_with_definitions)
        self.graph_manager.compute_embeddings()

        # Step 7: Create GraphRAG (optional - for advanced graph features)
        print("\nSTEP 7: CREATING GRAPHRAG (optional)")
        print("-" * 30)
        graph_chunks = self.graph_manager.create_graph_chunks()
        await self._build_graphrag(graph_chunks)

        return True

    def _merge_definitions(self, chunk_processor: ChunkProcessor) -> None:
        """Merge definitions from different chunks"""
        print(f"Merging definitions from {len(self.processed_chunks)} chunks...")

        term_definitions = defaultdict(list)
        for chunk_id, definitions in chunk_processor.chunk_definitions.items():
            for term, definition in definitions.items():
                term_definitions[term].append(definition)

        for term, definitions in term_definitions.items():
            if len(definitions) == 1:
                self.terms_with_definitions[term] = definitions[0]
            else:
                # Use first definition (could use most common or merge)
                self.terms_with_definitions[term] = definitions[0]

        print(f"Merged {len(self.terms_with_definitions)} definitions")

    async def _build_graphrag(self, graph_chunks: List[str]) -> bool:
        """Build the GraphRAG knowledge graph (optional for advanced features)"""
        try:
            rag = GraphRAG(
                working_dir=WORKING_DIR,
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
                best_model_max_token_size=MAX_TOKEN,
                cheap_model_max_token_size=MAX_TOKEN,
                convert_response_to_json_func=lambda x: (
                    json_repair.loads(x) if x else {}
                ),
                embedding_func_max_async=3,
                embedding_batch_num=4,
            )
            await rag.ainsert(graph_chunks)
            print("GraphRAG knowledge graph built successfully")
            return True
        except Exception as e:
            print(f"GraphRAG build failed (non-critical): {e}")
            return False

    def save_results(self) -> None:
        """Save the built dictionary and graph"""
        self.dictionary_manager.save()
        self.graph_manager.save()

        # Save processing stats
        stats_path = os.path.join(WORKING_DIR, "processing_stats.json")
        stats = {
            "total_chunks": len(self.processed_chunks),
            "total_terms": len(self.terms_with_definitions),
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {WORKING_DIR}")


# ============================================================================
# QUERY SYSTEM
# ============================================================================
class BookDictionaryQuery:
    """Query interface for dictionary and GraphRAG"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.dictionary = DictionaryManager(working_dir)
        self.graph = GraphManager(working_dir)
        self.load_all_data()

    def load_all_data(self):
        """Load dictionary and graph data"""
        print(f"Loading data from {self.working_dir}...")

        dict_loaded = self.dictionary.load()
        graph_loaded = self.graph.load()

        if not dict_loaded:
            print("No dictionary data found")
        if not graph_loaded:
            print("No graph data found")

    def dict_lookup(self, term: str, show_similar: bool = True) -> Dict[str, Any]:
        """
        Dictionary lookup with character-based similarity

        Args:
            term: Term to look up
            show_similar: If exact match not found, show similar terms
        """
        # Try exact match first
        exact_match = self.dictionary.lookup_exact(term)

        if exact_match:
            return {
                "found": True,
                "exact_match": True,
                "term": term.lower(),
                "definition": exact_match,
            }

        # No exact match, find similar
        if show_similar:
            similar_terms = self.dictionary.lookup_similar(term, top_n=5)
            return {
                "found": False,
                "exact_match": False,
                "query": term,
                "similar_terms": similar_terms,
                "message": f"No exact match for '{term}'. Did you mean one of these?",
            }
        else:
            return {
                "found": False,
                "exact_match": False,
                "query": term,
                "message": f"Term '{term}' not found in dictionary",
            }

    def graph_query(self, term: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Graph query with semantic similarity

        Args:
            term: Term to query
            top_n: Number of similar terms to return
        """
        # First check if term exists
        if term.lower() not in self.graph.terms_definitions:
            return {
                "found": False,
                "term": term,
                "message": f"Term '{term}' not found in graph",
            }

        term_normalized = term.lower()

        # Get semantically similar terms (based on definition similarity)
        semantic_similar = self.graph.find_semantically_similar(term_normalized, top_n)

        # Get related terms (terms that appear in definitions)
        related_terms = self.graph.find_related_by_definition(term_normalized)

        return {
            "found": True,
            "term": term_normalized,
            "definition": self.graph.terms_definitions[term_normalized],
            "semantically_similar": semantic_similar,
            "related_by_definition": related_terms,
        }

    def combined_query(self, query: str) -> Dict[str, Any]:
        """
        Combined query: dictionary lookup + graph relationships
        """
        # Step 1: Dictionary lookup
        dict_result = self.dict_lookup(query, show_similar=True)

        # Step 2: If exact match, also get graph data
        if dict_result.get("exact_match"):
            graph_result = self.graph_query(dict_result["term"])
            return {
                "method": "combined",
                "dictionary": dict_result,
                "graph": graph_result,
            }
        else:
            # No exact match, just return similar terms
            return {"method": "dictionary_only", "dictionary": dict_result}


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py build [book_file] - Build dictionary and graph")
        print(
            "  python script.py dict [term] - Dictionary lookup (character similarity)"
        )
        print(
            "  python script.py graph [term] [top_n] - Graph query (semantic similarity)"
        )
        print("  python script.py query [term] - Combined query")
        print("\nExamples:")
        print("  python script.py build book.txt")
        print("  python script.py dict impedance")
        print("  python script.py graph impedance 10")
        print("  python script.py query impedance")
        return

    command = sys.argv[1].lower()

    if command == "build":
        if len(sys.argv) < 3:
            print("Please specify a book file")
            return

        book_file = sys.argv[2]
        if not os.path.exists(book_file):
            print(f"Book file not found: {book_file}")
            return

        try:
            with open(book_file, encoding="utf-8") as f:
                book_text = f.read()
            print(f"Loaded book: {book_file} ({len(book_text)} characters)")
        except Exception as e:
            print(f"Error reading book: {e}")
            return

        builder = GraphRAGBuilder()
        success = await builder.build_graph(book_text)

        if success:
            builder.save_results()
            print("\nBuild completed successfully")

    elif command == "dict":
        if len(sys.argv) < 3:
            print("Usage: python script.py dict <term>")
            return

        term = sys.argv[2]
        query_system = BookDictionaryQuery()

        if not query_system.dictionary.dictionary:
            print(
                "No dictionary data. Build first with: python script.py build [book_file]"
            )
            return

        result = query_system.dict_lookup(term)

        print("\n" + "=" * 60)
        print("DICTIONARY LOOKUP (Character Similarity)")
        print("=" * 60)

        if result["exact_match"]:
            print(f"Term: {result['term']}")
            print(f"Definition: {result['definition']}")
        else:
            print(f"Query: {result['query']}")
            print(f"{result['message']}\n")

            if result.get("similar_terms"):
                for i, similar in enumerate(result["similar_terms"], 1):
                    print(
                        f"{i}. {similar['term']} (similarity: {similar['similarity']:.2f})"
                    )
                    print(f"   {similar['definition']}\n")

    elif command == "graph":
        if len(sys.argv) < 3:
            print("Usage: python script.py graph <term> [top_n]")
            return

        term = sys.argv[2]
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

        query_system = BookDictionaryQuery()

        if not query_system.graph.terms_definitions:
            print("No graph data. Build first with: python script.py build [book_file]")
            return

        result = query_system.graph_query(term, top_n)

        print("\n" + "=" * 60)
        print("GRAPH QUERY (Semantic Similarity)")
        print("=" * 60)

        if not result["found"]:
            print(f"{result['message']}")
            return

        print(f"Term: {result['term']}")
        print(f"Definition: {result['definition']}\n")

        # Semantically similar terms
        if result.get("semantically_similar"):
            print(f"SEMANTICALLY SIMILAR TERMS (based on definition similarity):")
            print("-" * 60)
            for i, similar in enumerate(result["semantically_similar"], 1):
                print(
                    f"\n{i}. {similar['term']} (similarity: {similar['similarity']:.3f})"
                )
                print(f"   {similar['definition']}")

        # Related by definition
        if result.get("related_by_definition"):
            print(f"\n\nRELATED TERMS (appearing in definitions):")
            print("-" * 60)
            for term in result["related_by_definition"]:
                print(f"  - {term}")

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python script.py query <term>")
            return

        term = sys.argv[2]
        query_system = BookDictionaryQuery()

        if not query_system.dictionary.dictionary:
            print("No data. Build first with: python script.py build [book_file]")
            return

        result = query_system.combined_query(term)

        print("\n" + "=" * 60)
        print("COMBINED QUERY")
        print("=" * 60)

        # Dictionary results
        dict_result = result["dictionary"]
        print("\nDICTIONARY LOOKUP:")
        print("-" * 60)

        if dict_result["exact_match"]:
            print(f"Term: {dict_result['term']}")
            print(f"Definition: {dict_result['definition']}")
        else:
            print(f"Query: {dict_result['query']}")
            print(f"{dict_result['message']}\n")

            if dict_result.get("similar_terms"):
                for i, similar in enumerate(dict_result["similar_terms"][:3], 1):
                    print(
                        f"{i}. {similar['term']} (similarity: {similar['similarity']:.2f})"
                    )
                    print(f"   {similar['definition']}\n")

        # Graph results (if exact match found)
        if result["method"] == "combined" and result.get("graph"):
            graph_result = result["graph"]

            if graph_result["found"]:
                print("\nGRAPH ANALYSIS:")
                print("-" * 60)

                # Semantically similar
                if graph_result.get("semantically_similar"):
                    print("\nSemantically similar terms:")
                    for i, similar in enumerate(
                        graph_result["semantically_similar"][:5], 1
                    ):
                        print(
                            f"  {i}. {similar['term']} (similarity: {similar['similarity']:.3f})"
                        )

                # Related terms
                if graph_result.get("related_by_definition"):
                    print("\nRelated terms (in definitions):")
                    for term in graph_result["related_by_definition"][:10]:
                        print(f"  - {term}")

    else:
        print(f"Unknown command: {command}")
        print("Use: build, dict, graph, or query")


if __name__ == "__main__":
    asyncio.run(main())
