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
MAX_CONCURRENT_DEFINITIONS = 15  # Set to 1 for synchronous, higher for concurrent
WORKING_DIR = "./graph_rag_cache"
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKEN = 2000
DEFINITION_SIMILARITY_THRESHOLD = (
    0.7  # If definitions are > threshold, consider them similar
)

# Initialize embedding model
try:
    EMBED_MODEL = SentenceTransformer("models/embeddings", local_files_only=True)
    print("✅ Local embedding model loaded")
except:
    try:
        EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("✅ Fallback embedding model loaded")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
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

    Args:
        text: Book text to extract terms from
        **kwargs: Additional parameters (not used now)

    Returns:
        Dict[str, str]: Dictionary of terms to definitions (empty for now)
    """
    # This is a placeholder - in reality, we would get a hardcoded list later
    # For now, we'll extract some basic terms as an example
    terms = {}

    # Simple pattern matching for technical terms (placeholder)
    patterns = [
        r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b",  # hyphenated terms
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # proper noun phrases
        r"\b[a-z]+[A-Z][a-z]*\b",  # camelCase terms
        r"\b[A-Z]{2,}\b",  # acronyms
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean the term
            term = match.lower().strip()
            if len(term) > 3:
                terms[term] = ""  # Empty definition for now

    print(f"🔍 Extracted {len(terms)} placeholder technical terms")
    return terms


# ============================================================================
# MODEL FOR LLM OUTPUT
# ============================================================================
class TermDefinition(BaseModel):
    """Pydantic model to ensure LLM returns correct format"""

    term: str = Field(..., description="The technical term being defined")
    definition: str = Field(
        ..., description="Clear, concise definition of the term (1-2 sentences)"
    )


class TermDefinitionsResponse(BaseModel):
    """Pydantic model for the full response"""

    definitions: List[TermDefinition] = Field(
        ..., description="List of term definitions found in the text chunk"
    )


# ============================================================================
# BOOK CHUNKER
# ============================================================================
class BookChunker:
    """Split book into manageable chunks for processing"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Try to get tiktoken for accurate token counting
        try:
            import tiktoken

            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except:
            self.tokenizer = None
            self.use_tiktoken = False
            print("⚠️ tiktoken not available, using approximate word-based chunking")

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from the book text"""
        print(f"📖 Creating chunks from book ({len(text)} characters)...")
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
                    "char_start": self._find_char_position(text, chunk_text, i > 0),
                    "char_end": self._find_char_position(text, chunk_text, i > 0)
                    + len(chunk_text),
                }
            )
        print(f"✅ Created {len(chunks)} token-based chunks")
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
                    "char_start": self._find_char_position(text, chunk_text, i > 0),
                    "char_end": self._find_char_position(text, chunk_text, i > 0)
                    + len(chunk_text),
                }
            )
        print(f"✅ Created {len(chunks)} word-based chunks")
        return chunks

    def _find_char_position(
        self, full_text: str, chunk_text: str, is_continuation: bool
    ) -> int:
        """Find character position of chunk in full text"""
        if not is_continuation:
            return 0
        # Find first few words to locate position
        first_words = " ".join(chunk_text.split()[:5])
        position = full_text.find(first_words)
        return max(0, position)


# ============================================================================
# GRAPH MANAGER
# ============================================================================
class GraphManager:
    """Manage the knowledge graph and entity relationships"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.entity_definitions = {}
        self.entity_contexts = defaultdict(list)
        os.makedirs(working_dir, exist_ok=True)

    def add_or_update_entities(
        self,
        terms_with_definitions: Dict[str, str],
        terms_with_contexts: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Add or update entities in the graph"""
        print(f"🕸️ Processing {len(terms_with_definitions)} entities for graph...")
        updates = []
        new_entities = []
        similar_entities = []

        for term, definition in terms_with_definitions.items():
            contexts = terms_with_contexts.get(term, [])

            if term in self.entity_definitions:
                # Entity exists - check similarity
                existing_def = self.entity_definitions[term]
                similarity = self._calculate_similarity(definition, existing_def)

                if similarity < DEFINITION_SIMILARITY_THRESHOLD:
                    # Definitions are different enough to append
                    updated_def = self._merge_definitions(existing_def, definition)
                    self.entity_definitions[term] = updated_def
                    self.entity_contexts[term].extend(contexts)
                    similar_entities.append(
                        {
                            "term": term,
                            "existing_definition": existing_def,
                            "new_definition": definition,
                            "similarity": similarity,
                            "updated_definition": updated_def,
                        }
                    )
                # If very similar (>= threshold), do nothing
            else:
                # New entity
                self.entity_definitions[term] = definition
                self.entity_contexts[term] = contexts.copy()
                new_entities.append(term)
                updates.append(
                    {"term": term, "definition": definition, "contexts": contexts}
                )

        print(f"✅ New entities: {len(new_entities)}")
        print(f"🔄 Updated entities: {len(similar_entities)}")
        return {
            "new_entities": new_entities,
            "updated_entities": similar_entities,
            "total_entities": len(self.entity_definitions),
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two definitions"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _merge_definitions(self, def1: str, def2: str) -> str:
        """Merge two definitions when they are sufficiently different"""
        # Simple merge - could be improved later
        return f"{def1} {def2}"

    def create_graph_chunks(self) -> List[str]:
        """Create chunks for GraphRAG from stored entities"""
        chunks = []
        for term, definition in self.entity_definitions.items():
            contexts = self.entity_contexts.get(term, [])
            chunk_lines = [f"**{term.title()}**", f"Definition: {definition}"]
            if contexts:
                chunk_lines.append("Contexts:")
                for i, context in enumerate(contexts[:2], 1):
                    clean_context = context.strip()[:200]
                    chunk_lines.append(f"{i}. {clean_context}")
            chunk_text = "\n".join(chunk_lines)
            chunks.append(chunk_text)
        print(f"📚 Created {len(chunks)} graph chunks")
        return chunks


# ============================================================================
# CHUNK PROCESSOR
# ============================================================================
class ChunkProcessor:
    """Process individual chunks to find terms and generate definitions"""

    def __init__(self):
        self.chunk_terms = {}  # chunk_id -> terms found
        self.chunk_definitions = {}  # chunk_id -> {term: definition}
        self.global_terms = set()  # all unique terms found

    async def process_chunk(
        self, chunk: Dict[str, Any], given_terms: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Process a single chunk to find terms and generate definitions

        Args:
            chunk: Dictionary containing chunk information
            given_terms: Dictionary of terms to look for (term: empty definition)

        Returns:
            Dictionary with processing results
        """
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
        print(
            f"🔍 Processing chunk {chunk_id} ({chunk.get('token_count', chunk.get('word_count', 0))} tokens/words)"
        )

        # Step 1: Find which given terms are present in this chunk
        present_terms = []
        for term in given_terms.keys():
            if term.lower() in chunk_text.lower():
                present_terms.append(term)

        print(f"  📋 Found {len(present_terms)} given terms in chunk")

        # Step 2: Generate definitions for terms in this chunk
        definitions = {}
        if present_terms:
            definitions = await self._generate_chunk_definitions(
                chunk_text, present_terms, chunk_id
            )

        # Store results
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
        # Process in smaller batches to avoid overwhelming the LLM
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
                    f"    ✅ Chunk {chunk_id}: Generated definitions for batch {i//batch_size + 1}"
                )
            except Exception as e:
                print(
                    f"    ❌ Failed to generate definitions for chunk {chunk_id}, batch {i//batch_size + 1}: {e}"
                )
                # Fallback definitions
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
        # Split response into sections
        sections = re.split(r"\n(?=TERM:)", response)
        for section in sections:
            # Extract term and definition
            term_match = re.search(r"TERM:\s*(.+?)(?:\n|$)", section, re.IGNORECASE)
            def_match = re.search(
                r"DEFINITION:\s*(.+?)(?=\nTERM:|\n|$)",
                section,
                re.IGNORECASE | re.DOTALL,
            )
            if term_match and def_match:
                term = term_match.group(1).strip().lower()
                definition = def_match.group(1).strip()
                # Clean up the definition
                definition = re.sub(r"\s+", " ", definition)
                if definition and not definition.endswith((".", "!", "?")):
                    definition += "."
                definitions[term] = definition

        # Ensure we got definitions for the expected terms
        missing_terms = [t for t in expected_terms if t.lower() not in definitions]
        if missing_terms:
            print(f"    ⚠️ Missing definitions for: {missing_terms}")

        return definitions


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
class GraphRAGBuilder:
    """Main class that builds the GraphRAG knowledge graph"""

    def __init__(self):
        self.graph_manager = GraphManager()
        self.processed_chunks = []
        self.terms_with_definitions = {}
        self.terms_with_contexts = defaultdict(list)

    async def build_graph(self, book_text: str, term_extractor=extract_technical_terms):
        """
        Build the GraphRAG knowledge graph from book text

        Args:
            book_text: Full book text
            term_extractor: Function to extract technical terms

        Returns:
            Boolean indicating success
        """
        print("🚀 Starting GraphRAG build process")
        print("=" * 60)

        # Step 1: Extract technical terms (placeholder for now)
        print("\nSTEP 1: EXTRACTING TECHNICAL TERMS")
        print("-" * 30)
        technical_terms = term_extractor(book_text)
        print(f"✅ Extracted {len(technical_terms)} technical terms")

        if not technical_terms:
            print("⚠️ No technical terms found. Cannot build graph without terms.")
            return False

        # Step 2: Create chunks from the entire book
        print("\nSTEP 2: CREATING BOOK CHUNKS")
        print("-" * 30)
        chunker = BookChunker(chunk_size=1000, overlap=200)
        chunks = chunker.create_chunks(book_text)

        # Step 3: Process chunks to find terms and generate definitions
        print("\nSTEP 3: PROCESSING CHUNKS")
        print("-" * 30)
        chunk_processor = ChunkProcessor()

        # Process chunks with concurrency control
        if MAX_CONCURRENT_DEFINITIONS <= 1:
            # Synchronous processing
            for i, chunk in enumerate(chunks):
                print(f"\n📖 Chunk {i+1}/{len(chunks)}:")
                result = await chunk_processor.process_chunk(chunk, technical_terms)
                self.processed_chunks.append(result)
                print(
                    f"  📊 Chunk summary: {result['terms_found']} terms, {result['definitions_generated']} definitions"
                )
        else:
            # Concurrent processing
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEFINITIONS)

            async def process_chunk_with_semaphore(chunk, idx):
                async with semaphore:
                    print(f"\n📖 Chunk {idx+1}/{len(chunks)}:")
                    result = await chunk_processor.process_chunk(chunk, technical_terms)
                    print(
                        f"  📊 Chunk summary: {result['terms_found']} terms, {result['definitions_generated']} definitions"
                    )
                    return result

            tasks = [
                process_chunk_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)
            ]
            self.processed_chunks = await asyncio.gather(*tasks)

        # Step 4: Compile contexts for all terms
        print("\nSTEP 4: COMPILING TERM CONTEXTS")
        print("-" * 30)
        self._compile_term_contexts(chunks, chunk_processor)

        # Step 5: Merge definitions from all chunks
        print("\nSTEP 5: MERGING DEFINITIONS")
        print("-" * 30)
        self._merge_definitions(chunk_processor)

        # Step 6: Add to graph
        print("\nSTEP 6: ADDING TO GRAPH")
        print("-" * 30)
        graph_result = self.graph_manager.add_or_update_entities(
            self.terms_with_definitions, self.terms_with_contexts
        )

        # Step 7: Create GraphRAG
        print("\nSTEP 7: CREATING GRAPHRAG")
        print("-" * 30)
        graph_chunks = self.graph_manager.create_graph_chunks()
        return await self._build_graphrag(graph_chunks)

    def _compile_term_contexts(
        self, chunks: List[Dict[str, Any]], chunk_processor: ChunkProcessor
    ) -> None:
        """Compile contexts for all terms from all chunks"""
        print(f"📚 Compiling contexts for all terms from {len(chunks)} chunks...")

        for chunk_id, terms in chunk_processor.chunk_terms.items():
            chunk = next((c for c in chunks if c["id"] == chunk_id), None)
            if not chunk:
                continue

            chunk_text = chunk["text"]
            for term in terms:
                # Create a focused context around the term
                context = self._extract_term_context(chunk_text, term)
                if context:
                    self.terms_with_contexts[term].append(context)

    def _extract_term_context(
        self, chunk_text: str, term: str, context_size: int = 200
    ) -> str:
        """Extract focused context around a term"""
        term_lower = term.lower()
        text_lower = chunk_text.lower()
        # Find the term in the text
        pos = text_lower.find(term_lower)
        if pos == -1:
            return ""
        # Extract context around the term
        start = max(0, pos - context_size // 2)
        end = min(len(chunk_text), pos + len(term) + context_size // 2)
        context = chunk_text[start:end].strip()
        # Add ellipsis if we truncated
        if start > 0:
            context = "..." + context
        if end < len(chunk_text):
            context = context + "..."
        return context

    def _merge_definitions(self, chunk_processor: ChunkProcessor) -> None:
        """Merge definitions from different chunks"""
        print(f"🔄 Merging definitions from {len(self.processed_chunks)} chunks...")

        # Group definitions by term
        term_definitions = defaultdict(list)
        for chunk_id, definitions in chunk_processor.chunk_definitions.items():
            for term, definition in definitions.items():
                term_definitions[term].append(definition)

        # Select the best definition for each term
        for term, definitions in term_definitions.items():
            if len(definitions) == 1:
                # Single definition, use as-is
                self.terms_with_definitions[term] = definitions[0]
            else:
                # Multiple definitions, use the first one for now (could be improved)
                self.terms_with_definitions[term] = definitions[0]

        print(f"✅ Merged {len(self.terms_with_definitions)} definitions")

    async def _build_graphrag(self, graph_chunks: List[str]) -> bool:
        """Build the GraphRAG knowledge graph (async version)"""
        try:
            # Initialize GraphRAG with more robust JSON handling
            rag = GraphRAG(
                working_dir=WORKING_DIR,
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
                best_model_max_token_size=MAX_TOKEN,
                cheap_model_max_token_size=MAX_TOKEN,
                # More robust JSON conversion function
                convert_response_to_json_func=lambda x: (
                    json_repair.loads(x) if x else {}
                ),
                embedding_func_max_async=3,
                embedding_batch_num=4,
            )
            # Insert data using async insert
            await rag.ainsert(graph_chunks)  # This is the key fix
            print("✅ GraphRAG knowledge graph built successfully!")
            return True
        except Exception as e:
            print(f"❌ GraphRAG build failed: {e}")
            return False

    def save_results(self) -> None:
        """Save the built graph and related data"""
        os.makedirs(WORKING_DIR, exist_ok=True)

        # Save entity definitions
        definitions_path = os.path.join(WORKING_DIR, "entity_definitions.json")
        with open(definitions_path, "w", encoding="utf-8") as f:
            json.dump(
                self.graph_manager.entity_definitions, f, indent=2, ensure_ascii=False
            )

        # Save entity contexts
        contexts_path = os.path.join(WORKING_DIR, "entity_contexts.json")
        with open(contexts_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: list(v) for k, v in self.graph_manager.entity_contexts.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save processing stats
        stats_path = os.path.join(WORKING_DIR, "processing_stats.json")
        stats = {
            "total_chunks": len(self.processed_chunks),
            "total_terms": len(self.terms_with_definitions),
            "total_entities": len(self.graph_manager.entity_definitions),
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved to {WORKING_DIR}")


# ============================================================================
# QUERY SYSTEM
# ============================================================================


class BookDictionaryQuery:
    """Query interface for dictionary and GraphRAG"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.vocabulary = {}
        self.graph_rag = None
        self.load_all_data()

    def load_all_data(self):
        """Load definitions and contexts"""
        print(f"📚 Loading dictionary data from {self.working_dir}...")

        # Load definitions
        definitions_path = os.path.join(self.working_dir, "entity_definitions.json")
        if os.path.exists(definitions_path):
            with open(definitions_path, "r", encoding="utf-8") as f:
                definitions = json.load(f)
            print(f"✅ Loaded {len(definitions)} definitions")
        else:
            definitions = {}
            print("⚠️ No definitions file found")

        # Load contexts
        contexts_path = os.path.join(self.working_dir, "entity_contexts.json")
        if os.path.exists(contexts_path):
            with open(contexts_path, "r", encoding="utf-8") as f:
                contexts = json.load(f)
            print(f"✅ Loaded contexts for {len(contexts)} terms")
        else:
            contexts = {}
            print("⚠️ No contexts file found")

        # Build vocabulary
        self.vocabulary = {}
        for term, definition in definitions.items():
            term_contexts = contexts.get(term, [])
            self.vocabulary[term] = {
                "definition": definition,
                "contexts": term_contexts,
            }

        print(f"📊 Dictionary ready: {len(self.vocabulary)} terms")

    def load_graphrag(self) -> bool:
        """Load GraphRAG if available"""
        try:
            self.graph_rag = GraphRAG(
                working_dir=self.working_dir,
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
            )
            print("✅ GraphRAG loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ GraphRAG not available: {e}")
            return False

    def dict_lookup(self, term: str) -> Dict[str, Any]:
        """Dictionary lookup for a term"""
        term_lower = term.lower().strip()

        if term_lower in self.vocabulary:
            vocab_entry = self.vocabulary[term_lower]
            return {
                "found": True,
                "term": term_lower,
                "definition": vocab_entry["definition"],
                "contexts": vocab_entry["contexts"][:3],
                "context_count": len(vocab_entry["contexts"]),
            }
        else:
            return {
                "found": False,
                "term": term_lower,
                "message": f"Term '{term}' not found in dictionary",
            }

    async def query_graphrag_raw(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Query GraphRAG and return raw entities and relationships"""
        if not self.graph_rag:
            if not self.load_graphrag():
                return {"entities": [], "relationships": []}

        try:
            # Access the graph directly
            graph_file = os.path.join(
                self.working_dir, "graph_chunk_entity_relation.graphml"
            )
            if not os.path.exists(graph_file):
                return {"entities": [], "relationships": []}

            import networkx as nx

            G = nx.read_graphml(graph_file)

            entities = []
            relationships = []

            # Search for matching nodes
            query_lower = query.lower()
            for node, data in G.nodes(data=True):
                node_lower = str(node).lower()
                description = data.get("description", "").lower()

                if (
                    query_lower in node_lower
                    or node_lower in query_lower
                    or query_lower in description
                ):

                    # Calculate similarity
                    if query_lower == node_lower:
                        similarity = 1.0
                    elif query_lower in node_lower or node_lower in query_lower:
                        similarity = 0.9
                    else:
                        similarity = 0.7

                    entities.append(
                        {
                            "name": node,
                            "description": data.get("description", ""),
                            "type": data.get("entity_type", ""),
                            "similarity": similarity,
                        }
                    )

                if len(entities) >= max_results:
                    break

            # Get relationships for found entities
            if entities:
                entity_names = {e["name"] for e in entities}
                for source, target, data in G.edges(data=True):
                    if source in entity_names or target in entity_names:
                        relationships.append(
                            {
                                "source": source,
                                "target": target,
                                "description": data.get("description", ""),
                                "weight": float(data.get("weight", 1.0)),
                            }
                        )
                        if len(relationships) >= max_results * 2:
                            break

            return {"entities": entities, "relationships": relationships}

        except Exception as e:
            print(f"Error in GraphRAG query: {e}")
            return {"entities": [], "relationships": []}

    async def search_dict_n_hop(
        self, term: str, n_hops: int = 2, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Dictionary-based n-hop search: Find term in dict, then traverse NetworkX graph

        Args:
            term: Term to search in dictionary
            n_hops: Number of hops to traverse
            max_results: Maximum results per hop
        """
        # Step 1: Find in dictionary
        dict_result = self.dict_lookup(term)

        if not dict_result["found"]:
            return {
                "error": f"Term '{term}' not found in dictionary",
                "start_method": "dict",
                "hops": [],
            }

        # Step 2: Do n-hop traversal from this term
        graph_file = os.path.join(
            self.working_dir, "graph_chunk_entity_relation.graphml"
        )
        if not os.path.exists(graph_file):
            return {
                "error": "Graph file not found",
                "dict_result": dict_result,
                "hops": [],
            }

        try:
            import networkx as nx

            G = nx.read_graphml(graph_file)

            # Find the node in graph (case-insensitive match)
            term_lower = term.lower()
            start_node = None
            for node in G.nodes():
                if term_lower == str(node).lower():
                    start_node = node
                    break

            if not start_node:
                return {
                    "error": f"Term '{term}' found in dict but not in graph",
                    "dict_result": dict_result,
                    "hops": [],
                }

            # Traverse n-hops
            visited = {start_node}
            hops_data = [
                {
                    "hop": 0,
                    "entities": [
                        {
                            "name": start_node,
                            "description": G.nodes[start_node].get("description", ""),
                            "type": G.nodes[start_node].get("entity_type", ""),
                        }
                    ],
                }
            ]

            current_nodes = [start_node]

            for hop in range(1, n_hops + 1):
                next_nodes = []
                hop_entities = []
                hop_relationships = []

                for node in current_nodes:
                    neighbors = list(G.neighbors(node))

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_nodes.append(neighbor)

                            hop_entities.append(
                                {
                                    "name": neighbor,
                                    "description": G.nodes[neighbor].get(
                                        "description", ""
                                    ),
                                    "type": G.nodes[neighbor].get("entity_type", ""),
                                    "from_node": node,
                                }
                            )

                            edge_data = G.get_edge_data(node, neighbor)
                            if edge_data:
                                hop_relationships.append(
                                    {
                                        "source": node,
                                        "target": neighbor,
                                        "description": edge_data.get("description", ""),
                                        "weight": float(edge_data.get("weight", 1.0)),
                                    }
                                )

                            if len(hop_entities) >= max_results:
                                break

                    if len(hop_entities) >= max_results:
                        break

                hops_data.append(
                    {
                        "hop": hop,
                        "entities": hop_entities,
                        "relationships": hop_relationships,
                    }
                )

                current_nodes = next_nodes
                if not current_nodes:
                    break

            return {
                "start_term": term,
                "start_method": "dict",
                "dict_result": dict_result,
                "n_hops": n_hops,
                "total_visited": len(visited),
                "hops": hops_data,
            }

        except Exception as e:
            return {"error": str(e), "dict_result": dict_result, "hops": []}

    async def search_graphrag_n_hop(
        self, query: str, n_hops: int = 2, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        GraphRAG-based n-hop search: Use GraphRAG to find top match, then traverse graph

        Args:
            query: Query to search with GraphRAG
            n_hops: Number of hops to traverse
            max_results: Maximum results per hop
        """
        # Step 1: Use GraphRAG to find top match
        graphrag_result = await self.query_graphrag_raw(query, max_results=1)

        if not graphrag_result.get("entities"):
            return {
                "error": f"No entities found for query '{query}'",
                "start_method": "graphrag",
                "hops": [],
            }

        # Get top entity
        top_entity = graphrag_result["entities"][0]
        start_term = top_entity["name"]

        # Step 2: Do n-hop traversal from this entity
        graph_file = os.path.join(
            self.working_dir, "graph_chunk_entity_relation.graphml"
        )
        if not os.path.exists(graph_file):
            return {
                "error": "Graph file not found",
                "graphrag_result": graphrag_result,
                "hops": [],
            }

        try:
            import networkx as nx

            G = nx.read_graphml(graph_file)

            # Verify node exists
            if start_term not in G.nodes():
                return {
                    "error": f"Top entity '{start_term}' not found in graph",
                    "graphrag_result": graphrag_result,
                    "hops": [],
                }

            # Traverse n-hops
            visited = {start_term}
            hops_data = [
                {
                    "hop": 0,
                    "entities": [
                        {
                            "name": start_term,
                            "description": G.nodes[start_term].get("description", ""),
                            "type": G.nodes[start_term].get("entity_type", ""),
                            "similarity": top_entity.get("similarity", 0),
                        }
                    ],
                }
            ]

            current_nodes = [start_term]

            for hop in range(1, n_hops + 1):
                next_nodes = []
                hop_entities = []
                hop_relationships = []

                for node in current_nodes:
                    neighbors = list(G.neighbors(node))

                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_nodes.append(neighbor)

                            hop_entities.append(
                                {
                                    "name": neighbor,
                                    "description": G.nodes[neighbor].get(
                                        "description", ""
                                    ),
                                    "type": G.nodes[neighbor].get("entity_type", ""),
                                    "from_node": node,
                                }
                            )

                            edge_data = G.get_edge_data(node, neighbor)
                            if edge_data:
                                hop_relationships.append(
                                    {
                                        "source": node,
                                        "target": neighbor,
                                        "description": edge_data.get("description", ""),
                                        "weight": float(edge_data.get("weight", 1.0)),
                                    }
                                )

                            if len(hop_entities) >= max_results:
                                break

                    if len(hop_entities) >= max_results:
                        break

                hops_data.append(
                    {
                        "hop": hop,
                        "entities": hop_entities,
                        "relationships": hop_relationships,
                    }
                )

                current_nodes = next_nodes
                if not current_nodes:
                    break

            return {
                "query": query,
                "start_term": start_term,
                "start_method": "graphrag",
                "graphrag_result": graphrag_result,
                "n_hops": n_hops,
                "total_visited": len(visited),
                "hops": hops_data,
            }

        except Exception as e:
            return {"error": str(e), "graphrag_result": graphrag_result, "hops": []}


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py build [book_file] - Build GraphRAG from book")
        print(
            "  python script.py query [term] X- Query X similar terms from built GraphRAG"
        )
        print(
            "  python script.py dict [query] - Search dictionary for term based on character similarity"
        )
        print("Example: python script.py build book2.txt")
        return

    command = sys.argv[1].lower()

    if command == "build":
        if len(sys.argv) < 3:
            print("❌ Please specify a book file")
            return

        book_file = sys.argv[2]
        if not os.path.exists(book_file):
            print(f"❌ Book file not found: {book_file}")
            return

        try:
            with open(book_file, encoding="utf-8") as f:
                book_text = f.read()
            print(f"✅ Loaded book: {book_file} ({len(book_text)} characters)")
        except Exception as e:
            print(f"❌ Error reading book: {e}")
            return

        builder = GraphRAGBuilder()
        success = await builder.build_graph(book_text)

        if success:
            builder.save_results()
            print("\n✅ GraphRAG build completed successfully!")
            print(
                f"   📚 Entity definitions saved to {os.path.join(WORKING_DIR, 'entity_definitions.json')}"
            )
            print(
                f"   📌 Entity contexts saved to {os.path.join(WORKING_DIR, 'entity_contexts.json')}"
            )
            print(
                f"   📊 Statistics saved to {os.path.join(WORKING_DIR, 'processing_stats.json')}"
            )
        else:
            print("\n❌ GraphRAG build failed")

    elif command == "query":
        # Query mode - return raw GraphRAG entities and relationships
        if len(sys.argv) < 3:
            print("Usage: python script.py query <term> [max_results]")
            print("Example: python script.py query impedance 10")
            return

        term = sys.argv[2]
        max_results = 5  # Default

        if len(sys.argv) >= 4:
            try:
                max_results = int(sys.argv[3])
            except ValueError:
                print(
                    f"Warning: Invalid max_results '{sys.argv[3]}', using default of 5"
                )

        print(f"🔍 Querying term: '{term}' (max results: {max_results})")

        # Initialize query system
        query_system = BookDictionaryQuery()

        # Check if we have data to query
        if not query_system.vocabulary:
            print(
                "❌ No vocabulary loaded. Please build the graph first with: python script.py build [book_file]"
            )
            return

        # Try to get raw GraphRAG data
        print(f"🕸️ Retrieving raw graph data for: '{term}'")

        try:
            raw_result = await query_system.query_graphrag_raw(
                term, max_results=max_results
            )

            print("\n" + "=" * 60)
            print(f"📖 Raw GraphRAG Results for '{term.upper()}'")
            print("=" * 60)

            # Display entities
            if raw_result.get("entities"):
                print(f"\n🔷 ENTITIES ({len(raw_result['entities'])} found):")
                for i, entity in enumerate(raw_result["entities"][:max_results], 1):
                    print(f"\n{i}. {entity['name'].upper()}")
                    if entity.get("description"):
                        print(f"   Description: {entity['description']}.")
                    print(f"   Similarity: {entity.get('similarity', 0):.3f}")
            else:
                print("\n🔷 ENTITIES: None found")

            # Display relationships
            if raw_result.get("relationships"):
                print(f"\n🔗 RELATIONSHIPS ({len(raw_result['relationships'])} found):")
                for i, rel in enumerate(raw_result["relationships"][:max_results], 1):
                    print(f"\n{i}. {rel['source']} → {rel['target']}")
                    if rel.get("description"):
                        print(f"   {rel['description']}")
                    if rel.get("weight"):
                        print(f"   Weight: {rel['weight']}")
            else:
                print("\n🔗 RELATIONSHIPS: None found")

            # Display text chunks
            if raw_result.get("text_chunks"):
                print(
                    f"\n📄 RELEVANT TEXT CHUNKS ({len(raw_result['text_chunks'])} found):"
                )
                for i, chunk in enumerate(raw_result["text_chunks"][:max_results], 1):
                    print(f"\n{i}. Chunk ID: {chunk.get('id', 'unknown')}")
                    print(f"   {chunk['content'][:300]}...")
                    if chunk.get("similarity"):
                        print(f"   Similarity: {chunk['similarity']:.3f}")
            else:
                print("\n📄 RELEVANT TEXT CHUNKS: None found")

        except Exception as e:
            print(f"❌ GraphRAG query failed: {e}")
            print(f"\nFalling back to dictionary lookup...")
            result = query_system.define_word(term)
            print(query_system.format_definition(result))

    elif command == "dict":
        # Dictionary lookup mode
        if len(sys.argv) < 3:
            print("Usage: python script.py dict <term>")
            print("Example: python script.py dict impedance")
            return

        term = sys.argv[2]
        print(f"📚 Looking up dictionary term: '{term}'")

        query_system = BookDictionaryQuery()

        if not query_system.vocabulary:
            print("❌ No vocabulary loaded. Please build the graph first")
            return

        result = query_system.dict_lookup(term)

        if result["found"]:
            print(f"\n📚 Dictionary Lookup:")
            print(f"   Term: {result['term']}")
            print(f"   Definition: {result['definition']}")
            print(f"   Contexts: {result['context_count']}")

            if result["contexts"]:
                print(f"\n   Sample context:")
                print(f"   {result['contexts'][0][:200]}...")
        else:
            print(f"\n❌ {result['message']}")

    elif command == "search":
        # Two modes: dict or query
        if len(sys.argv) < 4:
            print("Usage:")
            print("  python script.py search dict <term> [n_hops] [max_results]")
            print("  python script.py search query <query> [n_hops] [max_results]")
            print("Examples:")
            print("  python script.py search dict impedance 2 10")
            print("  python script.py search query 'electrical properties' 2 10")
            return

        search_mode = sys.argv[2].lower()
        search_term = sys.argv[3]
        n_hops = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        max_results = int(sys.argv[5]) if len(sys.argv) > 5 else 10

        query_system = BookDictionaryQuery()

        if search_mode == "dict":
            print(f"📚 Dictionary-based {n_hops}-hop search for: '{search_term}'")
            result = await query_system.search_dict_n_hop(
                search_term, n_hops, max_results
            )
        elif search_mode == "query":
            print(f"🕸️ GraphRAG-based {n_hops}-hop search for: '{search_term}'")
            result = await query_system.search_graphrag_n_hop(
                search_term, n_hops, max_results
            )
        else:
            print(f"❌ Invalid search mode: {search_mode}")
            print("Use 'dict' or 'query'")
            return

        # Display results
        print("\n" + "=" * 60)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print(f"Start method: {result['start_method']}")
        print(f"Start term: {result['start_term']}")
        print(f"Total entities visited: {result['total_visited']}")

        for hop_data in result["hops"]:
            print(f"\n🔷 HOP {hop_data['hop']}:")
            if hop_data.get("entities"):
                for entity in hop_data["entities"]:
                    print(f"  • {entity['name']}")
                    if entity.get("description"):
                        print(f"    {entity['description'][:100]}...")

            if hop_data.get("relationships"):
                print(f"  Relationships: {len(hop_data['relationships'])}")

    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())
