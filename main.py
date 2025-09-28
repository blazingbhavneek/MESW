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
WORKING_DIR = "./graphrag_cache"
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
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
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
    """Enhanced query interface for the chunk-based dictionary"""

    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.vocabulary = {}
        self.chunk_data = {}
        self.chunk_results = []
        self.processing_stats = {}
        self.graph_rag = None
        self.load_all_data()

    def load_all_data(self):
        """Load all saved data from the comprehensive build"""
        print(f"📚 Loading dictionary data from {self.working_dir}...")

        # Load definitions (main vocabulary) - using entity_definitions.json
        definitions_path = os.path.join(self.working_dir, "entity_definitions.json")
        if os.path.exists(definitions_path):
            with open(definitions_path, "r", encoding="utf-8") as f:
                definitions = json.load(f)
            print(f"✅ Loaded {len(definitions)} definitions")
        else:
            definitions = {}
            print("⚠️ No definitions file found")

        # Load terms with contexts - using entity_contexts.json
        contexts_path = os.path.join(self.working_dir, "entity_contexts.json")
        if os.path.exists(contexts_path):
            with open(contexts_path, "r", encoding="utf-8") as f:
                contexts = json.load(f)
            print(f"✅ Loaded contexts for {len(contexts)} terms")
        else:
            contexts = {}
            print("⚠️ No contexts file found")

        # Load processing stats
        stats_path = os.path.join(self.working_dir, "processing_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                self.processing_stats = json.load(f)
            print(f"✅ Loaded processing statistics")

        # Create vocabulary structure from loaded data
        self.vocabulary = {}
        for term, definition in definitions.items():
            # Get contexts for this term
            term_contexts = contexts.get(term, []) if isinstance(contexts, dict) else []

            # Find which chunks contain this term (if we have chunk results)
            found_in_chunks = []
            if hasattr(self, "chunk_results") and self.chunk_results:
                for result in self.chunk_results:
                    if term in result.get("terms", []):
                        found_in_chunks.append(result["chunk_id"])

            self.vocabulary[term] = {
                "definition": definition,
                "contexts": term_contexts,
                "found_in_chunks": found_in_chunks,
                "has_definition": True,
            }

        print(f"📊 Dictionary ready: {len(self.vocabulary)} terms")

    def _find_term_chunks(self, term: str) -> List[int]:
        """Find which chunks contain this term"""
        chunks_with_term = []
        for result in self.chunk_results:
            if term in result.get("terms", []):
                chunks_with_term.append(result["chunk_id"])
        return chunks_with_term

    def load_graphrag(self) -> bool:
        """Try to load GraphRAG if available"""
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

    def suggest_words(self, query: str, n: int = 5) -> List[str]:
        """Suggest similar words based on query"""
        query = query.lower().strip()
        vocab_words = list(self.vocabulary.keys())

        if query in vocab_words:
            return [query]

        from difflib import get_close_matches

        matches = get_close_matches(query, vocab_words, n=n, cutoff=0.6)

        # Prefix matching
        prefix_matches = [w for w in vocab_words if w.startswith(query)][:n]

        # Substring matching
        substring_matches = [w for w in vocab_words if query in w.lower()][:n]

        # Combine and deduplicate
        combined = matches + [w for w in prefix_matches if w not in matches]
        combined = combined + [w for w in substring_matches if w not in combined]

        return combined[:n]

    def define_word(self, word: str) -> Dict[str, Any]:
        """Get comprehensive definition for a word"""
        word_lower = word.lower().strip()
        start_time = time.time()

        if word_lower in self.vocabulary:
            vocab_entry = self.vocabulary[word_lower]

            result = {
                "word": word,
                "definition": vocab_entry["definition"],
                "found": True,
                "contexts": vocab_entry["contexts"],
                "found_in_chunks": vocab_entry["found_in_chunks"],
                "chunk_count": len(vocab_entry["found_in_chunks"]),
                "context_count": len(vocab_entry["contexts"]),
                "has_definition": vocab_entry["has_definition"],
                "query_time": time.time() - start_time,
                "suggestions": [],
            }
        else:
            suggestions = self.suggest_words(word)
            result = {
                "word": word,
                "definition": None,
                "found": False,
                "contexts": [],
                "found_in_chunks": [],
                "chunk_count": 0,
                "context_count": 0,
                "has_definition": False,
                "query_time": time.time() - start_time,
                "suggestions": suggestions,
            }

        return result

    async def query_graphrag(self, query: str, mode: str = "local") -> str:
        """Query using GraphRAG if available"""
        if not self.graph_rag:
            if not self.load_graphrag():
                return "GraphRAG not available"

        try:
            result = await self.graph_rag.aquery(
                query, param=QueryParam(mode=mode, top_k=5)
            )
            return result
        except Exception as e:
            return f"GraphRAG query failed: {e}"

    def search_in_chunks(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for terms or phrases across chunks"""
        query_lower = query.lower().strip()
        results = []

        for result in self.chunk_results:
            chunk_id = result["chunk_id"]

            # Check if query matches any terms in this chunk
            matching_terms = []
            for term in result.get("terms", []):
                if query_lower in term.lower() or term.lower() in query_lower:
                    matching_terms.append(term)

            # Check definitions for matches
            matching_definitions = []
            for term, definition in result.get("definitions", {}).items():
                if query_lower in definition.lower():
                    matching_definitions.append(
                        {"term": term, "definition": definition}
                    )

            if matching_terms or matching_definitions:
                chunk_info = None
                if self.chunk_data and "chunks" in self.chunk_data:
                    chunk_info = next(
                        (c for c in self.chunk_data["chunks"] if c["id"] == chunk_id),
                        None,
                    )

                results.append(
                    {
                        "chunk_id": chunk_id,
                        "matching_terms": matching_terms,
                        "matching_definitions": matching_definitions,
                        "chunk_info": chunk_info,
                        "relevance_score": len(matching_terms)
                        + len(matching_definitions),
                    }
                )

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]

    def get_term_distribution(self, term: str) -> Dict[str, Any]:
        """Get distribution of term across chunks"""
        word_lower = term.lower().strip()

        if word_lower not in self.vocabulary:
            return {"error": "Term not found"}

        vocab_entry = self.vocabulary[word_lower]
        chunk_distribution = []

        for chunk_id in vocab_entry["found_in_chunks"]:
            chunk_result = next(
                (r for r in self.chunk_results if r["chunk_id"] == chunk_id), None
            )
            if chunk_result:
                chunk_info = None
                if self.chunk_data and "chunks" in self.chunk_data:
                    chunk_info = next(
                        (c for c in self.chunk_data["chunks"] if c["id"] == chunk_id),
                        None,
                    )

                chunk_distribution.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_info": chunk_info,
                        "has_definition": term in chunk_result.get("definitions", {}),
                        "definition": chunk_result.get("definitions", {}).get(term, ""),
                    }
                )

        return {
            "term": term,
            "total_chunks": len(vocab_entry["found_in_chunks"]),
            "distribution": chunk_distribution,
            "contexts": vocab_entry["contexts"],
        }

    def format_definition(self, result: Dict[str, Any]) -> str:
        """Format definition result for display"""
        word = result["word"]
        definition = result["definition"]

        output = f"📖 **{word.upper()}**"
        if result["found"]:
            output += f" (found in {result['chunk_count']} chunks, {result['context_count']} contexts)"

        output += "\n" + "=" * (len(word) + 20) + "\n"

        if definition:
            output += f"💡 **Definition:** {definition}\n"

            if result["contexts"]:
                output += f"\n📚 **Contexts:**\n"
                for i, context in enumerate(result["contexts"][:3], 1):
                    output += f"  {i}. {context[:200]}...\n"

            if result["found_in_chunks"]:
                chunks_str = ", ".join(map(str, result["found_in_chunks"][:10]))
                if len(result["found_in_chunks"]) > 10:
                    chunks_str += f" and {len(result['found_in_chunks']) - 10} more"
                output += f"\n🧩 **Found in chunks:** {chunks_str}\n"

        else:
            output += f"❌ No definition found for '{word}'\n"
            if result["suggestions"]:
                output += (
                    f"\n💡 **Similar words:** {', '.join(result['suggestions'][:5])}\n"
                )

        output += f"\n⏱️ Query time: {result['query_time']:.3f}s"
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "total_terms": len(self.vocabulary),
            "total_chunks": self.chunk_data.get("total_chunks", 0),
            "chunks_with_terms": len(
                [r for r in self.chunk_results if r.get("terms_found", 0) > 0]
            ),
            "chunks_with_definitions": len(
                [r for r in self.chunk_results if r.get("definitions_generated", 0) > 0]
            ),
            "average_terms_per_chunk": 0,
            "terms_with_multiple_chunks": 0,
            "processing_strategy": "unknown",
        }

        if self.chunk_results:
            total_terms = sum(r.get("terms_found", 0) for r in self.chunk_results)
            stats["average_terms_per_chunk"] = total_terms / len(self.chunk_results)

        # Count terms that appear in multiple chunks
        for term_info in self.vocabulary.values():
            if len(term_info.get("found_in_chunks", [])) > 1:
                stats["terms_with_multiple_chunks"] += 1

        if self.processing_stats:
            stats["processing_strategy"] = self.processing_stats.get(
                "processing_stats", {}
            ).get("strategy", "unknown")

        return stats

    async def interactive_query(self):
        """Enhanced interactive query interface with the new search capabilities"""
        print(
            f"""
    📚 Enhanced Book Dictionary Interface
    {"="*60}
    📊 Dictionary: {len(self.vocabulary)} terms from {self.chunk_data.get('total_chunks', 0)} chunks
    📈 Coverage: {len([r for r in self.chunk_results if r.get('terms_found', 0) > 0])}/{self.chunk_data.get('total_chunks', 0)} chunks have terms
    🔗 Multi-chunk terms: {len([t for t in self.vocabulary.values() if len(t.get('found_in_chunks', [])) > 1])}

    Commands:
    <word>           - Look up word definition
    search <query>   - Search across chunks
    sim <term> [hops] - Find similar terms (default hops=1)
    dist <word>      - Show word distribution across chunks
    graphrag <query> - Query using GraphRAG (if available)
    vocab            - Show vocabulary sample
    stats            - Show detailed statistics
    chunks           - Show chunk information
    quit/exit        - Exit
    {"="*60}"""
        )

        has_graphrag = self.load_graphrag()

        while True:
            try:
                user_input = input("\n🔍 Enter command: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    continue

                parts = user_input.split(" ", 2)
                command = parts[0].lower()
                query = parts[1] if len(parts) > 1 else ""
                params = parts[2] if len(parts) > 2 else ""

                if command == "sim":
                    hops = 1
                    if params:
                        try:
                            hops = int(params)
                        except:
                            pass

                    if not query:
                        print("❌ Please provide a term")
                        continue

                    result = self.enhanced_query(query, hops=hops)
                    print(self.format_enhanced_query_result(result))

                elif command == "graphrag":
                    if not has_graphrag:
                        print("❌ GraphRAG not available")
                        continue
                    if not query:
                        print("❌ Please provide a GraphRAG query")
                        continue
                    print(f"🕸️ Querying GraphRAG: '{query}'...")
                    result = await self.query_graphrag(query)
                    print(f"\n💡 **GraphRAG Response:**\n{result}")

                elif command == "search":
                    if not query:
                        print("❌ Please provide a search query")
                        continue
                    search_results = self.search_in_chunks(query)
                    print(
                        f"\n🔍 **SEARCH RESULTS for '{query}'** ({len(search_results)} matches)"
                    )
                    for result in search_results:
                        chunk_id = result["chunk_id"]
                        print(
                            f"\n📄 **Chunk {chunk_id}** (relevance: {result['relevance_score']})"
                        )
                        if result["matching_terms"]:
                            print(f"  🏷️ Terms: {', '.join(result['matching_terms'])}")
                        if result["matching_definitions"]:
                            for def_match in result["matching_definitions"][:2]:
                                print(
                                    f"  💡 {def_match['term']}: {def_match['definition'][:100]}..."
                                )

                elif command == "dist":
                    if not query:
                        print("❌ Please provide a term")
                        continue
                    distribution = self.get_term_distribution(query)
                    if "error" in distribution:
                        print(f"❌ {distribution['error']}")
                    else:
                        print(f"\n📊 **DISTRIBUTION of '{distribution['term']}'**")
                        print(f"Found in {distribution['total_chunks']} chunks:")
                        for chunk_dist in distribution["distribution"][:10]:
                            chunk_id = chunk_dist["chunk_id"]
                            has_def = "✓" if chunk_dist["has_definition"] else "○"
                            print(f"  Chunk {chunk_id}: {has_def}")

                elif command == "vocab":
                    vocab_sample = sorted(list(self.vocabulary.keys()))[:20]
                    print(
                        f"\n📝 **VOCABULARY SAMPLE** (first 20 of {len(self.vocabulary)}):"
                    )
                    for i, word in enumerate(vocab_sample, 1):
                        info = self.vocabulary[word]
                        chunks = len(info["found_in_chunks"])
                        contexts = len(info["contexts"])
                        print(
                            f"  {i:2d}. {word} (chunks: {chunks}, contexts: {contexts})"
                        )

                elif command == "stats":
                    detailed_stats = self.get_stats()
                    print(f"\n📊 **DETAILED STATISTICS**")
                    for key, value in detailed_stats.items():
                        formatted_key = key.replace("_", " ").title()
                        if isinstance(value, float):
                            print(f"  {formatted_key}: {value:.2f}")
                        else:
                            print(f"  {formatted_key}: {value}")

                elif command == "chunks":
                    if self.chunk_data and "chunks" in self.chunk_data:
                        print(
                            f"\n🧩 **CHUNK INFORMATION** ({self.chunk_data['total_chunks']} total)"
                        )
                        for chunk in self.chunk_data["chunks"][:10]:
                            chunk_result = next(
                                (
                                    r
                                    for r in self.chunk_results
                                    if r["chunk_id"] == chunk["id"]
                                ),
                                {},
                            )
                            terms_count = chunk_result.get("terms_found", 0)
                            defs_count = chunk_result.get("definitions_generated", 0)
                            print(
                                f"  Chunk {chunk['id']}: {terms_count} terms, {defs_count} definitions"
                            )
                            print(f"    Preview: {chunk.get('preview', 'No preview')}")
                        if len(self.chunk_data["chunks"]) > 10:
                            print(
                                f"  ... and {len(self.chunk_data['chunks']) - 10} more chunks"
                            )
                    else:
                        print("❌ No chunk information available")

                else:
                    # Default: word lookup
                    word_to_lookup = user_input
                    result = self.enhanced_query(word_to_lookup)
                    print(self.format_enhanced_query_result(result))

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def find_similar_terms(
        self, term: str, hops: int = 1, max_terms: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar terms in the graph up to the specified number of hops.

        Args:
            term: The term to search from
            hops: Number of hops to traverse in the graph
            max_terms: Maximum number of terms to return

        Returns:
            List of similar terms with their definitions and hop distance
        """
        if not self.vocabulary:
            return []

        term_lower = term.lower().strip()

        # If the term exists, start from it
        if term_lower in self.vocabulary:
            seed_terms = [term_lower]
        else:
            # If not found, use suggestions as seed
            seed_terms = self.suggest_words(term, n=1)
            if not seed_terms:
                return []

        # Initialize the search
        visited = set()
        results = []
        current_terms = seed_terms
        current_hop = 0

        while current_terms and current_hop <= hops:
            next_terms = []

            for t in current_terms:
                if t in visited:
                    continue

                visited.add(t)

                # Add to results with hop information
                if t in self.vocabulary:
                    results.append(
                        {
                            "term": t,
                            "definition": self.vocabulary[t]["definition"],
                            "hop": current_hop,
                            "is_original": t == term_lower,
                            "context_count": len(self.vocabulary[t]["contexts"]),
                        }
                    )

                # Find neighbors at this level if not at max hops
                if current_hop < hops:
                    neighbors = self._find_similar_terms(t, threshold=0.5)
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor not in next_terms:
                            next_terms.append(neighbor)

            current_terms = next_terms
            current_hop += 1

        # Sort by hop distance and context count
        results.sort(key=lambda x: (x["hop"], -x["context_count"]))

        # Return only up to max_terms
        return results[:max_terms]

    def _find_similar_terms(self, term: str, threshold: float = 0.6) -> List[str]:
        """Find similar terms in the vocabulary"""
        similar_terms = []
        for t in self.vocabulary.keys():
            if t == term:
                continue
            similarity = SequenceMatcher(None, term.lower(), t.lower()).ratio()
            if similarity >= threshold:
                similar_terms.append(t)

        # Sort by similarity
        similar_terms.sort(
            key=lambda x: SequenceMatcher(None, term.lower(), x.lower()).ratio(),
            reverse=True,
        )
        return similar_terms[:5]  # Return top 5 similar terms

    def enhanced_query(self, term: str, hops: int = 1) -> Dict[str, Any]:
        """
        Enhanced query that handles both exact matches and similar terms.

        Args:
            term: The term to search for
            hops: Number of hops for similar term search

        Returns:
            Dictionary with query results
        """
        term_lower = term.lower().strip()

        # Check if term exists exactly
        if term_lower in self.vocabulary:
            definition = self.vocabulary[term_lower]["definition"]
            contexts = self.vocabulary[term_lower]["contexts"][:3]
            return {
                "found": True,
                "term": term_lower,
                "definition": definition,
                "contexts": contexts,
                "similar_terms": self.find_similar_terms(term, hops=hops, max_terms=3),
                "hops": hops,
            }

        # Term not found, suggest similar terms
        similar_terms = self.find_similar_terms(term, hops=hops, max_terms=3)

        return {
            "found": False,
            "term": term_lower,
            "message": f"Term '{term}' not found in the dictionary. Here are similar terms:",
            "similar_terms": similar_terms,
            "hops": hops,
        }

    def format_enhanced_query_result(self, result: Dict[str, Any]) -> str:
        """Format enhanced query results for display"""
        output = ""

        if result["found"]:
            output += f"📖 **{result['term'].upper()}**\n"
            output += "=" * (len(result["term"]) + 20) + "\n"
            output += f"💡 **Definition:** {result['definition']}\n"

            if result["contexts"]:
                output += "\n📚 **Contexts:**\n"
                for i, context in enumerate(result["contexts"], 1):
                    output += f"  {i}. {context[:200]}...\n"

            output += f"\n🔗 Found in {len(self.vocabulary[result['term']]['found_in_chunks'])} chunks"
        else:
            output += f"❌ **{result['term'].upper()}**\n"
            output += "=" * (len(result["term"]) + 20) + "\n"
            output += f"{result['message']}\n"

        if result["similar_terms"]:
            output += f"\n🔍 **SIMILAR TERMS (up to {result['hops']} hop{'s' if result['hops'] > 1 else ''}):**\n"
            for i, term_info in enumerate(result["similar_terms"], 1):
                hop_label = (
                    "Original"
                    if term_info["is_original"]
                    else f"Hop {term_info['hop']}"
                )
                output += f"\n{i}. **{term_info['term'].upper()}** [{hop_label}]"
                output += f" (contexts: {term_info['context_count']})"
                if "definition" in term_info:
                    output += f"\n   💡 {term_info['definition'][:150]}..."

        return output


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py build [book_file] - Build GraphRAG from book")
        print("  python script.py query [term] - Query term from built GraphRAG")
        print("  python script.py interactive - Interactive query mode")
        print("  python script.py search [query] - Search across chunks")
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
        # Query mode - look up specific term
        if len(sys.argv) < 3:
            print("❌ Please specify a term to query")
            return

        term = " ".join(sys.argv[2:])
        print(f"🔍 Querying term: '{term}'")

        # Initialize query system
        query_system = BookDictionaryQuery()

        # Check if we have data to query
        if not query_system.vocabulary:
            print(
                "❌ No vocabulary loaded. Please build the graph first with: python script.py build [book_file]"
            )
            return

        # Get definition for the term
        result = query_system.define_word(term)

        # Format and display result
        print(query_system.format_definition(result))

        # If term not found, show similar terms
        if not result["found"]:
            print(f"\n💡 Similar terms you might be looking for:")
            suggestions = query_system.suggest_words(term, n=5)
            for i, suggestion in enumerate(suggestions, 1):
                if suggestion != term:
                    print(f"  {i}. {suggestion}")

    elif command == "interactive":
        # Interactive query mode
        print("📚 Starting interactive query interface...")
        query_system = BookDictionaryQuery()

        # Check if we have data to query
        if not query_system.vocabulary:
            print(
                "❌ No vocabulary loaded. Please build the graph first with: python script.py build [book_file]"
            )
            return

        # Start interactive mode
        await query_system.interactive_query()

    elif command == "search":
        # Search mode - search across chunks
        if len(sys.argv) < 3:
            print("❌ Please specify a search query")
            return

        query = " ".join(sys.argv[2:])
        print(f"🔍 Searching for: '{query}'")

        # Initialize query system
        query_system = BookDictionaryQuery()

        # Check if we have data to query
        if not query_system.vocabulary:
            print(
                "❌ No vocabulary loaded. Please build the graph first with: python script.py build [book_file]"
            )
            return

        # Search in chunks
        results = query_system.search_in_chunks(query)

        # Display results
        print(f"\n🔍 **SEARCH RESULTS for '{query}'** ({len(results)} matches)")
        for result in results:
            chunk_id = result["chunk_id"]
            print(f"\n📄 **Chunk {chunk_id}** (relevance: {result['relevance_score']})")
            if result["matching_terms"]:
                print(f"  🏷️ Terms: {', '.join(result['matching_terms'])}")
            if result["matching_definitions"]:
                for def_match in result["matching_definitions"][:2]:
                    print(
                        f"  💡 {def_match['term']}: {def_match['definition'][:100]}..."
                    )

    else:
        print(f"❌ Unknown command: {command}")
        print("Usage:")
        print("  python script.py build [book_file] - Build GraphRAG from book")
        print("  python script.py query [term] - Query term from built GraphRAG")
        print("  python script.py interactive - Interactive query mode")
        print("  python script.py search [query] - Search across chunks")


if __name__ == "__main__":
    asyncio.run(main())
