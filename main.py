#!/usr/bin/env python3
"""
Adaptive Technical Dictionary Builder with GraphRAG
For Japanese Power System Technical Terms
"""

import asyncio
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from openai import AsyncOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================
TOKENIZER_NAME = "sudachi"  # Options: "sudachi", "mecab"
TERM_FOLDER = "./term_folder"
WORKING_DIR = "./graph_rag_cache"
MAX_CONCURRENT_DEFINITIONS = 15
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKEN = 2000
SEMANTIC_SIMILARITY_THRESHOLD = 0.75
CHUNK_SIZE = 1000  # tokens/words
CHUNK_OVERLAP = 200

# ============================================================================
# EMBEDDING MODEL
# ============================================================================
try:
    EMBED_MODEL = SentenceTransformer("models/embeddings", local_files_only=True)
    print("✓ Local embedding model loaded")
except:
    try:
        EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("✓ Fallback embedding model loaded")
    except Exception as e:
        print(f"✗ Failed to load embedding model: {e}")
        sys.exit(1)

# vLLM client
vllm_client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_HOST + "/v1")


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class FilteredTerms(BaseModel):
    """Pass 1: LLM filters which terms are actually technical"""

    terms: List[str]


class TermDefinition(BaseModel):
    """Pass 2: LLM generates definition for a term"""

    term: str
    definition: str


# ============================================================================
# TOKENIZERS
# ============================================================================
class BaseTokenizer(ABC):
    """Base tokenizer interface - returns only surface and pos"""

    @abstractmethod
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """
        Returns list of tokens with:
        - surface: original text
        - pos: normalized POS tag (名詞, 動詞, etc.)
        """
        pass


class SudachiTokenizer(BaseTokenizer):
    """SudachiPy tokenizer wrapper"""

    def __init__(self):
        try:
            from sudachipy import dictionary
            from sudachipy import tokenizer as sudachi_tokenizer

            self.tokenizer_obj = dictionary.Dictionary(dict="full").create()
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.C
            print("✓ SudachiPy tokenizer initialized")
        except ImportError:
            raise ImportError("SudachiPy not available. Install: pip install sudachipy")
        except TypeError:
            self.tokenizer_obj = dictionary.Dictionary(dict_type="full").create()
            self.mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """Tokenize text and return surface + pos only"""
        tokens = self.tokenizer_obj.tokenize(text, self.mode)
        results = []

        for token in tokens:
            surface = token.surface()
            pos = token.part_of_speech()
            main_pos = pos[0] if pos else "UNKNOWN"

            results.append({"surface": surface, "pos": main_pos})

        return results


class MeCabTokenizer(BaseTokenizer):
    """MeCab tokenizer wrapper"""

    def __init__(self):
        try:
            import MeCab

            self.mecab = MeCab.Tagger("")
            print("✓ MeCab tokenizer initialized")
        except ImportError:
            raise ImportError("MeCab not available. Install: pip install mecab-python3")

    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """Tokenize text and return surface + pos only"""
        result = self.mecab.parse(text)
        lines = result.strip().split("\n")[:-1]  # Remove EOS
        results = []

        for line in lines:
            if "\t" not in line:
                continue

            surface, features = line.split("\t", 1)
            feature_parts = features.split(",")
            pos = feature_parts[0] if feature_parts else "UNKNOWN"

            results.append({"surface": surface, "pos": pos})

        return results


def get_tokenizer(tokenizer_name: str) -> BaseTokenizer:
    """Factory function to get tokenizer by name"""
    if tokenizer_name.lower() == "sudachi":
        return SudachiTokenizer()
    elif tokenizer_name.lower() == "mecab":
        return MeCabTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


# ============================================================================
# ANALYTICS LOADER
# ============================================================================
class AnalyticsLoader:
    """Load pre-filtered technical terms from analytics JSON"""

    def __init__(self, term_folder: str, tokenizer_name: str):
        self.term_folder = Path(term_folder)
        self.tokenizer_name = tokenizer_name
        self.terms: Set[str] = set()

    def load_terms(self) -> Set[str]:
        """Load terms from analytics_<tokenizer>.json"""
        analytics_file = self.term_folder / f"analytics_{self.tokenizer_name}.json"

        if not analytics_file.exists():
            raise FileNotFoundError(
                f"Analytics file not found: {analytics_file}\n"
                f"Expected path: {analytics_file.absolute()}"
            )

        print(f"Loading terms from: {analytics_file}")

        with open(analytics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract only the "word" field from "unknown_words" list
        unknown_words = data.get("unknown_words", [])
        self.terms = {item["word"] for item in unknown_words if "word" in item}

        print(f"✓ Loaded {len(self.terms)} technical terms")
        return self.terms


# ============================================================================
# BOOK CHUNKER
# ============================================================================
class BookChunker:
    """Split book into chunks using selected tokenizer"""

    def __init__(self, tokenizer: BaseTokenizer, chunk_size: int, overlap: int):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks based on tokens"""
        print(f"Creating chunks from book ({len(text)} characters)...")

        # Tokenize entire text
        all_tokens = self.tokenizer.tokenize(text)
        chunks = []

        for i in range(0, len(all_tokens), self.chunk_size - self.overlap):
            chunk_tokens = all_tokens[i : i + self.chunk_size]

            # Reconstruct text from tokens
            chunk_text = "".join([t["surface"] for t in chunk_tokens])

            chunks.append(
                {
                    "id": len(chunks),
                    "text": chunk_text,
                    "tokens": chunk_tokens,
                    "start_token": i,
                    "end_token": i + len(chunk_tokens),
                }
            )

        print(f"✓ Created {len(chunks)} chunks")
        return chunks


# ============================================================================
# TERM MATCHER
# ============================================================================
class TermMatcher:
    """Match tokens against pre-filtered technical terms"""

    def __init__(self, technical_terms: Set[str]):
        self.technical_terms = technical_terms

    def match_terms(
        self, tokens: List[Dict[str, str]]
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Match tokens against technical terms.

        Returns:
        - full_matches: List of exact matches
        - partial_matches: Dict[partial_term -> List[compounds it's part of]]
        """
        full_matches = []
        partial_matches = defaultdict(list)

        for token in tokens:
            surface = token["surface"]
            pos = token["pos"]

            # FULL MATCH: exact term found
            if surface in self.technical_terms:
                full_matches.append(surface)

            # PARTIAL MATCH: token is substring of a compound term (only nouns)
            elif "名詞" in pos:
                for term in self.technical_terms:
                    if surface in term and surface != term:
                        partial_matches[surface].append(term)

        # Deduplicate
        full_matches = list(set(full_matches))

        # Convert defaultdict to regular dict
        partial_matches = {k: list(set(v)) for k, v in partial_matches.items()}

        return full_matches, partial_matches


# ============================================================================
# LLM FUNCTIONS
# ============================================================================
async def vllm_complete(prompt: str, system_prompt: str = None) -> str:
    """General vLLM completion"""
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Respond only with the requested information in the exact format specified."

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
            extra_body={"repetition_penalty": 1.1, "top_p": 0.9},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"✗ vLLM request failed: {e}")
        return ""


async def filter_technical_terms(
    chunk_text: str, candidate_terms: List[str]
) -> List[str]:
    """Pass 1: LLM filters which terms are actually technical in this context"""
    prompt = f"""Given this text chunk, identify which of the following candidate terms are actually technical terms in this context.

Text chunk:
{chunk_text[:1000]}...

Candidate terms:
{', '.join(candidate_terms)}

Return ONLY a JSON object with a "terms" field containing a list of technical terms.
Example: {{"terms": ["term1", "term2"]}}

Do not include explanations or any other text."""

    system_prompt = "You are an expert at identifying technical terms in Japanese power system documents. Return only valid JSON."

    response = await vllm_complete(prompt, system_prompt)

    try:
        # Parse JSON response
        result = json.loads(response)
        validated = FilteredTerms(**result)
        return validated.terms
    except Exception as e:
        print(f"  ✗ Failed to parse LLM response: {e}")
        print(f"  Response was: {response[:200]}")
        return []


async def generate_definition(term: str, context: str) -> Optional[TermDefinition]:
    """Pass 2: LLM generates definition for a term"""
    prompt = f"""Based on the following context, provide a clear, concise definition for this technical term.

Term: {term}

Context:
{context[:1500]}

Return ONLY a JSON object with "term" and "definition" fields.
Example: {{"term": "{term}", "definition": "A technical device that..."}}

Keep the definition to 1-2 sentences. Do not include explanations or any other text."""

    system_prompt = "You are an expert at defining technical terms in Japanese power systems. Return only valid JSON with term and definition."

    response = await vllm_complete(prompt, system_prompt)

    try:
        result = json.loads(response)
        validated = TermDefinition(**result)
        return validated
    except Exception as e:
        print(f"  ✗ Failed to parse definition for '{term}': {e}")
        return None


# ============================================================================
# CHUNK PROCESSOR
# ============================================================================
class ChunkProcessor:
    """Two-pass chunk processor"""

    def __init__(self, matcher: TermMatcher):
        self.matcher = matcher

    async def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chunk in two passes:
        1. Match terms and filter with LLM
        2. Generate definitions for filtered terms
        """
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
        tokens = chunk["tokens"]

        print(f"\n{'='*60}")
        print(f"Processing Chunk {chunk_id}")
        print(f"{'='*60}")

        # PASS 1: Match and filter
        full_matches, partial_matches = self.matcher.match_terms(tokens)

        all_candidates = full_matches + list(partial_matches.keys())
        print(f"  Found {len(full_matches)} full matches")
        print(f"  Found {len(partial_matches)} partial matches")

        if not all_candidates:
            print("  ✓ No terms to process in this chunk")
            return {
                "chunk_id": chunk_id,
                "terms": [],
                "definitions": {},
                "partial_info": {},
            }

        # Filter with LLM
        print(f"  → Filtering {len(all_candidates)} candidates with LLM...")
        filtered_terms = await filter_technical_terms(chunk_text, all_candidates)
        print(f"  ✓ LLM confirmed {len(filtered_terms)} technical terms")

        # PASS 2: Generate definitions (concurrent)
        print(
            f"  → Generating definitions (max {MAX_CONCURRENT_DEFINITIONS} concurrent)..."
        )
        definitions = await self._generate_definitions_concurrent(
            filtered_terms, chunk_text
        )
        print(f"  ✓ Generated {len(definitions)} definitions")

        # Prepare partial match info
        partial_info = {
            term: partial_matches[term]
            for term in filtered_terms
            if term in partial_matches
        }

        return {
            "chunk_id": chunk_id,
            "terms": filtered_terms,
            "definitions": definitions,
            "partial_info": partial_info,
        }

    async def _generate_definitions_concurrent(
        self, terms: List[str], context: str
    ) -> Dict[str, str]:
        """Generate definitions concurrently with semaphore"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEFINITIONS)

        async def generate_with_semaphore(term: str) -> Tuple[str, Optional[str]]:
            async with semaphore:
                result = await generate_definition(term, context)
                if result:
                    return result.term, result.definition
                return term, None

        tasks = [generate_with_semaphore(term) for term in terms]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        definitions = {term: defn for term, defn in results if defn is not None}
        return definitions


# ============================================================================
# DICTIONARY MANAGER
# ============================================================================
class DictionaryManager:
    """Simple dictionary for fast exact/fuzzy lookup"""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.dictionary: Dict[str, str] = {}
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def add_terms(self, terms_with_definitions: Dict[str, str]) -> None:
        """Add or update terms in dictionary"""
        for term, definition in terms_with_definitions.items():
            self.dictionary[term] = definition

    def lookup_exact(self, term: str) -> Optional[str]:
        """Exact lookup"""
        return self.dictionary.get(term.lower())

    def lookup_fuzzy(self, term: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Fuzzy lookup using character similarity"""
        from difflib import SequenceMatcher

        term_normalized = term.lower()
        similarities = []

        for dict_term, definition in self.dictionary.items():
            similarity = SequenceMatcher(None, term_normalized, dict_term).ratio()
            similarities.append(
                {"term": dict_term, "definition": definition, "similarity": similarity}
            )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    def save(self) -> None:
        """Save dictionary to disk"""
        dict_path = self.working_dir / "dictionary.json"
        with open(dict_path, "w", encoding="utf-8") as f:
            json.dump(self.dictionary, f, indent=2, ensure_ascii=False)
        print(f"✓ Dictionary saved: {len(self.dictionary)} terms")

    def load(self) -> bool:
        """Load dictionary from disk"""
        dict_path = self.working_dir / "dictionary.json"
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
            print(f"✓ Loaded {len(self.dictionary)} terms from dictionary")
            return True
        return False


# ============================================================================
# GRAPH MANAGER WITH RELATIONSHIPS
# ============================================================================
class GraphManager:
    """Manages GraphRAG with explicit relationships"""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.nodes: Dict[str, Dict[str, str]] = {}  # term -> {definition, ...}
        self.edges: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # term -> [{target, type, weight}]
        self.embeddings: Dict[str, np.ndarray] = {}
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def add_terms(self, terms_with_definitions: Dict[str, str]) -> None:
        """Add terms as graph nodes"""
        for term, definition in terms_with_definitions.items():
            self.nodes[term] = {"definition": definition}

    def add_compound_relationships(self, partial_info: Dict[str, List[str]]) -> None:
        """Add COMPOUND_OF edges"""
        for partial_term, compound_terms in partial_info.items():
            for compound in compound_terms:
                if compound in self.nodes:
                    # compound ←COMPOUND_OF― partial
                    self.edges[compound].append(
                        {"target": partial_term, "type": "COMPOUND_OF", "weight": 1.0}
                    )

    def compute_embeddings(self) -> None:
        """Compute embeddings for all definitions"""
        print("Computing embeddings for graph nodes...")
        terms = list(self.nodes.keys())
        definitions = [self.nodes[t]["definition"] for t in terms]

        embeddings = EMBED_MODEL.encode(definitions, normalize_embeddings=True)

        for term, embedding in zip(terms, embeddings):
            self.embeddings[term] = embedding

        print(f"✓ Computed embeddings for {len(self.embeddings)} terms")

    def build_relationships(self) -> None:
        """Build all relationship types"""
        print("Building graph relationships...")

        # 1. SIMILAR_MEANING (semantic similarity)
        self._build_similar_meaning_edges()

        # 2. PARENT_TERM (term appears in definition)
        self._build_parent_term_edges()

        # 3. CATEGORY (taxonomy patterns)
        self._build_category_edges()

        print(f"✓ Built relationships for {len(self.nodes)} nodes")

    def _build_similar_meaning_edges(self) -> None:
        """Add SIMILAR_MEANING edges based on embedding similarity"""
        terms = list(self.embeddings.keys())

        for i, term1 in enumerate(terms):
            emb1 = self.embeddings[term1]

            for term2 in terms[i + 1 :]:
                emb2 = self.embeddings[term2]
                similarity = float(np.dot(emb1, emb2))

                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    self.edges[term1].append(
                        {
                            "target": term2,
                            "type": "SIMILAR_MEANING",
                            "weight": similarity,
                        }
                    )
                    self.edges[term2].append(
                        {
                            "target": term1,
                            "type": "SIMILAR_MEANING",
                            "weight": similarity,
                        }
                    )

    def _build_parent_term_edges(self) -> None:
        """Add PARENT_TERM edges when term appears in another's definition"""
        for term, node_data in self.nodes.items():
            definition = node_data["definition"].lower()

            for other_term in self.nodes.keys():
                if other_term != term and other_term in definition:
                    # term uses other_term in its definition
                    self.edges[term].append(
                        {"target": other_term, "type": "PARENT_TERM", "weight": 1.0}
                    )

    def _build_category_edges(self) -> None:
        """Add CATEGORY edges by detecting taxonomy patterns"""
        category_patterns = [
            r"(.+?)の一種",  # "X is a type of Y"
            r"(.+?)に属する",  # "belongs to X"
            r"(.+?)系",  # "X-system"
        ]

        for term, node_data in self.nodes.items():
            definition = node_data["definition"]

            for pattern in category_patterns:
                matches = re.findall(pattern, definition)
                for match in matches:
                    category = match.strip()
                    if category in self.nodes:
                        self.edges[term].append(
                            {"target": category, "type": "CATEGORY", "weight": 1.0}
                        )

    async def build_graphrag(self) -> bool:
        """Build GraphRAG knowledge graph"""
        print("Building GraphRAG knowledge graph...")

        # Create chunks: term + definition
        chunks = [f"{term}: {data['definition']}" for term, data in self.nodes.items()]

        try:
            rag = GraphRAG(
                working_dir=str(self.working_dir),
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
                best_model_max_token_size=MAX_TOKEN,
                cheap_model_max_token_size=MAX_TOKEN,
                embedding_func_max_async=3,
                embedding_batch_num=4,
            )
            await rag.ainsert(chunks)
            print("✓ GraphRAG built successfully")
            return True
        except Exception as e:
            print(f"✗ GraphRAG build failed: {e}")
            return False

    def query_graph(self, term: str, max_depth: int = 2) -> Dict[str, Any]:
        """Query graph for related terms (BFS traversal)"""
        if term not in self.nodes:
            return {"found": False, "term": term}

        visited = set()
        queue = [(term, 0)]  # (term, depth)
        related_nodes = {}
        related_edges = []

        while queue:
            current_term, depth = queue.pop(0)

            if current_term in visited or depth > max_depth:
                continue

            visited.add(current_term)
            related_nodes[current_term] = self.nodes[current_term]

            # Get edges from this node
            for edge in self.edges.get(current_term, []):
                target = edge["target"]
                related_edges.append(
                    {
                        "source": current_term,
                        "target": target,
                        "type": edge["type"],
                        "weight": edge["weight"],
                    }
                )

                if target not in visited and depth + 1 <= max_depth:
                    queue.append((target, depth + 1))

        return {
            "found": True,
            "query_term": term,
            "nodes": related_nodes,
            "edges": related_edges,
            "total_nodes": len(related_nodes),
            "total_edges": len(related_edges),
        }

    def save(self) -> None:
        """Save graph data"""
        # Save nodes
        nodes_path = self.working_dir / "graph_nodes.json"
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(self.nodes, f, indent=2, ensure_ascii=False)

        # Save edges
        edges_path = self.working_dir / "graph_edges.json"
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.edges), f, indent=2, ensure_ascii=False)

        # Save embeddings
        if self.embeddings:
            embeddings_path = self.working_dir / "graph_embeddings.npz"
            terms_list = list(self.embeddings.keys())
            embeddings_array = np.array([self.embeddings[t] for t in terms_list])
            np.savez(embeddings_path, terms=terms_list, embeddings=embeddings_array)

        print(
            f"✓ Graph saved: {len(self.nodes)} nodes, {sum(len(e) for e in self.edges.values())} edges"
        )

    def load(self) -> bool:
        """Load graph data"""
        nodes_path = self.working_dir / "graph_nodes.json"
        edges_path = self.working_dir / "graph_edges.json"

        if not nodes_path.exists():
            return False

        with open(nodes_path, "r", encoding="utf-8") as f:
            self.nodes = json.load(f)

        if edges_path.exists():
            with open(edges_path, "r", encoding="utf-8") as f:
                edges_data = json.load(f)
                self.edges = defaultdict(list, edges_data)

        # Load embeddings
        embeddings_path = self.working_dir / "graph_embeddings.npz"
        if embeddings_path.exists():
            data = np.load(embeddings_path, allow_pickle=True)
            terms_list = data["terms"]
            embeddings_array = data["embeddings"]
            self.embeddings = {
                term: embedding for term, embedding in zip(terms_list, embeddings_array)
            }

        print(f"✓ Loaded graph: {len(self.nodes)} nodes")
        return True


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
class AdaptiveDictionaryBuilder:
    """Main builder class"""

    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.analytics_loader = AnalyticsLoader(TERM_FOLDER, tokenizer_name)
        self.dictionary = DictionaryManager(WORKING_DIR)
        self.graph = GraphManager(WORKING_DIR)
        self.technical_terms: Set[str] = set()

    async def build(self, book_path: str) -> bool:
        """Build dictionary and graph from book"""
        print(f"\n{'='*60}")
        print("ADAPTIVE DICTIONARY BUILDER")
        print(f"{'='*60}")
        print(f"Tokenizer: {self.tokenizer_name}")
        print(f"Book: {book_path}")
        print(f"{'='*60}\n")

        # Step 1: Load book
        if not Path(book_path).exists():
            print(f"✗ Book file not found: {book_path}")
            return False

        with open(book_path, "r", encoding="utf-8") as f:
            book_text = f.read()
        print(f"✓ Loaded book: {len(book_text)} characters\n")

        # Step 2: Load technical terms
        self.technical_terms = self.analytics_loader.load_terms()

        # Step 3: Create chunks
        chunker = BookChunker(self.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = chunker.create_chunks(book_text)

        # Step 4: Process chunks
        matcher = TermMatcher(self.technical_terms)
        processor = ChunkProcessor(matcher)

        all_definitions = {}
        all_partial_info = {}

        if MAX_CONCURRENT_DEFINITIONS <= 1:
            # Sequential processing
            for chunk in chunks:
                result = await processor.process_chunk(chunk)
                all_definitions.update(result["definitions"])
                all_partial_info.update(result["partial_info"])
        else:
            # Parallel processing
            tasks = [processor.process_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)

            for result in results:
                all_definitions.update(result["definitions"])
                all_partial_info.update(result["partial_info"])

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total definitions generated: {len(all_definitions)}")
        print(f"Partial match relationships: {len(all_partial_info)}")

        # Step 5: Add to dictionary
        print(f"\n{'='*60}")
        print("BUILDING DICTIONARY")
        print(f"{'='*60}")
        self.dictionary.add_terms(all_definitions)

        # Step 6: Build graph
        print(f"\n{'='*60}")
        print("BUILDING GRAPH")
        print(f"{'='*60}")
        self.graph.add_terms(all_definitions)
        self.graph.add_compound_relationships(all_partial_info)
        self.graph.compute_embeddings()
        self.graph.build_relationships()

        # Step 7: Build GraphRAG (optional)
        print(f"\n{'='*60}")
        print("BUILDING GRAPHRAG")
        print(f"{'='*60}")
        await self.graph.build_graphrag()

        return True

    def save(self) -> None:
        """Save all data"""
        print(f"\n{'='*60}")
        print("SAVING DATA")
        print(f"{'='*60}")
        self.dictionary.save()
        self.graph.save()

        # Save metadata
        metadata_path = Path(WORKING_DIR) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer": self.tokenizer_name,
                    "total_terms": len(self.dictionary.dictionary),
                    "technical_terms_loaded": len(self.technical_terms),
                },
                f,
                indent=2,
            )
        print("✓ All data saved")


# ============================================================================
# QUERY SYSTEM
# ============================================================================
class QuerySystem:
    """Query interface for dictionary and graph"""

    def __init__(self):
        self.dictionary = DictionaryManager(WORKING_DIR)
        self.graph = GraphManager(WORKING_DIR)
        self.graphrag: Optional[GraphRAG] = None

    def load_data(self) -> bool:
        """Load dictionary and graph data"""
        print(f"Loading data from {WORKING_DIR}...")

        dict_loaded = self.dictionary.load()
        graph_loaded = self.graph.load()

        if not dict_loaded:
            print("✗ No dictionary data found")
            return False

        if not graph_loaded:
            print("⚠ No graph data found")

        # Try to load GraphRAG
        try:
            self.graphrag = GraphRAG(
                working_dir=WORKING_DIR,
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
            )
            print("✓ GraphRAG loaded")
        except Exception as e:
            print(f"⚠ GraphRAG not available: {e}")

        return True

    def dict_lookup(self, term: str) -> Dict[str, Any]:
        """Dictionary lookup only (exact or fuzzy)"""
        # Try exact match
        exact = self.dictionary.lookup_exact(term)

        if exact:
            return {
                "found": True,
                "exact_match": True,
                "term": term,
                "definition": exact,
            }

        # Fuzzy search
        fuzzy_results = self.dictionary.lookup_fuzzy(term, top_n=10)

        return {
            "found": False,
            "exact_match": False,
            "query": term,
            "fuzzy_matches": fuzzy_results,
            "message": f"No exact match for '{term}'. Showing similar terms:",
        }

    async def full_query(self, term: str) -> Dict[str, Any]:
        """Full query: dict → graph (Pipeline C)"""
        # Step 1: Try dictionary
        dict_result = self.dict_lookup(term)

        if dict_result["exact_match"]:
            # Found in dictionary, also get graph context
            graph_result = self.graph.query_graph(term, max_depth=2)

            return {
                "method": "exact_match",
                "term": term,
                "definition": dict_result["definition"],
                "graph_context": graph_result,
            }

        # Step 2: Not found - Pipeline C
        print(f"\n{'='*60}")
        print("PIPELINE C: ADVANCED SEMANTIC SEARCH")
        print(f"{'='*60}")

        # Get fuzzy matches
        fuzzy_matches = dict_result["fuzzy_matches"][:5]
        print(f"Found {len(fuzzy_matches)} fuzzy matches")

        # Query graph for each fuzzy match
        graph_results = []
        for match in fuzzy_matches:
            graph_data = self.graph.query_graph(match["term"], max_depth=2)
            if graph_data["found"]:
                graph_results.append(
                    {
                        "fuzzy_term": match["term"],
                        "similarity": match["similarity"],
                        "graph_data": graph_data,
                    }
                )

        # Query GraphRAG if available
        graphrag_result = None
        if self.graphrag:
            try:
                print("Querying GraphRAG...")
                graphrag_result = await self.graphrag.aquery(
                    term, param=QueryParam(mode="local")
                )
            except Exception as e:
                print(f"⚠ GraphRAG query failed: {e}")

        return {
            "method": "pipeline_c",
            "query": term,
            "found_exact": False,
            "fuzzy_matches": fuzzy_matches,
            "graph_results": graph_results,
            "graphrag_result": graphrag_result,
            "message": "No exact match found. Showing related context:",
        }


# ============================================================================
# MAIN CLI
# ============================================================================
async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py build <book_file> <tokenizer_name>")
        print("  python main.py dict <term>")
        print("  python main.py query <term>")
        print("\nExamples:")
        print("  python main.py build power_systems.txt sudachi")
        print("  python main.py dict 高圧送電線")
        print("  python main.py query 直流送電")
        print("\nTokenizer options: sudachi, mecab")
        return

    command = sys.argv[1].lower()

    # ========================================================================
    # BUILD COMMAND
    # ========================================================================
    if command == "build":
        if len(sys.argv) < 4:
            print("Usage: python main.py build <book_file> <tokenizer_name>")
            print("Example: python main.py build book.txt sudachi")
            return

        book_file = sys.argv[2]
        tokenizer_name = sys.argv[3]

        builder = AdaptiveDictionaryBuilder(tokenizer_name)
        success = await builder.build(book_file)

        if success:
            builder.save()
            print(f"\n{'='*60}")
            print("BUILD COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")

    # ========================================================================
    # DICT COMMAND
    # ========================================================================
    elif command == "dict":
        if len(sys.argv) < 3:
            print("Usage: python main.py dict <term>")
            return

        term = sys.argv[2]
        query_system = QuerySystem()

        if not query_system.load_data():
            print("\n✗ No data found. Build first with:")
            print("  python main.py build <book_file> <tokenizer_name>")
            return

        result = query_system.dict_lookup(term)

        print(f"\n{'='*60}")
        print("DICTIONARY LOOKUP")
        print(f"{'='*60}")

        if result["exact_match"]:
            print(f"\n✓ EXACT MATCH FOUND")
            print(f"\nTerm: {result['term']}")
            print(f"Definition: {result['definition']}")
        else:
            print(f"\nQuery: {result['query']}")
            print(f"{result['message']}\n")

            for i, match in enumerate(result["fuzzy_matches"], 1):
                print(f"{i}. {match['term']} (similarity: {match['similarity']:.3f})")
                print(f"   {match['definition']}\n")

    # ========================================================================
    # QUERY COMMAND
    # ========================================================================
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <term>")
            return

        term = sys.argv[2]
        query_system = QuerySystem()

        if not query_system.load_data():
            print("\n✗ No data found. Build first with:")
            print("  python main.py build <book_file> <tokenizer_name>")
            return

        result = await query_system.full_query(term)

        print(f"\n{'='*60}")
        print("FULL QUERY RESULTS")
        print(f"{'='*60}")

        if result["method"] == "exact_match":
            # Exact match found
            print(f"\n✓ EXACT MATCH FOUND")
            print(f"\nTerm: {result['term']}")
            print(f"Definition: {result['definition']}")

            # Show graph context
            graph_ctx = result["graph_context"]
            if graph_ctx.get("found"):
                print(f"\n{'─'*60}")
                print("GRAPH CONTEXT")
                print(f"{'─'*60}")
                print(f"Related nodes: {graph_ctx['total_nodes']}")
                print(f"Related edges: {graph_ctx['total_edges']}")

                # Show relationships
                print("\nRelationships:")
                edge_types = defaultdict(list)
                for edge in graph_ctx["edges"]:
                    edge_types[edge["type"]].append(
                        f"{edge['source']} → {edge['target']}"
                    )

                for edge_type, edges in edge_types.items():
                    print(f"\n  {edge_type}:")
                    for edge in edges[:5]:  # Show first 5 of each type
                        print(f"    - {edge}")

        else:
            # Pipeline C results
            print(f"\n⚠ NO EXACT MATCH")
            print(f"Query: {result['query']}")
            print(f"{result['message']}")

            # Fuzzy matches
            print(f"\n{'─'*60}")
            print("FUZZY MATCHES")
            print(f"{'─'*60}")
            for i, match in enumerate(result["fuzzy_matches"], 1):
                print(f"\n{i}. {match['term']} (similarity: {match['similarity']:.3f})")
                print(f"   {match['definition']}")

            # Graph results
            if result["graph_results"]:
                print(f"\n{'─'*60}")
                print("GRAPH ANALYSIS (RAW OUTPUT)")
                print(f"{'─'*60}")

                for graph_res in result["graph_results"]:
                    print(f"\nBased on fuzzy match: {graph_res['fuzzy_term']}")
                    graph_data = graph_res["graph_data"]

                    print(f"  Nodes found: {graph_data['total_nodes']}")
                    print(f"  Edges found: {graph_data['total_edges']}")

                    # Show node details
                    print("\n  Related terms:")
                    for node_term, node_data in list(graph_data["nodes"].items())[:5]:
                        print(f"    - {node_term}: {node_data['definition'][:80]}...")

                    # Show edge details
                    if graph_data["edges"]:
                        print("\n  Relationships:")
                        for edge in graph_data["edges"][:10]:
                            print(
                                f"    {edge['source']} --[{edge['type']}]--> {edge['target']}"
                            )

            # GraphRAG results
            if result.get("graphrag_result"):
                print(f"\n{'─'*60}")
                print("GRAPHRAG SEMANTIC SEARCH")
                print(f"{'─'*60}")
                print(result["graphrag_result"])

    else:
        print(f"✗ Unknown command: {command}")
        print("Use: build, dict, or query")


if __name__ == "__main__":
    asyncio.run(main())
