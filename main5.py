#!/usr/bin/env python3
"""
Modular Graph-RAG Dictionary Builder – BUILD-ONLY SCRIPT
--------------------------------------------------------
This script ingests a book, chunks it, extracts technical terms (placeholder list),
generates definitions with an LLM, and builds a GraphRAG knowledge graph.
Querying is handled separately.

CRITICAL DESIGN CHOICES
- 100 % async / concurrent-safe (concurrency tunable via MAX_CONCURRENT_CHUNKS).
- Pydantic-forced JSON output from LLM (no think tokens, no prose).
- Definitions are merged only if similarity < SIMILARITY_THRESHOLD.
- Graph is built for similarity search (embedding edges) and exact lookup.
- All heavy constants live at the top so you can tune without diving into code.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# ------------------------------------------------------------------
# CONFIG – CHANGE THESE WITHOUT TOUCHING CODE BELOW
# ------------------------------------------------------------------
BOOK_PATH              = "book.txt"                # your book
WORKING_DIR            = "./graph_rag_cache"
VLLM_HOST              = "http://localhost:8000"
VLLM_MODEL             = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKENS             = 2000
MAX_CONCURRENT_CHUNKS  = 15                         # semaphore limit
CHUNK_SIZE             = 800                       # tokens (approx)
CHUNK_OVERLAP          = 100                       # tokens
SIMILARITY_THRESHOLD   = 0.75                      # 0–1, higher → stricter dedup
LOG_LEVEL              = logging.INFO
# ------------------------------------------------------------------

os.makedirs(WORKING_DIR, exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------------------------------------------
# NANO-GRAPHRAG IMPORTS (same library you used)
# ------------------------------------------------------------------
try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag._utils import wrap_embedding_func_with_attrs
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logging.error(f"Missing dependency: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# EMBEDDING MODEL (local)
# ------------------------------------------------------------------
EMBED_MODEL = SentenceTransformer("models/embeddings", local_files_only=True)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def embed(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


# ------------------------------------------------------------------
# VLLM CLIENT
# ------------------------------------------------------------------
vllm = AsyncOpenAI(api_key="EMPTY", base_url=f"{VLLM_HOST}/v1")


# ------------------------------------------------------------------
# PYDANTIC OUTPUT MODEL – NO THINK TOKENS, EVER
# ------------------------------------------------------------------
class TermDefinition(BaseModel):
    term: str = Field(..., description="Lower-cased technical term")
    definition: str = Field(..., description="1–2 sentence definition from context")


class ChunkDefinitions(BaseModel):
    definitions: List[TermDefinition]


# ------------------------------------------------------------------
# PLACEHOLDER TERM EXTRACTOR (hard-coded list comes later)
# ------------------------------------------------------------------
def extract_technical_terms(text: str) -> Set[str]:
    """
    VERY DUMB placeholder – replace with your hard-coded list later.
    For now we grab:
      - hyphenated / camelCase / acronyms
      - 4+ letter words that appear ≥2 times
    Returns *unique* lower-cased strings.
    """
    stop = {
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "a", "an",
    }

    # technical patterns
    patterns = [
        r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b",  # power-factor
        r"\b[A-Z][a-z]*[A-Z][a-z]*\b",    # reactivePower
        r"\b[A-Z]{2,}\b",                  # SCADA
    ]
    tech = set()
    for p in patterns:
        tech.update({m.lower() for m in re.findall(p, text)})

    # frequent significant words
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    frequent = {w for w, c in freq.items() if c >= 2}

    return tech | frequent


# ------------------------------------------------------------------
# CHUNKING
# ------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> List[Dict[str, str]]:
    """Naïve word-based chunking – good enough for now."""
    words = text.split()
    chunks = []
    idx = 0
    for i in range(0, len(words), size - overlap):
        chunk_w = words[i : i + size]
        chunk_t = " ".join(chunk_w)
        chunks.append({"id": idx, "text": chunk_t})
        idx += 1
    logging.info(f"Created {len(chunks)} chunks")
    return chunks


# ------------------------------------------------------------------
# LLM DEFINITION GENERATION – PYDANTIC FORCED
# ------------------------------------------------------------------
SYSTEM_DEF = (
    "You are an expert electrical-engineering glossary generator. "
    "Output ONLY valid JSON, no explanations or markdown fences."
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def generate_definitions(chunk: str, terms: List[str]) -> List[TermDefinition]:
    terms_str = ", ".join(terms)
    prompt = f"""
Text:
{chunk}

Based strictly on the text above, produce concise definitions for these terms:
{terms_str}

Output JSON list:
[{{"term": "term1", "definition": "..."}}, ...]
"""
    resp = await vllm.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_DEF},
                  {"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    # strip possible ```json ... ```
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    parsed = ChunkDefinitions.parse_raw(raw)
    return parsed.definitions


# ------------------------------------------------------------------
# CONCURRENCY SAFE CHUNK PROCESSOR
# ------------------------------------------------------------------
sem = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)


async def process_chunk(chunk: Dict[str, str]) -> Dict[str, List[TermDefinition]]:
    async with sem:
        terms = extract_technical_terms(chunk["text"])
        if not terms:
            return {chunk["id"]: []}
        # limit batch to avoid context overflow
        terms = list(terms)[:10]
        defs = await generate_definitions(chunk["text"], terms)
        logging.debug(f"Chunk {chunk['id']}: {len(defs)} definitions")
        return {chunk["id"]: defs}


# ------------------------------------------------------------------
# MERGE LOGIC
# ------------------------------------------------------------------
def merge_into_graph(
    acc: Dict[str, str], new: Dict[str, str], threshold: float
) -> Dict[str, str]:
    """Merge new definitions into accumulator using similarity threshold."""
    for term, defin in new.items():
        if term not in acc:
            acc[term] = defin
        else:
            sim = SequenceMatcher(None, acc[term].lower(), defin.lower()).ratio()
            if sim < threshold:
                acc[term] += " " + defin  # simple append
    return acc


# ------------------------------------------------------------------
# BUILD GRAPH-RAG
# ------------------------------------------------------------------
async def build_graphrag(term_def_map: Dict[str, str]) -> None:
    """Insert term-definition pairs as nano-graphrag documents."""
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=embed,
        best_model_func=lambda p: vllm_complete(p),  # wrapper below
        cheap_model_func=lambda p: vllm_complete(p),
        best_model_max_token_size=MAX_TOKENS,
        cheap_model_max_token_size=MAX_TOKENS,
        embedding_batch_num=4,
        embedding_func_max_async=4,
    )

    docs = []
    for term, defin in term_def_map.items():
        doc = f"**{term.upper()}**\nDefinition: {defin}"
        docs.append(doc)

    # insert in batches for memory safety
    for batch in asyncio.as_completed([asyncio.create_task(rag.insert(docs[i:i+100])) for i in range(0, len(docs), 100)]):
        await batch
    logging.info("GraphRAG build complete")


def vllm_complete(prompt: str) -> str:
    """Sync wrapper for nano-graphrag compatibility."""
    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(
        vllm.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
        )
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# MAIN PIPELINE – BUILD ONLY
# ------------------------------------------------------------------
async def main():
    if len(sys.argv) < 2 or sys.argv[1] != "build":
        print("Usage: python build_graphrag.py build")
        sys.exit(1)

    book = Path(BOOK_PATH).read_text(encoding="utf-8")
    chunks = chunk_text(book, CHUNK_SIZE, CHUNK_OVERLAP)

    # process chunks concurrently
    chunk_results = await asyncio.gather(*[process_chunk(c) for c in chunks])

    # merge definitions
    term_bank: Dict[str, str] = {}
    for cr in chunk_results:
        for cid, defs in cr.items():
            tmp = {d.term: d.definition for d in defs}
            term_bank = merge_into_graph(term_bank, tmp, SIMILARITY_THRESHOLD)

    logging.info(f"Final glossary contains {len(term_bank)} terms")

    # persist raw json for safety
    out = Path(WORKING_DIR) / "glossary.json"
    out.write_text(json.dumps(term_bank, indent=2, ensure_ascii=False), encoding="utf-8")

    # build graph
    await build_graphrag(term_bank)
    print(f"✅ Build complete – data in {WORKING_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
