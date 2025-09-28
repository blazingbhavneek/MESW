# Modular Book Dictionary Builder - Complete Clean Version
# Processes entire book through chunks, extracts terms, generates definitions, builds GraphRAG

import sys
import logging
import numpy as np
import asyncio
import os
import json
import re
import time
from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import Counter, defaultdict
from difflib import get_close_matches, SequenceMatcher
from abc import ABC, abstractmethod

# Configuration
MAX_CONCURRENT_DEFINITIONS = 15
WORKING_DIR = "./book_dictionary_cache"
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKEN = 1000

# Fix scipy.linalg.triu issue
def patch_scipy_triu():
    try:
        import scipy.linalg
        if not hasattr(scipy.linalg, 'triu'):
            print("🔧 Patching scipy.linalg.triu with numpy.triu...")
            scipy.linalg.triu = np.triu
            print("✅ Successfully patched scipy.linalg.triu")
    except ImportError:
        pass

patch_scipy_triu()

# Imports after patch
try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag._utils import wrap_embedding_func_with_attrs
    from sentence_transformers import SentenceTransformer
    import json_repair
    print("✅ Dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Initialize models
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
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
    try:
        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKEN,
            temperature=0.1,  # Lowered for more direct, deterministic output
            extra_body={
                "repetition_penalty": 1.1,
                "top_p": 0.9,
            },
            # enable_thinking=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"vLLM request failed: {e}")
        return ""

# ============================================================================
# ABSTRACT BASE CLASS FOR TERM EXTRACTORS
# ============================================================================

class BaseTermExtractor(ABC):
    """Abstract base class for term extractors"""
    
    @abstractmethod
    def extract_terms(self, text: str, **kwargs) -> List[str]:
        """Extract terms from text and return list of terms"""
        pass
    
    @abstractmethod
    def get_language(self) -> str:
        """Return the language this extractor handles"""
        pass

# ============================================================================
# ENGLISH TERM EXTRACTOR
# ============================================================================

class EnglishTermExtractor(BaseTermExtractor):
    """Extract English technical terms and vocabulary from text"""
    
    def __init__(self):
        self.language = "english"
        self.stop_words = self._get_stopwords()
    
    def get_language(self) -> str:
        return self.language
    
    def _get_stopwords(self) -> Set[str]:
        """Get English stopwords"""
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))
        except:
            # Fallback stopwords
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 
                'them', 'my', 'your', 'his', 'our', 'their', 'this', 'that', 'these', 'those'
            }
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _simple_word_tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        words = re.findall(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b", text)
        return words
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical and compound terms"""
        technical_terms = set()
        
        # Patterns for technical terms
        patterns = [
            r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b',  # hyphenated terms
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',   # proper noun phrases
            r'\b[a-z]+[A-Z][a-z]*\b',         # camelCase terms
            r'\b[A-Z]{2,}\b',                 # acronyms
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                term = match.lower().strip()
                if len(term) > 2 and term not in self.stop_words:
                    technical_terms.add(term)
        
        return list(technical_terms)
    
    def _extract_frequent_terms(self, text: str, min_frequency: int = 3) -> List[str]:
        """Extract frequently occurring significant terms"""
        sentences = self._simple_sentence_split(text)
        
        all_words = []
        for sentence in sentences:
            words = self._simple_word_tokenize(sentence.lower())
            significant_words = [
                w for w in words 
                if w.isalpha() and len(w) > 3 and w not in self.stop_words
            ]
            all_words.extend(significant_words)
        
        word_freq = Counter(all_words)
        frequent_terms = [word for word, freq in word_freq.items() if freq >= min_frequency]
        
        return frequent_terms
    
    def extract_terms(self, text: str, min_frequency: int = 3, max_terms: int = 200) -> List[str]:
        """
        Extract terms from English text
        Returns: List of unique terms
        """
        print(f"🔍 Extracting English terms from text ({len(text)} characters)...")
        
        # Extract different types of terms
        technical_terms = self._extract_technical_terms(text)
        frequent_terms = self._extract_frequent_terms(text, min_frequency)
        
        # Combine and deduplicate
        all_terms = list(set(technical_terms + frequent_terms))
        
        # Sort by length and frequency preference (technical terms first)
        def term_priority(term):
            is_technical = term in technical_terms
            return (is_technical, len(term), term)
        
        all_terms.sort(key=term_priority, reverse=True)
        
        # Limit results
        selected_terms = all_terms[:max_terms]
        
        print(f"✅ Extracted {len(selected_terms)} English terms")
        return selected_terms

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
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'id': len(chunks),
                'text': chunk_text,
                'start_token': i,
                'end_token': i + len(chunk_tokens),
                'token_count': len(chunk_tokens),
                'char_start': self._find_char_position(text, chunk_text, i > 0),
                'char_end': self._find_char_position(text, chunk_text, i > 0) + len(chunk_text)
            })
        
        print(f"✅ Created {len(chunks)} token-based chunks")
        return chunks
    
    def _create_word_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks based on word count (fallback)"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': len(chunks),
                'text': chunk_text,
                'start_word': i,
                'end_word': i + len(chunk_words),
                'word_count': len(chunk_words),
                'char_start': self._find_char_position(text, chunk_text, i > 0),
                'char_end': self._find_char_position(text, chunk_text, i > 0) + len(chunk_text)
            })
        
        print(f"✅ Created {len(chunks)} word-based chunks")
        return chunks
    
    def _find_char_position(self, full_text: str, chunk_text: str, is_continuation: bool) -> int:
        """Find character position of chunk in full text"""
        if not is_continuation:
            return 0
        
        # Find first few words to locate position
        first_words = ' '.join(chunk_text.split()[:5])
        position = full_text.find(first_words)
        return max(0, position)

# ============================================================================
# CHUNK PROCESSOR
# ============================================================================

class ChunkProcessor:
    """Process individual chunks to find terms and generate definitions"""
    
    def __init__(self, term_extractor: BaseTermExtractor):
        self.term_extractor = term_extractor
        self.chunk_terms = {}  # chunk_id -> terms found
        self.chunk_definitions = {}  # chunk_id -> {term: definition}
        self.global_terms = set()  # all unique terms found
    
    async def process_chunk(self, chunk: Dict[str, Any], 
                           given_terms: List[str] = None) -> Dict[str, Any]:
        """
        Process a single chunk:
        1. Extract new terms OR check for given terms
        2. Generate definitions for terms found in this chunk
        3. Handle technical terms that might be missed
        """
        chunk_id = chunk['id']
        chunk_text = chunk['text']
        
        print(f"🔍 Processing chunk {chunk_id} ({chunk.get('token_count', chunk.get('word_count', 0))} tokens/words)")
        
        # Step 1: Find terms in this chunk
        if given_terms:
            # Check which given terms are present in this chunk
            present_terms = []
            for term in given_terms:
                if term.lower() in chunk_text.lower():
                    present_terms.append(term)
            chunk_terms = present_terms
            print(f"  📋 Found {len(present_terms)} given terms in chunk")
        else:
            # Extract new terms from this chunk
            chunk_terms = self.term_extractor.extract_terms(chunk_text, min_frequency=1, max_terms=50)
            print(f"  🔍 Extracted {len(chunk_terms)} new terms from chunk")
        
        # Step 2: Generate definitions for terms in this chunk
        definitions = {}
        if chunk_terms:
            definitions = await self._generate_chunk_definitions(
                chunk_text, chunk_terms, chunk_id
            )
        
        # Step 3: Ask LLM to identify missed technical terms
        additional_terms = await self._find_missed_technical_terms(
            chunk_text, chunk_terms
        )
        
        if additional_terms:
            print(f"  💡 Found {len(additional_terms)} additional technical terms")
            additional_definitions = await self._generate_chunk_definitions(
                chunk_text, additional_terms, chunk_id, is_technical=True
            )
            definitions.update(additional_definitions)
            chunk_terms.extend(additional_terms)
        
        # Store results
        self.chunk_terms[chunk_id] = chunk_terms
        self.chunk_definitions[chunk_id] = definitions
        self.global_terms.update(chunk_terms)
        
        return {
            'chunk_id': chunk_id,
            'terms_found': len(chunk_terms),
            'definitions_generated': len(definitions),
            'terms': chunk_terms,
            'definitions': definitions
        }
    
    async def _generate_chunk_definitions(self, chunk_text: str, terms: List[str], 
                                        chunk_id: int, is_technical: bool = False) -> Dict[str, str]:
        """Generate definitions for terms found in a specific chunk"""
        definitions = {}
        
        # Process in smaller batches to avoid overwhelming the LLM
        batch_size = 5
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i:i + batch_size]
            
            prompt = self._create_definition_prompt(chunk_text, batch_terms, is_technical)
            
            try:
                response = await vllm_complete(prompt, self._get_definition_system_prompt())
                batch_definitions = self._parse_definitions_response(response, batch_terms)
                definitions.update(batch_definitions)
                
                print(f"    ✅ Chunk {chunk_id}: Generated definitions for batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"    ❌ Failed to generate definitions for chunk {chunk_id}, batch {i//batch_size + 1}: {e}")
                # Fallback definitions
                for term in batch_terms:
                    definitions[term] = f"Term from chunk {chunk_id} context: {chunk_text[:200]}..."
        
        return definitions
        
    def _create_definition_prompt(self, chunk_text: str, terms: List[str], is_technical: bool) -> str:
        """Create prompt for generating definitions from chunk context"""
        term_list = ", ".join(terms)
        technical_note = " (these are technical terms that may need specialized definitions)" if is_technical else ""
        
        return f"""Based on the following text chunk, provide clear, concise definitions for these terms{technical_note}:

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
        return """You are an expert at creating clear, contextual definitions for terms found in books. 
    Based on the provided text chunk, create precise definitions for the requested terms. 
    Use ONLY the context provided. Keep definitions to 1-2 sentences.
    Follow the exact format requested: TERM: [name] followed by DEFINITION: [definition].
    Do not include any reasoning, explanations, or additional text outside the format."""

    def _parse_definitions_response(self, response: str, expected_terms: List[str]) -> Dict[str, str]:
        """Parse the LLM response to extract definitions"""
        definitions = {}
        
        # Split response into sections
        sections = re.split(r'\n(?=TERM:)', response)
        
        for section in sections:
            # Extract term and definition
            term_match = re.search(r'TERM:\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
            def_match = re.search(r'DEFINITION:\s*(.+?)(?=\nTERM:|\n\n|$)', section, re.IGNORECASE | re.DOTALL)
            
            if term_match and def_match:
                term = term_match.group(1).strip().lower()
                definition = def_match.group(1).strip()
                
                # Clean up the definition
                definition = re.sub(r'\s+', ' ', definition)
                if definition and not definition.endswith(('.', '!', '?')):
                    definition += '.'
                
                definitions[term] = definition
        
        # Ensure we got definitions for the expected terms
        missing_terms = [t for t in expected_terms if t.lower() not in definitions]
        if missing_terms:
            print(f"    ⚠️ Missing definitions for: {missing_terms}")
        
        return definitions
    
    async def _find_missed_technical_terms(self, chunk_text: str, found_terms: List[str]) -> List[str]:
        """Ask LLM to identify technical terms that might have been missed"""
        found_terms_str = ", ".join(found_terms) if found_terms else "none"
        
        prompt = f"""Analyze this text chunk and identify any technical terms, specialized vocabulary, or important concepts that might have been missed.

Already identified terms: {found_terms_str}

Text chunk:
{chunk_text[:1000]}

List any additional technical terms, proper nouns, or specialized concepts that appear important but weren't in the already identified terms. Focus on:
- Technical jargon
- Specialized terminology  
- Important proper nouns (names, places, organizations)
- Domain-specific concepts
- Abbreviations or acronyms

Respond with just a comma-separated list of terms, or "NONE" if no additional terms are found."""
        
        system_prompt = """You are an expert at identifying technical and specialized terms in text. 
Look for terms that would benefit from definition or explanation. 
Be selective - only identify truly important or technical terms.
Do not include any reasoning or extra text. Output only the comma-separated list or "NONE"."""
        
        try:
            response = await vllm_complete(prompt, system_prompt)
            response = response.strip()
            
            if response.upper() == "NONE" or not response:
                return []
            
            # Parse comma-separated terms
            additional_terms = [term.strip() for term in response.split(',')]
            additional_terms = [term for term in additional_terms if term and len(term) > 2]
            
            # Filter out terms we already found
            found_lower = [t.lower() for t in found_terms]
            additional_terms = [term for term in additional_terms if term.lower() not in found_lower]
            
            return additional_terms[:10]  # Limit to prevent overwhelming
            
        except Exception as e:
            print(f"    ⚠️ Failed to find additional technical terms: {e}")
            return []

# ============================================================================
# TERM CONTEXT MANAGER
# ============================================================================

class TermContextManager:
    """Manage term contexts from processed chunks"""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.chunk_lookup = {chunk['id']: chunk for chunk in chunks}
    
    def compile_term_contexts(self, chunk_processor: ChunkProcessor) -> Dict[str, List[str]]:
        """Compile contexts for all terms from all chunks"""
        print(f"📚 Compiling contexts for all terms from {len(self.chunks)} chunks...")
        
        term_contexts = defaultdict(list)
        
        # Go through each chunk and collect contexts for each term
        for chunk_id, terms in chunk_processor.chunk_terms.items():
            chunk = self.chunk_lookup[chunk_id]
            chunk_text = chunk['text']
            
            for term in terms:
                # Create a focused context around the term
                context = self._extract_term_context(chunk_text, term)
                if context:
                    term_contexts[term].append({
                        'context': context,
                        'chunk_id': chunk_id,
                        'chunk_info': f"Chunk {chunk_id} (chars {chunk.get('char_start', 0)}-{chunk.get('char_end', 0)})"
                    })
        
        # Convert to simple format and limit contexts per term
        simplified_contexts = {}
        for term, context_list in term_contexts.items():
            contexts = [ctx['context'] for ctx in context_list[:5]]  # Max 5 contexts per term
            if contexts:
                simplified_contexts[term] = contexts
        
        print(f"✅ Compiled contexts for {len(simplified_contexts)} terms")
        return simplified_contexts
    
    def _extract_term_context(self, chunk_text: str, term: str, context_size: int = 200) -> str:
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

# ============================================================================
# GRAPH MANAGER
# ============================================================================

class GraphManager:
    """Manage the knowledge graph and entity relationships"""
    
    def __init__(self, working_dir: str = WORKING_DIR):
        self.working_dir = working_dir
        self.graph_data = {}
        self.entity_definitions = {}
        
    def add_or_update_entities(self, terms_with_definitions: Dict[str, str], 
                              terms_with_contexts: Dict[str, List[str]]) -> Dict[str, Any]:
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
                
                if similarity < 0.7:  # Definitions are different
                    similar_entities.append({
                        'term': term,
                        'existing_definition': existing_def,
                        'new_definition': definition,
                        'similarity': similarity
                    })
                # If very similar (>= 0.7), do nothing
            else:
                # New entity
                self.entity_definitions[term] = definition
                new_entities.append(term)
                
                # Find similar terms for connections
                similar_terms = self._find_similar_terms(term)
                if similar_terms:
                    updates.append({
                        'term': term,
                        'definition': definition,
                        'similar_terms': similar_terms,
                        'contexts': contexts
                    })
        
        print(f"✅ New entities: {len(new_entities)}")
        print(f"🔄 Similar entities found: {len(similar_entities)}")
        
        return {
            'new_entities': new_entities,
            'updates': updates,
            'similar_entities': similar_entities,
            'total_entities': len(self.entity_definitions)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two definitions"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _find_similar_terms(self, term: str, threshold: float = 0.6) -> List[str]:
        """Find similar terms in existing entities"""
        similar = []
        existing_terms = list(self.entity_definitions.keys())
        
        matches = get_close_matches(term, existing_terms, n=5, cutoff=threshold)
        return matches
    
    def create_graph_chunks(self, terms_with_definitions: Dict[str, str], 
                           terms_with_contexts: Dict[str, List[str]]) -> List[str]:
        """Create chunks for GraphRAG"""
        chunks = []
        
        for term, definition in terms_with_definitions.items():
            contexts = terms_with_contexts.get(term, [])
            
            chunk_lines = [
                f"**{term.title()}**",
                f"Definition: {definition}"
            ]
            
            if contexts:
                chunk_lines.append("Contexts:")
                for i, context in enumerate(contexts[:2], 1):
                    clean_context = context.strip()[:200]
                    chunk_lines.append(f"{i}. {clean_context}")
            
            chunk_text = '\n'.join(chunk_lines)
            chunks.append(chunk_text)
        
        print(f"📚 Created {len(chunks)} graph chunks")
        return chunks

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class BookDictionaryBuilder:
    """Main class that orchestrates the entire process"""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.term_extractor = self._get_extractor(language)
        self.context_manager = None
        self.graph_manager = GraphManager()
    
    def _get_extractor(self, language: str) -> BaseTermExtractor:
        """Get appropriate term extractor based on language"""
        if language.lower() == "english":
            return EnglishTermExtractor()
        else:
            raise ValueError(f"Language '{language}' not supported yet")
    
    async def process_book_comprehensive(self, text: str, 
                                       chunk_size: int = 1000, 
                                       overlap: int = 200,
                                       use_global_terms: bool = False,
                                       **kwargs) -> Dict[str, Any]:
        """
        Main processing pipeline that processes the entire book through chunks
        
        Args:
            text: Full book text
            chunk_size: Size of each chunk (tokens or words)
            overlap: Overlap between chunks
            use_global_terms: If True, extract terms globally first, then check each chunk
                             If False, extract terms from each chunk independently
        """
        print(f"📚 Processing entire book comprehensively with {self.language} term extractor...")
        print("="*60)
        
        # Step 1: Create chunks from the entire book
        print("\nSTEP 1: CREATING BOOK CHUNKS")
        print("-" * 30)
        chunker = BookChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.create_chunks(text)
        
        # Step 2: Decide on term extraction strategy
        print(f"\nSTEP 2: TERM EXTRACTION STRATEGY")
        print("-" * 30)
        global_terms = []
        if use_global_terms:
            print("🌐 Extracting terms globally from entire book first...")
            global_terms = self.term_extractor.extract_terms(text, **kwargs)
            print(f"✅ Extracted {len(global_terms)} global terms")
        else:
            print("📝 Will extract terms from each chunk individually...")
        
        # Step 3: Process each chunk
        print(f"\nSTEP 3: PROCESSING ALL CHUNKS")
        print("-" * 30)
        chunk_processor = ChunkProcessor(self.term_extractor)
        
        # Process chunks with progress tracking
        chunk_results = []
        total_terms = set()
        total_definitions = {}
        
        for i, chunk in enumerate(chunks):
            print(f"\n📖 Chunk {i+1}/{len(chunks)}:")
            
            if use_global_terms:
                # Check which global terms are in this chunk
                result = await chunk_processor.process_chunk(chunk, given_terms=global_terms)
            else:
                # Extract terms from this specific chunk
                result = await chunk_processor.process_chunk(chunk)
            
            chunk_results.append(result)
            total_terms.update(result['terms'])
            total_definitions.update(result['definitions'])
            
            print(f"  📊 Chunk summary: {result['terms_found']} terms, {result['definitions_generated']} definitions")
        
        # Step 4: Compile contexts from all chunks
        print(f"\nSTEP 4: COMPILING TERM CONTEXTS")
        print("-" * 30)
        self.context_manager = TermContextManager(chunks)
        terms_with_contexts = self.context_manager.compile_term_contexts(chunk_processor)
        
        # Step 5: Merge and deduplicate definitions
        print(f"\nSTEP 5: MERGING DEFINITIONS")
        print("-" * 30)
        merged_definitions = await self._merge_chunk_definitions(total_definitions, terms_with_contexts)
        
        # Step 6: Process graph entities
        print(f"\nSTEP 6: PROCESSING GRAPH")
        print("-" * 30)
        graph_result = self.graph_manager.add_or_update_entities(
            merged_definitions,
            terms_with_contexts
        )
        
        # Step 7: Handle conflicts/questions
        print(f"\nSTEP 7: REVIEWING CONFLICTS")
        print("-" * 30)
        await self._handle_conflicts(graph_result['similar_entities'])
        
        return {
            'chunks': chunks,
            'chunk_results': chunk_results,
            'terms': list(total_terms),
            'terms_with_contexts': terms_with_contexts,
            'definitions': merged_definitions,
            'graph_result': graph_result,
            'processing_stats': {
                'total_chunks': len(chunks),
                'total_terms': len(total_terms),
                'total_definitions': len(merged_definitions),
                'strategy': 'global_terms' if use_global_terms else 'chunk_by_chunk'
            }
        }
    
    async def _merge_chunk_definitions(self, chunk_definitions: Dict[str, str], 
                                     terms_with_contexts: Dict[str, List[str]]) -> Dict[str, str]:
        """Merge definitions from different chunks, handling conflicts"""
        print(f"🔄 Merging {len(chunk_definitions)} definitions from all chunks...")
        
        # Group definitions by term
        term_definitions = defaultdict(list)
        for term, definition in chunk_definitions.items():
            term_definitions[term].append(definition)
        
        merged_definitions = {}
        conflicts_resolved = 0
        
        for term, definitions in term_definitions.items():
            if len(definitions) == 1:
                # Single definition, use as-is
                merged_definitions[term] = definitions[0]
            else:
                # Multiple definitions, need to merge
                merged_def = await self._resolve_multiple_definitions(term, definitions, terms_with_contexts.get(term, []))
                merged_definitions[term] = merged_def
                conflicts_resolved += 1
        
        print(f"✅ Merged definitions complete. Resolved {conflicts_resolved} conflicts.")
        return merged_definitions
    
    async def _resolve_multiple_definitions(self, term: str, definitions: List[str], contexts: List[str]) -> str:
        """Resolve conflicts when multiple definitions exist for the same term"""
        if len(definitions) <= 1:
            return definitions[0] if definitions else ""
        
        # Check if definitions are similar
        similarities = []
        for i in range(len(definitions)):
            for j in range(i + 1, len(definitions)):
                sim = SequenceMatcher(None, definitions[i].lower(), definitions[j].lower()).ratio()
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        if avg_similarity > 0.7:
            # Definitions are similar, just use the first one
            return definitions[0]
        else:
            # Definitions are different, ask LLM to create a comprehensive definition
            return await self._create_comprehensive_definition(term, definitions, contexts)
    
    async def _create_comprehensive_definition(self, term: str, definitions: List[str], contexts: List[str]) -> str:
        """Create a comprehensive definition from multiple conflicting definitions"""
        definitions_text = "\n".join([f"Definition {i+1}: {def_}" for i, def_ in enumerate(definitions)])
        contexts_text = "\n".join([f"Context {i+1}: {ctx[:200]}..." for i, ctx in enumerate(contexts[:3])])
        
        prompt = f"""Multiple definitions have been found for the term "{term}" from different parts of the book. 
Create a single, comprehensive definition that captures all aspects.

Term: {term}

Multiple definitions found:
{definitions_text}

Additional contexts:
{contexts_text}

Create a comprehensive definition that incorporates the key aspects from all definitions above. 
The definition should be 1-2 sentences and capture the full meaning of the term as used in this book:"""
        
        system_prompt = """You are an expert at synthesizing multiple definitions into comprehensive, accurate definitions. 
Create a single definition that captures all important aspects from the provided definitions while remaining clear and concise."""
        
        try:
            comprehensive_def = await vllm_complete(prompt, system_prompt)
            return comprehensive_def.strip()
        except:
            # Fallback: combine the definitions
            return f"{definitions[0]} {' '.join([d for d in definitions[1:] if not d.lower().startswith(definitions[0][:20].lower())])}"
    
    async def _handle_conflicts(self, similar_entities: List[Dict]) -> None:
        """Handle definition conflicts by asking questions or suggesting alternatives"""
        if not similar_entities:
            print("✅ No conflicts found")
            return
        
        print(f"⚠️ Found {len(similar_entities)} potential conflicts:")
        
        for i, entity in enumerate(similar_entities, 1):
            term = entity['term']
            existing = entity['existing_definition']
            new = entity['new_definition']
            similarity = entity['similarity']
            
            print(f"\n{i}. Term: '{term}' (similarity: {similarity:.2f})")
            print(f"   Existing: {existing[:100]}...")
            print(f"   New:      {new[:100]}...")
            
            # Generate suggestion
            suggestion = await self._generate_conflict_resolution(term, existing, new)
            print(f"   💡 Suggestion: {suggestion}")
            
            # For now, append to existing (could add user interaction here)
            combined_definition = f"{existing} Additionally, {new.lower()}"
            self.graph_manager.entity_definitions[term] = combined_definition
            print(f"   ✅ Updated with combined definition")
    
    async def _generate_conflict_resolution(self, term: str, existing: str, new: str) -> str:
        """Generate suggestion for resolving definition conflicts"""
        prompt = f"""Two different definitions exist for the term "{term}":

Definition 1: {existing}
Definition 2: {new}

Suggest how to resolve this conflict. Should we:
1. Keep the first definition
2. Keep the second definition  
3. Combine both definitions
4. Create a more comprehensive definition

Provide your recommendation in one sentence:"""
        
        system_prompt = "You are an expert at resolving definition conflicts. Provide clear, practical recommendations."
        
        try:
            suggestion = await vllm_complete(prompt, system_prompt)
            return suggestion
        except:
            return "Consider combining both definitions to capture all aspects of the term."
    
    async def build_graphrag(self, terms_with_definitions: Dict[str, str], 
                           terms_with_contexts: Dict[str, List[str]]) -> bool:
        """Build the GraphRAG knowledge graph"""
        print("\nSTEP 6: BUILDING GRAPHRAG")
        print("-" * 30)
        
        try:
            # Create chunks
            chunks = self.graph_manager.create_graph_chunks(terms_with_definitions, terms_with_contexts)
            
            # Initialize GraphRAG
            rag = GraphRAG(
                working_dir=WORKING_DIR,
                embedding_func=local_embedding,
                best_model_func=vllm_complete,
                cheap_model_func=vllm_complete,
                best_model_max_token_size=MAX_TOKEN,
                cheap_model_max_token_size=MAX_TOKEN,
                best_model_max_async=5,
                cheap_model_max_async=5,
                convert_response_to_json_func=lambda x: json_repair.loads(x) if x else {},
                embedding_func_max_async=3,
                embedding_batch_num=4,
            )
            
            # Insert data
            rag.insert(chunks[:1000])  # Limit for memory
            print("✅ GraphRAG knowledge graph built successfully!")
            return True
            
        except Exception as e:
            print(f"❌ GraphRAG build failed: {e}")
            return False
    
    def save_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Save all comprehensive results to files"""
        print("\nSTEP 8: SAVING COMPREHENSIVE RESULTS")
        print("-" * 30)
        
        os.makedirs(WORKING_DIR, exist_ok=True)
        
        # Save chunk information
        chunks_path = os.path.join(WORKING_DIR, "book_chunks.json")
        chunk_data = {
            'total_chunks': len(results['chunks']),
            'chunks': [{
                'id': chunk['id'],
                'char_start': chunk.get('char_start', 0),
                'char_end': chunk.get('char_end', 0),
                'token_count': chunk.get('token_count', chunk.get('word_count', 0)),
                'preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            } for chunk in results['chunks']]
        }
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        # Save chunk processing results
        chunk_results_path = os.path.join(WORKING_DIR, "chunk_results.json")
        with open(chunk_results_path, 'w', encoding='utf-8') as f:
            json.dump(results['chunk_results'], f, indent=2, ensure_ascii=False)
        
        # Save terms list
        terms_path = os.path.join(WORKING_DIR, "extracted_terms.json")
        with open(terms_path, 'w', encoding='utf-8') as f:
            json.dump(results['terms'], f, indent=2, ensure_ascii=False)
        
        # Save terms with contexts
        contexts_path = os.path.join(WORKING_DIR, "terms_with_contexts.json")
        with open(contexts_path, 'w', encoding='utf-8') as f:
            json.dump(results['terms_with_contexts'], f, indent=2, ensure_ascii=False)
        
        # Save definitions
        definitions_path = os.path.join(WORKING_DIR, "generated_definitions.json")
        with open(definitions_path, 'w', encoding='utf-8') as f:
            json.dump(results['definitions'], f, indent=2, ensure_ascii=False)
        
        # Save graph entities
        entities_path = os.path.join(WORKING_DIR, "graph_entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph_manager.entity_definitions, f, indent=2, ensure_ascii=False)
        
        # Save processing statistics
        stats_path = os.path.join(WORKING_DIR, "processing_stats.json")
        comprehensive_stats = {
            'processing_stats': results['processing_stats'],
            'graph_stats': results['graph_result'],
            'coverage': {
                'chunks_with_terms': len([r for r in results['chunk_results'] if r['terms_found'] > 0]),
                'chunks_with_definitions': len([r for r in results['chunk_results'] if r['definitions_generated'] > 0]),
                'average_terms_per_chunk': sum(r['terms_found'] for r in results['chunk_results']) / len(results['chunk_results'])
            }
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Comprehensive results saved to {WORKING_DIR}")
        print(f"   📚 Chunks: {chunks_path}")
        print(f"   🔍 Chunk results: {chunk_results_path}")
        print(f"   📋 Terms: {terms_path}")
        print(f"   📚 Contexts: {contexts_path}")
        print(f"   💡 Definitions: {definitions_path}")
        print(f"   🕸️ Entities: {entities_path}")
        print(f"   📊 Statistics: {stats_path}")

    # Keep the old method for backward compatibility
    async def process_book(self, text: str, **kwargs) -> Dict[str, Any]:
        """Legacy method - now calls the comprehensive version"""
        return await self.process_book_comprehensive(text, use_global_terms=True, **kwargs)

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
        
        # Load definitions (main vocabulary)
        definitions_path = os.path.join(self.working_dir, "generated_definitions.json")
        if os.path.exists(definitions_path):
            with open(definitions_path, 'r', encoding='utf-8') as f:
                definitions = json.load(f)
            print(f"✅ Loaded {len(definitions)} definitions")
        else:
            definitions = {}
            print("⚠️ No definitions file found")
        
        # Load terms with contexts
        contexts_path = os.path.join(self.working_dir, "terms_with_contexts.json")
        if os.path.exists(contexts_path):
            with open(contexts_path, 'r', encoding='utf-8') as f:
                contexts = json.load(f)
            print(f"✅ Loaded contexts for {len(contexts)} terms")
        else:
            contexts = {}
            print("⚠️ No contexts file found")
        
        # Load chunk data
        chunks_path = os.path.join(self.working_dir, "book_chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunk_data = json.load(f)
            print(f"✅ Loaded {self.chunk_data.get('total_chunks', 0)} chunks info")
        
        # Load chunk results
        chunk_results_path = os.path.join(self.working_dir, "chunk_results.json")
        if os.path.exists(chunk_results_path):
            with open(chunk_results_path, 'r', encoding='utf-8') as f:
                self.chunk_results = json.load(f)
            print(f"✅ Loaded {len(self.chunk_results)} chunk results")
        
        # Load processing stats
        stats_path = os.path.join(self.working_dir, "processing_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.processing_stats = json.load(f)
            print(f"✅ Loaded processing statistics")
        
        # Combine into vocabulary structure
        self.vocabulary = {}
        for term, definition in definitions.items():
            self.vocabulary[term] = {
                'definition': definition,
                'contexts': contexts.get(term, []),
                'found_in_chunks': self._find_term_chunks(term),
                'has_definition': True
            }
        
        print(f"📊 Dictionary ready: {len(self.vocabulary)} terms")
    
    def _find_term_chunks(self, term: str) -> List[int]:
        """Find which chunks contain this term"""
        chunks_with_term = []
        for result in self.chunk_results:
            if term in result.get('terms', []):
                chunks_with_term.append(result['chunk_id'])
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
        
        # Fuzzy matching
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
                'word': word,
                'definition': vocab_entry['definition'],
                'found': True,
                'contexts': vocab_entry['contexts'],
                'found_in_chunks': vocab_entry['found_in_chunks'],
                'chunk_count': len(vocab_entry['found_in_chunks']),
                'context_count': len(vocab_entry['contexts']),
                'has_definition': vocab_entry['has_definition'],
                'query_time': time.time() - start_time,
                'suggestions': []
            }
        else:
            suggestions = self.suggest_words(word)
            result = {
                'word': word,
                'definition': None,
                'found': False,
                'contexts': [],
                'found_in_chunks': [],
                'chunk_count': 0,
                'context_count': 0,
                'has_definition': False,
                'query_time': time.time() - start_time,
                'suggestions': suggestions
            }
        
        return result
    
    async def query_graphrag(self, query: str, mode: str = "local") -> str:
        """Query using GraphRAG if available"""
        if not self.graph_rag:
            if not self.load_graphrag():
                return "GraphRAG not available"
        
        try:
            result = await self.graph_rag.aquery(
                query, 
                param=QueryParam(mode=mode, top_k=5)
            )
            return result
        except Exception as e:
            return f"GraphRAG query failed: {e}"
    
    def search_in_chunks(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for terms or phrases across chunks"""
        query_lower = query.lower().strip()
        results = []
        
        for result in self.chunk_results:
            chunk_id = result['chunk_id']
            
            # Check if query matches any terms in this chunk
            matching_terms = []
            for term in result.get('terms', []):
                if query_lower in term.lower() or term.lower() in query_lower:
                    matching_terms.append(term)
            
            # Check definitions for matches
            matching_definitions = []
            for term, definition in result.get('definitions', {}).items():
                if query_lower in definition.lower():
                    matching_definitions.append({
                        'term': term,
                        'definition': definition
                    })
            
            if matching_terms or matching_definitions:
                chunk_info = None
                if self.chunk_data and 'chunks' in self.chunk_data:
                    chunk_info = next((c for c in self.chunk_data['chunks'] if c['id'] == chunk_id), None)
                
                results.append({
                    'chunk_id': chunk_id,
                    'matching_terms': matching_terms,
                    'matching_definitions': matching_definitions,
                    'chunk_info': chunk_info,
                    'relevance_score': len(matching_terms) + len(matching_definitions)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_results]
    
    def get_term_distribution(self, term: str) -> Dict[str, Any]:
        """Get distribution of term across chunks"""
        word_lower = term.lower().strip()
        
        if word_lower not in self.vocabulary:
            return {'error': 'Term not found'}
        
        vocab_entry = self.vocabulary[word_lower]
        chunk_distribution = []
        
        for chunk_id in vocab_entry['found_in_chunks']:
            chunk_result = next((r for r in self.chunk_results if r['chunk_id'] == chunk_id), None)
            if chunk_result:
                chunk_info = None
                if self.chunk_data and 'chunks' in self.chunk_data:
                    chunk_info = next((c for c in self.chunk_data['chunks'] if c['id'] == chunk_id), None)
                
                chunk_distribution.append({
                    'chunk_id': chunk_id,
                    'chunk_info': chunk_info,
                    'has_definition': term in chunk_result.get('definitions', {}),
                    'definition': chunk_result.get('definitions', {}).get(term, '')
                })
        
        return {
            'term': term,
            'total_chunks': len(vocab_entry['found_in_chunks']),
            'distribution': chunk_distribution,
            'contexts': vocab_entry['contexts']
        }
    
    def format_definition(self, result: Dict[str, Any]) -> str:
        """Format definition result for display"""
        word = result['word']
        definition = result['definition']
        
        output = f"📖 **{word.upper()}**"
        if result['found']:
            output += f" (found in {result['chunk_count']} chunks, {result['context_count']} contexts)"
        
        output += "\n" + "="*(len(word) + 20) + "\n"
        
        if definition:
            output += f"💡 **Definition:** {definition}\n"
            
            if result['contexts']:
                output += f"\n📚 **Contexts:**\n"
                for i, context in enumerate(result['contexts'][:3], 1):
                    output += f"  {i}. {context[:200]}...\n"
            
            if result['found_in_chunks']:
                chunks_str = ", ".join(map(str, result['found_in_chunks'][:10]))
                if len(result['found_in_chunks']) > 10:
                    chunks_str += f" and {len(result['found_in_chunks']) - 10} more"
                output += f"\n🧩 **Found in chunks:** {chunks_str}\n"
        
        else:
            output += f"❌ No definition found for '{word}'\n"
            if result['suggestions']:
                output += f"\n💡 **Similar words:** {', '.join(result['suggestions'][:5])}\n"
        
        output += f"\n⏱️ Query time: {result['query_time']:.3f}s"
        return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'total_terms': len(self.vocabulary),
            'total_chunks': self.chunk_data.get('total_chunks', 0),
            'chunks_with_terms': len([r for r in self.chunk_results if r.get('terms_found', 0) > 0]),
            'chunks_with_definitions': len([r for r in self.chunk_results if r.get('definitions_generated', 0) > 0]),
            'average_terms_per_chunk': 0,
            'terms_with_multiple_chunks': 0,
            'processing_strategy': 'unknown'
        }
        
        if self.chunk_results:
            total_terms = sum(r.get('terms_found', 0) for r in self.chunk_results)
            stats['average_terms_per_chunk'] = total_terms / len(self.chunk_results)
        
        # Count terms that appear in multiple chunks
        for term_info in self.vocabulary.values():
            if len(term_info.get('found_in_chunks', [])) > 1:
                stats['terms_with_multiple_chunks'] += 1
        
        if self.processing_stats:
            stats['processing_strategy'] = self.processing_stats.get('processing_stats', {}).get('strategy', 'unknown')
        
        return stats
    
    async def interactive_query(self):
        """Enhanced interactive query interface"""
        print(f"\n📚 Enhanced Book Dictionary Interface")
        print("="*60)
        
        stats = self.get_stats()
        print(f"📊 Dictionary: {stats['total_terms']} terms from {stats['total_chunks']} chunks")
        print(f"📈 Coverage: {stats['chunks_with_terms']}/{stats['total_chunks']} chunks have terms")
        print(f"🔄 Strategy: {stats['processing_strategy']}")
        print(f"🔗 Multi-chunk terms: {stats['terms_with_multiple_chunks']}")
        
        # Try to load GraphRAG
        has_graphrag = self.load_graphrag()
        print(f"🕸️ GraphRAG: {'Available' if has_graphrag else 'Not available'}")
        
        print("\nCommands:")
        print("  <word>           - Look up word definition")
        print("  search <query>   - Search across chunks") 
        print("  dist <word>      - Show word distribution across chunks")
        print("  graphrag <query> - Query using GraphRAG (if available)")
        print("  vocab            - Show vocabulary sample")
        print("  stats            - Show detailed statistics")
        print("  chunks           - Show chunk information")
        print("  quit/exit        - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🔍 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                query = parts[1] if len(parts) > 1 else ""
                
                if command == 'search':
                    if not query:
                        print("❌ Please provide a search query")
                        continue
                    
                    search_results = self.search_in_chunks(query)
                    print(f"\n🔍 **SEARCH RESULTS for '{query}'** ({len(search_results)} matches)")
                    
                    for result in search_results:
                        chunk_id = result['chunk_id']
                        print(f"\n📄 **Chunk {chunk_id}** (relevance: {result['relevance_score']})")
                        
                        if result['matching_terms']:
                            print(f"  🏷️ Terms: {', '.join(result['matching_terms'])}")
                        
                        if result['matching_definitions']:
                            for def_match in result['matching_definitions'][:2]:
                                print(f"  💡 {def_match['term']}: {def_match['definition'][:100]}...")
                
                elif command == 'dist':
                    if not query:
                        print("❌ Please provide a term")
                        continue
                    
                    distribution = self.get_term_distribution(query)
                    if 'error' in distribution:
                        print(f"❌ {distribution['error']}")
                    else:
                        print(f"\n📊 **DISTRIBUTION of '{distribution['term']}'**")
                        print(f"Found in {distribution['total_chunks']} chunks:")
                        
                        for chunk_dist in distribution['distribution'][:10]:
                            chunk_id = chunk_dist['chunk_id']
                            has_def = "✓" if chunk_dist['has_definition'] else "○"
                            print(f"  Chunk {chunk_id}: {has_def}")
                
                elif command == 'graphrag':
                    if not has_graphrag:
                        print("❌ GraphRAG not available")
                        continue
                    
                    if not query:
                        print("❌ Please provide a GraphRAG query")
                        continue
                    
                    print(f"🕸️ Querying GraphRAG: '{query}'...")
                    result = await self.query_graphrag(query)
                    print(f"\n💡 **GraphRAG Response:**\n{result}")
                
                elif command == 'vocab':
                    vocab_sample = sorted(list(self.vocabulary.keys()))[:20]
                    print(f"\n📝 **VOCABULARY SAMPLE** (first 20 of {len(self.vocabulary)}):")
                    for i, word in enumerate(vocab_sample, 1):
                        info = self.vocabulary[word]
                        chunks = len(info['found_in_chunks'])
                        contexts = len(info['contexts'])
                        print(f"  {i:2d}. {word} (chunks: {chunks}, contexts: {contexts})")
                
                elif command == 'stats':
                    detailed_stats = self.get_stats()
                    print(f"\n📊 **DETAILED STATISTICS**")
                    for key, value in detailed_stats.items():
                        formatted_key = key.replace('_', ' ').title()
                        if isinstance(value, float):
                            print(f"  {formatted_key}: {value:.2f}")
                        else:
                            print(f"  {formatted_key}: {value}")
                
                elif command == 'chunks':
                    if self.chunk_data and 'chunks' in self.chunk_data:
                        print(f"\n🧩 **CHUNK INFORMATION** ({self.chunk_data['total_chunks']} total)")
                        for chunk in self.chunk_data['chunks'][:10]:
                            chunk_result = next((r for r in self.chunk_results if r['chunk_id'] == chunk['id']), {})
                            terms_count = chunk_result.get('terms_found', 0)
                            defs_count = chunk_result.get('definitions_generated', 0)
                            print(f"  Chunk {chunk['id']}: {terms_count} terms, {defs_count} definitions")
                            print(f"    Preview: {chunk.get('preview', 'No preview')}")
                        
                        if len(self.chunk_data['chunks']) > 10:
                            print(f"  ... and {len(self.chunk_data['chunks']) - 10} more chunks")
                    else:
                        print("❌ No chunk information available")
                
                else:
                    # Default: word lookup
                    word_to_lookup = user_input
                    result = self.define_word(word_to_lookup)
                    print(f"\n{self.format_definition(result)}")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py build [language] [strategy] - Build dictionary")
        print("    Strategies:")
        print("      global     - Extract terms from entire book, then check each chunk (default)")
        print("      chunk      - Extract terms from each chunk independently")
        print("  python script.py extract [language] - Just extract terms")
        print("  python script.py chunks [size] [overlap] - Just show chunk breakdown")
        print("  python script.py query <word> - Look up specific word")
        print("  python script.py interactive - Interactive query mode")
        print("  python script.py search <query> - Search across chunks")
        print("  python script.py graphrag <query> - Query using GraphRAG")
        print("  python script.py analyze - Show comprehensive analysis")
        return
    
    command = sys.argv[1].lower()
    language = sys.argv[2] if len(sys.argv) > 2 else "english"
    
    # Load book
    book_files = ["book2.txt"]
    book_file = None
    for filename in book_files:
        if os.path.exists(filename):
            book_file = filename
            break
    
    if book_file is None and command not in ['query', 'interactive', 'search', 'graphrag', 'analyze']:
        print("❌ No book file found. Please ensure book2.txt or book.txt exists.")
        return
    
    if book_file:
        try:
            with open(book_file, encoding="utf-8") as f:
                book_text = f.read()
            print(f"✅ Loaded book: {book_file} ({len(book_text)} characters)")
        except Exception as e:
            print(f"❌ Error reading book: {e}")
            return
    
    if command == "chunks":
        # Show chunk breakdown
        chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        
        chunker = BookChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.create_chunks(book_text)
        
        print(f"\n📚 BOOK CHUNK BREAKDOWN")
        print("="*50)
        print(f"Book length: {len(book_text)} characters")
        print(f"Chunk size: {chunk_size} tokens/words")
        print(f"Overlap: {overlap} tokens/words")
        print(f"Total chunks: {len(chunks)}")
        
        print(f"\n📖 SAMPLE CHUNKS:")
        for i, chunk in enumerate(chunks[:5]):
            size_info = f"{chunk.get('token_count', chunk.get('word_count', 0))} tokens/words"
            print(f"Chunk {i}: {size_info}")
            print(f"  Preview: {chunk['text'][:100]}...")
            print()
        
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks")
    
    elif command == "extract":
        # Just extract terms (legacy mode)
        builder = BookDictionaryBuilder(language)
        terms = builder.term_extractor.extract_terms(book_text)
        print(f"\n📋 EXTRACTED TERMS ({len(terms)}):")
        for i, term in enumerate(terms[:20], 1):
            print(f"  {i:2d}. {term}")
        if len(terms) > 20:
            print(f"  ... and {len(terms) - 20} more")
    
    elif command == "build":
        # Full comprehensive build process
        strategy = sys.argv[3] if len(sys.argv) > 3 else "global"
        use_global_terms = strategy.lower() == "global"
        
        print(f"🚀 Building with strategy: {'Global terms first' if use_global_terms else 'Chunk-by-chunk extraction'}")
        
        builder = BookDictionaryBuilder(language)
        results = await builder.process_book_comprehensive(
            book_text, 
            chunk_size=1000,
            overlap=200,
            use_global_terms=use_global_terms,
            min_frequency=2 if use_global_terms else 1,
            max_terms=300 if use_global_terms else 50
        )
        
        # Build GraphRAG
        await builder.build_graphrag(results['definitions'], results['terms_with_contexts'])
        
        # Save comprehensive results
        builder.save_comprehensive_results(results)
        
        # Show comprehensive summary
        print(f"\n" + "="*60)
        print("✅ COMPREHENSIVE BOOK DICTIONARY BUILD COMPLETE!")
        print("="*60)
        
        stats = results['processing_stats']
        print(f"📚 Book processed: {book_file}")
        print(f"🧩 Total chunks: {stats['total_chunks']}")
        print(f"📊 Strategy: {stats['strategy']}")
        print(f"🏷️ Terms found: {stats['total_terms']}")
        print(f"💡 Definitions generated: {stats['total_definitions']}")
        print(f"🕸️ Graph entities: {results['graph_result']['total_entities']}")
        print(f"🌍 Language: {language}")
        
        # Show chunk coverage
        chunk_coverage = len([r for r in results['chunk_results'] if r['terms_found'] > 0])
        coverage_percent = (chunk_coverage / stats['total_chunks']) * 100
        print(f"📖 Chunk coverage: {chunk_coverage}/{stats['total_chunks']} ({coverage_percent:.1f}%)")
        
        # Show sample definitions from different chunks
        print(f"\n📖 SAMPLE DEFINITIONS FROM DIFFERENT CHUNKS:")
        sample_count = 0
        for chunk_result in results['chunk_results'][:10]:  # Check first 10 chunks
            if chunk_result['definitions'] and sample_count < 5:
                chunk_id = chunk_result['chunk_id']
                sample_term = list(chunk_result['definitions'].keys())[0]
                sample_def = chunk_result['definitions'][sample_term]
                print(f"{sample_count + 1}. **{sample_term.upper()}** (from chunk {chunk_id}): {sample_def[:100]}...")
                sample_count += 1
        
        print(f"\n🚀 Ready to use! Test some queries:")
        sample_terms = list(results['definitions'].keys())[:3]
        for term in sample_terms:
            print(f"   python {sys.argv[0]} query {term}")
    
    elif command == "query":
        # Query mode
        if not os.path.exists(WORKING_DIR):
            print("❌ No dictionary cache found. Build first with: python script.py build")
            return
        
        dictionary = BookDictionaryQuery()
        
        if len(sys.argv) > 2:
            # Direct word lookup
            word = " ".join(sys.argv[2:])
            result = dictionary.define_word(word)
            print(dictionary.format_definition(result))
        else:
            # Interactive mode
            await dictionary.interactive_query()
    
    elif command == "interactive":
        # Interactive query mode
        if not os.path.exists(WORKING_DIR):
            print("❌ No dictionary cache found. Build first with: python script.py build")
            return
        
        dictionary = BookDictionaryQuery()
        await dictionary.interactive_query()
    
    elif command == "search":
        # Search mode
        if not os.path.exists(WORKING_DIR):
            print("❌ No dictionary cache found. Build first with: python script.py build")
            return
        
        if len(sys.argv) < 3:
            print("❌ Please provide a search query")
            return
        
        query = " ".join(sys.argv[2:])
        dictionary = BookDictionaryQuery()
        results = dictionary.search_in_chunks(query)
        
        print(f"\n🔍 **SEARCH RESULTS for '{query}'** ({len(results)} matches)")
        for result in results:
            chunk_id = result['chunk_id']
            print(f"\n📄 **Chunk {chunk_id}** (relevance: {result['relevance_score']})")
            
            if result['matching_terms']:
                print(f"  🏷️ Terms: {', '.join(result['matching_terms'])}")
            
            if result['matching_definitions']:
                for def_match in result['matching_definitions'][:2]:
                    print(f"  💡 {def_match['term']}: {def_match['definition'][:100]}...")
    
    elif command == "graphrag":
        # GraphRAG query mode
        if not os.path.exists(WORKING_DIR):
            print("❌ No dictionary cache found. Build first with: python script.py build")
            return
        
        if len(sys.argv) < 3:
            print("❌ Please provide a GraphRAG query")
            return
        
        query = " ".join(sys.argv[2:])
        dictionary = BookDictionaryQuery()
        
        print(f"🕸️ Querying GraphRAG: '{query}'...")
        result = await dictionary.query_graphrag(query)
        print(f"\n💡 **GraphRAG Response:**\n{result}")
    
    elif command == "analyze":
        # Analysis mode - show comprehensive stats
        if not os.path.exists(WORKING_DIR):
            print("❌ No dictionary cache found. Build first with: python script.py build")
            return
        
        dictionary = BookDictionaryQuery()
        stats = dictionary.get_stats()
        
        print(f"\n📊 **COMPREHENSIVE ANALYSIS**")
        print("="*50)
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"{formatted_key}: {value:.2f}")
            else:
                print(f"{formatted_key}: {value}")
        
        # Show top terms by chunk distribution
        print(f"\n🔝 **TOP TERMS BY CHUNK DISTRIBUTION:**")
        term_chunks = [(term, len(info['found_in_chunks'])) 
                      for term, info in dictionary.vocabulary.items()]
        term_chunks.sort(key=lambda x: x[1], reverse=True)
        
        for i, (term, chunk_count) in enumerate(term_chunks[:10], 1):
            print(f"  {i:2d}. {term} (appears in {chunk_count} chunks)")

if __name__ == "__main__":
    asyncio.run(main())
