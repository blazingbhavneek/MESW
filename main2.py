# Modular Book Dictionary Builder
# Separated term extraction, definition generation, and graph building

import sys
import logging
import numpy as np
import asyncio
import faiss
import pickle
import os
import json_repair
import tiktoken
import requests
import json
from openai import AsyncOpenAI
import re
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import Counter, defaultdict
import time
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
        system_prompt = "You are a helpful assistant that provides clear, concise responses."
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
    try:
        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKEN,
            temperature=0.3,
            extra_body={
                "repetition_penalty": 1.1,
                "top_p": 0.9,
            }
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
# TERM CONTEXT MANAGER
# ============================================================================

class TermContextManager:
    """Manage term contexts and metadata"""
    
    def __init__(self, text: str):
        self.text = text
        self.sentences = self._split_sentences(text)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_contexts_for_terms(self, terms: List[str], max_contexts_per_term: int = 3) -> Dict[str, List[str]]:
        """Get contexts for each term"""
        term_contexts = {}
        
        print(f"📚 Finding contexts for {len(terms)} terms...")
        
        for term in terms:
            contexts = []
            term_lower = term.lower()
            
            for sentence in self.sentences:
                if term_lower in sentence.lower():
                    # Truncate long contexts
                    context = sentence[:300] + "..." if len(sentence) > 300 else sentence
                    contexts.append(context)
                    
                    if len(contexts) >= max_contexts_per_term:
                        break
            
            term_contexts[term] = contexts
        
        # Filter out terms with no contexts
        term_contexts = {term: contexts for term, contexts in term_contexts.items() if contexts}
        
        print(f"✅ Found contexts for {len(term_contexts)} terms")
        return term_contexts

# ============================================================================
# DEFINITION GENERATOR
# ============================================================================

class DefinitionGenerator:
    """Generate definitions for terms using LLM"""
    
    def __init__(self):
        self.generated_definitions = {}
    
    async def generate_definition_batch(self, terms_with_contexts: Dict[str, List[str]], 
                                      batch_size: int = MAX_CONCURRENT_DEFINITIONS) -> Dict[str, str]:
        """Generate definitions for a batch of terms"""
        print(f"📝 Generating definitions for {len(terms_with_contexts)} terms...")
        
        terms_list = list(terms_with_contexts.items())
        
        for i in range(0, len(terms_list), batch_size):
            batch = terms_list[i:i+batch_size]
            print(f"📋 Processing batch {i//batch_size + 1}/{(len(terms_list)-1)//batch_size + 1}")
            
            tasks = [self._generate_single_definition(term, contexts) 
                    for term, contexts in batch]
            definitions = await asyncio.gather(*tasks)
            
            for (term, _), definition in zip(batch, definitions):
                if definition:
                    self.generated_definitions[term] = definition
                    print(f"✅ {term}: {definition[:60]}...")
        
        return self.generated_definitions
    
    async def _generate_single_definition(self, term: str, contexts: List[str]) -> str:
        """Generate definition for a single term"""
        if not contexts:
            return ""
        
        contexts_text = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts[:3])])
        
        prompt = f"""Based on the following contexts from a book, provide a clear, concise definition of the term "{term}".

Term: {term}

Book contexts:
{contexts_text}

Provide ONLY the definition (1-2 sentences), no extra text:"""

        system_prompt = """You are a helpful assistant that creates clear, concise definitions for terms based on book contexts. 
Provide ONLY the definition text, no thinking steps or additional commentary. Keep it to 1-2 sentences."""
        
        try:
            definition = await vllm_complete(prompt, system_prompt)
            
            # Clean the definition
            definition = re.sub(r'<think>.*?</think>', '', definition, flags=re.DOTALL | re.IGNORECASE)
            definition = re.sub(r'^\s*(?:Definition:|Term:|Thinking)[:\s]*', '', definition, flags=re.IGNORECASE)
            definition = definition.strip()
            
            if definition and not definition.endswith(('.', '!', '?')):
                definition += '.'
            
            return definition
            
        except Exception as e:
            print(f"❌ Failed to generate definition for '{term}': {e}")
            return f"Technical term from the book that appears in {len(contexts)} contexts."

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
        self.definition_generator = DefinitionGenerator()
        self.graph_manager = GraphManager()
    
    def _get_extractor(self, language: str) -> BaseTermExtractor:
        """Get appropriate term extractor based on language"""
        if language.lower() == "english":
            return EnglishTermExtractor()
        else:
            raise ValueError(f"Language '{language}' not supported yet")
    
    async def process_book(self, text: str, **kwargs) -> Dict[str, Any]:
        """Main processing pipeline"""
        print(f"📚 Processing book with {self.language} term extractor...")
        print("="*60)
        
        # Step 1: Extract terms
        print("\nSTEP 1: EXTRACTING TERMS")
        print("-" * 30)
        terms = self.term_extractor.extract_terms(text, **kwargs)
        
        # Step 2: Get contexts for terms
        print("\nSTEP 2: FINDING CONTEXTS")
        print("-" * 30)
        self.context_manager = TermContextManager(text)
        terms_with_contexts = self.context_manager.get_contexts_for_terms(terms)
        
        # Step 3: Generate definitions
        print("\nSTEP 3: GENERATING DEFINITIONS")
        print("-" * 30)
        await self.definition_generator.generate_definition_batch(terms_with_contexts)
        
        # Step 4: Process graph entities
        print("\nSTEP 4: PROCESSING GRAPH")
        print("-" * 30)
        graph_result = self.graph_manager.add_or_update_entities(
            self.definition_generator.generated_definitions,
            terms_with_contexts
        )
        
        # Step 5: Handle conflicts/questions
        print("\nSTEP 5: REVIEWING CONFLICTS")
        print("-" * 30)
        await self._handle_conflicts(graph_result['similar_entities'])
        
        return {
            'terms': terms,
            'terms_with_contexts': terms_with_contexts,
            'definitions': self.definition_generator.generated_definitions,
            'graph_result': graph_result
        }
    
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
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save all results to files"""
        print("\nSTEP 7: SAVING RESULTS")
        print("-" * 30)
        
        os.makedirs(WORKING_DIR, exist_ok=True)
        
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
        
        print(f"✅ Results saved to {WORKING_DIR}")
        print(f"   📋 Terms: {terms_path}")
        print(f"   📚 Contexts: {contexts_path}")
        print(f"   💡 Definitions: {definitions_path}")
        print(f"   🕸️ Entities: {entities_path}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script.py build [language] - Build dictionary (default: english)")
        print("  python script.py extract [language] - Just extract terms")
        return
    
    command = sys.argv[1].lower()
    language = sys.argv[2] if len(sys.argv) > 2 else "english"
    
    # Load book
    book_files = ["book2.txt", "book.txt"]
    book_file = None
    for filename in book_files:
        if os.path.exists(filename):
            book_file = filename
            break
    
    if book_file is None:
        print("❌ No book file found. Please ensure book2.txt or book.txt exists.")
        return
    
    try:
        with open(book_file, encoding="utf-8") as f:
            book_text = f.read()
        print(f"✅ Loaded book: {book_file} ({len(book_text)} characters)")
    except Exception as e:
        print(f"❌ Error reading book: {e}")
        return
    
    if command == "extract":
        # Just extract terms
        builder = BookDictionaryBuilder(language)
        terms = builder.term_extractor.extract_terms(book_text)
        print(f"\n📋 EXTRACTED TERMS ({len(terms)}):")
        for i, term in enumerate(terms[:20], 1):
            print(f"  {i:2d}. {term}")
        if len(terms) > 20:
            print(f"  ... and {len(terms) - 20} more")
    
    elif command == "build":
        # Full build process
        builder = BookDictionaryBuilder(language)
        results = await builder.process_book(book_text, min_frequency=3, max_terms=200)
        
        # Build GraphRAG
        await builder.build_graphrag(results['definitions'], results['terms_with_contexts'])
        
        # Save results
        builder.save_results(results)
        
        # Show summary
        print(f"\n" + "="*60)
        print("✅ BOOK DICTIONARY BUILD COMPLETE!")
        print("="*60)
        print(f"📊 Terms extracted: {len(results['terms'])}")
        print(f"💡 Definitions generated: {len(results['definitions'])}")
        print(f"🕸️ Graph entities: {results['graph_result']['total_entities']}")
        print(f"🌍 Language: {language}")
        
        # Show sample
        print(f"\n📖 SAMPLE DEFINITIONS:")
        for i, (term, definition) in enumerate(list(results['definitions'].items())[:5], 1):
            print(f"{i}. **{term.upper()}**: {definition[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
