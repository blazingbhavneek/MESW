# Complete Book Dictionary Builder - Everything in One File
# Builds vocabulary, generates definitions, creates GraphRAG, and provides query interface

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
from typing import List, Dict, Tuple, Set, Any
from collections import Counter, defaultdict
import time
from difflib import get_close_matches

MAX_CONCURRENT_DEFINITIONS = 15  # Set to 1 if using Ollama or non-concurrent backend; for VLLM, 15-20 is fine

# Fix scipy.linalg.triu issue by patching before nano_graphrag import
def patch_scipy_triu():
    """Patch scipy.linalg to use numpy.triu instead of the removed scipy version"""
    try:
        import scipy.linalg
        if not hasattr(scipy.linalg, 'triu'):
            print("🔧 Patching scipy.linalg.triu with numpy.triu...")
            scipy.linalg.triu = np.triu
            print("✅ Successfully patched scipy.linalg.triu")
    except ImportError:
        pass

# Apply the patch before importing nano_graphrag
patch_scipy_triu()

print("🔧 Checking dependencies and versions...")

# Now safely import nano-graphrag
try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag._utils import wrap_embedding_func_with_attrs
    from nano_graphrag.base import BaseVectorStorage
    print("✅ nano-graphrag imported successfully")
except ImportError as e:
    print(f"❌ Failed to import nano-graphrag: {e}")
    print("🔧 Install with: pip install nano-graphrag")
    sys.exit(1)

# Import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers available")
except ImportError:
    print("❌ sentence-transformers not available")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)

# Initialize NLTK availability flag
USE_NLTK = True
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Downloading NLTK data...")
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                print("⚠️ NLTK download failed, using simple tokenization fallback")
                USE_NLTK = False
except ImportError:
    print("⚠️ NLTK not available, using simple tokenization")
    USE_NLTK = False

if USE_NLTK:
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
    except ImportError:
        USE_NLTK = False

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano_graphrag").setLevel(logging.INFO)

WORKING_DIR = "./book_dictionary_cache"

# vLLM Server Configuration
VLLM_HOST = "http://localhost:8000"
VLLM_MODEL = "Qwen3-0.6B-Q8_0.gguf"
MAX_TOKEN = 1000
CONCURRENCY = 10

# Try to load embedding model
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

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

# vLLM OpenAI-compatible client
vllm_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=VLLM_HOST + "/v1",
)

async def book_dictionary_vllm_complete(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """LLM completion for definitions and GraphRAG"""
    if history_messages is None:
        history_messages = []
    
    if system_prompt is None:
        system_prompt = """You are creating a dictionary from book content. Provide clear, concise definitions based on context. Respond only with the definition text, no thinking steps, explanations, or additional formatting."""
    
    messages = []
    messages.append({'role': 'system', 'content': system_prompt})
    messages += history_messages
    messages.append({'role': 'user', 'content': prompt})
    
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

def convert_to_json(response: str) -> dict:
    try:
        return json_repair.loads(response)
    except:
        return {}

class BookVocabularyExtractor:
    """Extract vocabulary and important terms from book content"""
    
    def __init__(self):
        if USE_NLTK:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self._get_basic_stopwords()
        else:
            self.stop_words = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self) -> Set[str]:
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 
            'them', 'my', 'your', 'his', 'our', 'their', 'this', 'that', 'these', 'those'
        }
    
    def _simple_sent_tokenize(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _simple_word_tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b", text)
        return words

    def extract_vocabulary(self, text: str, min_frequency: int = 2) -> Dict[str, Dict]:
        print(f"🔍 Extracting technical terms from book...")
        
        if USE_NLTK:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = self._simple_sent_tokenize(text)
        else:
            sentences = self._simple_sent_tokenize(text)
        
        all_words = []
        word_contexts = {}
        
        for sentence in sentences:
            if USE_NLTK:
                try:
                    words = word_tokenize(sentence.lower())
                except:
                    words = self._simple_word_tokenize(sentence.lower())
            else:
                words = self._simple_word_tokenize(sentence.lower())
            
            significant_words = [
                w for w in words 
                if w.isalpha() and len(w) > 3 and w not in self.stop_words
            ]
            all_words.extend(significant_words)
            
            short_sentence = sentence[:300] + "..." if len(sentence) > 300 else sentence
            for word in significant_words:
                if word not in word_contexts:
                    word_contexts[word] = []
                if len(word_contexts[word]) < 5:
                    word_contexts[word].append(short_sentence)
        
        word_freq = Counter(all_words)
        technical_terms = self._extract_technical_terms(text)
        
        vocabulary = {}
        
        # Add technical terms only
        for term, contexts in list(technical_terms.items())[:100]:
            vocabulary[term] = {
                'frequency': len(contexts),
                'type': 'technical_term',
                'contexts': contexts[:3],
                'definition_source': 'contextual'
            }
        
        print(f"✅ Extracted {len(vocabulary)} technical terms")
        return vocabulary


    def _extract_proper_nouns(self, text: str) -> Dict[str, int]:
        proper_nouns = []
        
        if USE_NLTK:
            try:
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    for i, word in enumerate(words):
                        if (i > 0 and word[0].isupper() and word.isalpha() and 
                            len(word) > 2 and word.lower() not in self.stop_words):
                            proper_nouns.append(word)
            except:
                proper_nouns = self._simple_extract_proper_nouns(text)
        else:
            proper_nouns = self._simple_extract_proper_nouns(text)
        
        return Counter(proper_nouns)
    
    def _simple_extract_proper_nouns(self, text: str) -> List[str]:
        proper_nouns = []
        sentences = self._simple_sent_tokenize(text)
        
        for sentence in sentences:
            words = self._simple_word_tokenize(sentence)
            for i, word in enumerate(words):
                if (i > 0 and word[0].isupper() and word.isalpha() and 
                    len(word) > 2 and word.lower() not in self.stop_words):
                    proper_nouns.append(word)
        
        return proper_nouns
    
    def _extract_technical_terms(self, text: str) -> Dict[str, List[str]]:
        technical_terms = {}
        
        compound_patterns = [
            r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b',
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        ]
        
        for pattern in compound_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                term = match.group().lower()
                if term not in technical_terms:
                    technical_terms[term] = []
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                technical_terms[term].append(context)
        
        return technical_terms

class BookDefinitionGenerator:
    """Generate definitions from vocabulary contexts"""
    
    def __init__(self, vocabulary: Dict[str, Dict]):
        self.vocabulary = vocabulary
        self.generated_definitions = {}

    # Update generate_definition_from_contexts for better cleaning and higher tokens
    async def generate_definition_from_contexts(self, word: str, contexts: List[str], word_info: Dict) -> str:
        word_type = word_info.get('type', 'word')
        frequency = word_info.get('frequency', 0)
        
        contexts_text = "\n".join([f"Context {i+1}: {ctx[:250]}..." for i, ctx in enumerate(contexts[:3])])
        
        prompt = f"""Based on the following contexts from a book, provide a clear, definition of the technical term "{word}".

    Term: {word}

    Book contexts:
    {contexts_text}

    Definition:"""

        try:
            response = await vllm_client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that creates clear, concise definitions for technical terms from book contexts. Output ONLY the definition text, starting directly after 'Definition:'. Do not include any thinking steps, <think> tags, explanations, notes, or extra text of any kind. The definition should be 1-2 sentences only."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1,
                extra_body={
                    "repetition_penalty": 1.1,
                    "top_p": 0.9,
                }
            )
            
            definition = response.choices[0].message.content.strip()
            if definition.startswith("Definition:"):
                definition = definition[11:].strip()
            
            # Remove <think> blocks and any artifacts
            definition = re.sub(r'<think>.*?</think>', '', definition, flags=re.DOTALL | re.IGNORECASE)
            definition = re.sub(r'^\s*(?:Thinking|Step|Note|Explanation|Reasoning)[:\s]*', '', definition, flags=re.IGNORECASE).strip()
            # Ensure it ends with a period if not empty
            if definition and not definition.endswith(('.', '!', '?')):
                definition += '.'
            
            return definition
            
        except Exception as e:
            print(f"❌ Failed to generate definition for '{word}': {e}")
            return self.create_fallback_definition(word, contexts, word_info)


    def create_fallback_definition(self, word: str, contexts: List[str], word_info: Dict) -> str:
        word_type = word_info.get('type', 'word')
        frequency = word_info.get('frequency', 0)
        
        if word_type == 'proper_noun':
            return f"In this book, {word} appears {frequency} times as a proper noun (likely a character, place, or organization). Context: {contexts[0][:150]}..."
        elif word_type == 'technical_term':
            return f"In this book, {word} is a technical term that appears {frequency} times. Based on context: {contexts[0][:150]}..."
        else:
            for context in contexts:
                patterns = [
                    rf'{word.lower()}.*?(?:is|are|means|refers to)\s+([^.!?]+)',
                    rf'(?:is|are|means|refers to).*?{word.lower()}.*?([^.!?]+)',
                    rf'{word.lower()}.*?defined as\s+([^.!?]+)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, context.lower())
                    if match:
                        definition_part = match.group(1).strip()
                        if len(definition_part) > 10:
                            return f"In this book, {word} {definition_part}."
            
            return f"In this book, {word} appears {frequency} times. Main context: {contexts[0][:200]}..."
    
    async def generate_all_definitions(self, batch_size: int = MAX_CONCURRENT_DEFINITIONS):
        print(f"📝 Generating definitions for vocabulary terms...")
        
        words_to_process = []
        for word, info in self.vocabulary.items():
            contexts = info.get('contexts', [])
            if contexts:
                words_to_process.append((word, info, contexts))
        
        print(f"🔄 Processing {len(words_to_process)} words with contexts...")
        
        for i in range(0, len(words_to_process), batch_size):
            batch = words_to_process[i:i+batch_size]
            print(f"📋 Processing batch {i//batch_size + 1}/{(len(words_to_process)-1)//batch_size + 1}")
            
            tasks = [self.generate_definition_from_contexts(word, contexts, info) for word, info, contexts in batch]
            definitions = await asyncio.gather(*tasks)
            
            for (word, _, _), definition in zip(batch, definitions):
                self.generated_definitions[word] = definition
                print(f"✅ {word}: {definition[:80]}...")

def create_vocabulary_chunks(vocabulary: Dict[str, Dict], generated_definitions: Dict[str, str]) -> List[str]:
    """Create chunks with definitions for GraphRAG"""
    chunks = []
    
    for word, info in vocabulary.items():
        word_type = info.get('type', 'unknown')
        frequency = info.get('frequency', 0)
        
        chunk_lines = []
        chunk_lines.append(f"**{word.title()}** ({word_type}, frequency: {frequency})")
        
        # Add generated definition if available
        if word in generated_definitions:
            chunk_lines.append(f"Definition: {generated_definitions[word]}")
        
        # Add contexts
        contexts = info.get('contexts', [])
        if contexts:
            chunk_lines.append("Contexts:")
            for i, context in enumerate(contexts[:2], 1):
                clean_context = context.strip()[:200]
                chunk_lines.append(f"{i}. {clean_context}")
        
        chunk_text = '\n'.join(chunk_lines)
        chunks.append(chunk_text)
    
    print(f"📚 Created {len(chunks)} vocabulary chunks with definitions")
    return chunks

async def check_vllm_server():
    """Check if vLLM server is running"""
    try:
        response = await vllm_client.models.list()
        print(f"✅ vLLM server is running with models: {[model.id for model in response.data]}")
        return True
    except Exception as e:
        print(f"❌ vLLM server not accessible: {e}")
        return False

class BookDictionaryQuery:
    """Query interface for the built dictionary"""
    
    def __init__(self):
        self.vocabulary = {}
        self.load_vocabulary()
    
    def load_vocabulary(self):
        vocab_path = os.path.join(WORKING_DIR, "extracted_vocabulary.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
            print(f"✅ Loaded {len(self.vocabulary)} vocabulary terms")
        else:
            print("❌ No vocabulary file found")
    
    def suggest_words(self, query: str, n: int = 5) -> List[str]:
        query = query.lower().strip()
        vocab_words = list(self.vocabulary.keys())
        
        if query in vocab_words:
            return [query]
        
        matches = get_close_matches(query, vocab_words, n=n, cutoff=0.6)
        prefix_matches = [w for w in vocab_words if w.startswith(query)][:n]
        
        combined = matches + [w for w in prefix_matches if w not in matches]
        return combined[:n]
    
    def define_word(self, word: str) -> Dict[str, Any]:
        word_lower = word.lower().strip()
        start_time = time.time()
        
        if word_lower in self.vocabulary:
            vocab_entry = self.vocabulary[word_lower]
            
            # Check for generated definition
            if 'generated_definition' in vocab_entry:
                definition = vocab_entry['generated_definition']
            else:
                # Fallback to contexts
                contexts = vocab_entry.get('contexts', [])
                if contexts:
                    definition = f"Context from book: {contexts[0][:300]}..."
                else:
                    definition = "Found in vocabulary but no definition available"
            
            result = {
                'word': word,
                'definition': definition,
                'found': True,
                'type': vocab_entry.get('type', 'unknown'),
                'frequency': vocab_entry.get('frequency', 0),
                'contexts': vocab_entry.get('contexts', []),
                'has_generated_definition': 'generated_definition' in vocab_entry,
                'query_time': time.time() - start_time,
                'suggestions': []
            }
        else:
            suggestions = self.suggest_words(word)
            result = {
                'word': word,
                'definition': None,
                'found': False,
                'type': 'unknown',
                'frequency': 0,
                'contexts': [],
                'has_generated_definition': False,
                'query_time': time.time() - start_time,
                'suggestions': suggestions
            }
        
        return result
    
    def format_definition(self, result: Dict[str, Any]) -> str:
        word = result['word']
        definition = result['definition']
        
        output = f"📖 **{word.upper()}**"
        if result['found']:
            output += f" ({result['type']}, appears {result['frequency']} times)"
        
        output += "\n" + "-" * (len(word) + 20) + "\n"
        
        if definition:
            output += f"{definition}\n"
        else:
            output += f"❌ No definition found for '{word}'\n"
            if result['suggestions']:
                output += f"\n💡 Similar words: {', '.join(result['suggestions'][:5])}\n"
        
        output += f"\n⏱️ Query time: {result['query_time']:.3f}s"
        
        return output
    
    def interactive_query(self):
        print(f"\n📚 Book Dictionary Interface")
        print("="*50)
        print(f"📊 Vocabulary: {len(self.vocabulary)} words")
        
        definitions_count = sum(1 for info in self.vocabulary.values() 
                              if 'generated_definition' in info)
        print(f"💡 Generated definitions: {definitions_count}/{len(self.vocabulary)}")
        
        print("\nCommands:")
        print("  <word>     - Look up word")
        print("  vocab      - Show vocabulary sample") 
        print("  stats      - Show statistics")
        print("  quit/exit  - Exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🔍 Enter word or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'vocab':
                    vocab_sample = sorted(list(self.vocabulary.keys()))[:20]
                    print(f"\n📝 **VOCABULARY SAMPLE** (first 20 of {len(self.vocabulary)}):")
                    for i, word in enumerate(vocab_sample, 1):
                        info = self.vocabulary[word]
                        word_type = info['type']
                        freq = info['frequency']
                        has_def = '✓' if 'generated_definition' in info else '○'
                        print(f"  {i:2d}. {word} ({word_type}, freq: {freq}) {has_def}")
                
                elif user_input.lower() == 'stats':
                    stats = self.get_stats()
                    print(f"\n📊 **STATISTICS**")
                    print(f"  Total words: {stats['total_words']}")
                    print(f"  With definitions: {stats['with_definitions']}")
                    print(f"  Word types:")
                    for word_type, count in stats['word_types'].items():
                        print(f"    {word_type.replace('_', ' ').title()}: {count}")
                
                else:
                    result = self.define_word(user_input)
                    print(f"\n{self.format_definition(result)}")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        type_counts = defaultdict(int)
        definitions_count = 0
        
        for info in self.vocabulary.values():
            word_type = info.get('type', 'unknown')
            type_counts[word_type] += 1
            
            if 'generated_definition' in info:
                definitions_count += 1
        
        return {
            'total_words': len(self.vocabulary),
            'with_definitions': definitions_count,
            'word_types': dict(type_counts)
        }

async def build_complete_dictionary():
    """Build complete book dictionary with definitions and GraphRAG"""
    
    print("📚 Complete Book Dictionary Builder")
    print("="*60)
    
    # Check vLLM server
    if not await check_vllm_server():
        return
    
    # Read book content
    book_files = ["book2.txt"]
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
    
    # Step 1: Extract vocabulary
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING VOCABULARY")
    print("="*60)
    
    extractor = BookVocabularyExtractor()
    vocabulary = extractor.extract_vocabulary(book_text, min_frequency=3)
    
    # Step 2: Generate definitions
    print("\n" + "="*60)
    print("STEP 2: GENERATING DEFINITIONS")
    print("="*60)
    
    definition_generator = BookDefinitionGenerator(vocabulary)
    await definition_generator.generate_all_definitions(batch_size=MAX_CONCURRENT_DEFINITIONS)
    
    # Update vocabulary with definitions
    for word, definition in definition_generator.generated_definitions.items():
        if word in vocabulary:
            vocabulary[word]['generated_definition'] = definition
            vocabulary[word]['has_definition'] = True
    
    # Step 3: Create chunks for GraphRAG
    print("\n" + "="*60)
    print("STEP 3: CREATING GRAPHRAG CHUNKS")
    print("="*60)
    
    vocab_chunks = create_vocabulary_chunks(vocabulary, definition_generator.generated_definitions)
    
    # Add some context chunks
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(book_text)
        
        context_chunks = []
        chunk_size = 250
        overlap = 50
        max_tokens = 10000  # Limit for memory
        
        for i in range(0, min(len(tokens), max_tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk = encoding.decode(chunk_tokens)
            context_chunks.append(f"[Context] {chunk}")
        
        all_chunks = vocab_chunks[:1000] + context_chunks[:50]  # Memory limits
        print(f"📖 Total chunks: {len(all_chunks)}")
        
    except Exception as e:
        print(f"⚠️ Tiktoken chunking failed: {e}")
        all_chunks = vocab_chunks[:1000]
        print(f"📖 Using vocabulary chunks only: {len(all_chunks)}")
    
    # Step 4: Build GraphRAG
    print("\n" + "="*60)
    print("STEP 4: BUILDING GRAPHRAG")
    print("="*60)
    
    try:
        rag = GraphRAG(
            working_dir=WORKING_DIR,
            embedding_func=local_embedding,
            best_model_func=book_dictionary_vllm_complete,
            cheap_model_func=book_dictionary_vllm_complete,
            best_model_max_token_size=MAX_TOKEN,
            cheap_model_max_token_size=MAX_TOKEN,
            best_model_max_async=5,  # Conservative
            cheap_model_max_async=5,
            convert_response_to_json_func=convert_to_json,
            embedding_func_max_async=3,
            embedding_batch_num=4,
        )
        print("✅ GraphRAG initialized")
        
        rag.insert(all_chunks)
        print("✅ GraphRAG knowledge graph built!")
        
    except Exception as e:
        print(f"❌ GraphRAG build failed: {e}")
        print("Continuing with vocabulary-only system...")
    
    # Step 5: Save everything
    print("\n" + "="*60)
    print("STEP 5: SAVING RESULTS")
    print("="*60)
    
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Save vocabulary
    vocab_path = os.path.join(WORKING_DIR, "extracted_vocabulary.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    
    # Save definitions
    definitions_path = os.path.join(WORKING_DIR, "generated_definitions.json")
    with open(definitions_path, 'w', encoding='utf-8') as f:
        json.dump(definition_generator.generated_definitions, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Vocabulary saved: {vocab_path}")
    print(f"✅ Definitions saved: {definitions_path}")
    
    # Show summary
    print("\n" + "="*60)
    print("✅ COMPLETE BOOK DICTIONARY BUILT!")
    print("="*60)
    
    definitions_count = len(definition_generator.generated_definitions)
    print(f"📊 Extracted: {len(vocabulary)} vocabulary terms")
    print(f"💡 Generated: {definitions_count} definitions")
    print(f"🕸️ GraphRAG: Knowledge graph with {len(all_chunks)} chunks")
    print(f"📁 Cache: {WORKING_DIR}")
    
    # Show sample definitions
    print(f"\n📖 **SAMPLE DEFINITIONS:**")
    sample_words = list(definition_generator.generated_definitions.keys())[:5]
    for i, word in enumerate(sample_words, 1):
        definition = definition_generator.generated_definitions[word][:100] + "..."
        word_info = vocabulary[word]
        print(f"{i}. **{word.upper()}** ({word_info['type']}): {definition}")
    
    print(f"\n🚀 Ready to use! Test some queries:")
    sample_test_words = list(vocabulary.keys())[:3]
    for word in sample_test_words:
        print(f"   python {sys.argv[0]} query {word}")

def main():
    """Main function - handles both building and querying"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "build":
            # Build the dictionary
            asyncio.run(build_complete_dictionary())
            
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
                dictionary.interactive_query()
                
        elif command == "interactive":
            # Interactive query mode
            if not os.path.exists(WORKING_DIR):
                print("❌ No dictionary cache found. Build first with: python script.py build")
                return
            
            dictionary = BookDictionaryQuery()
            dictionary.interactive_query()
            
        elif command == "test":
            # Test with GraphRAG if available
            if not os.path.exists(WORKING_DIR):
                print("❌ No dictionary cache found. Build first with: python script.py build")
                return
                
            print("🧪 Testing GraphRAG queries...")
            
            try:
                # Try to load GraphRAG
                rag = GraphRAG(
                    working_dir=WORKING_DIR,
                    embedding_func=local_embedding,
                    best_model_func=book_dictionary_vllm_complete,
                    cheap_model_func=book_dictionary_vllm_complete,
                )
                
                # Load vocabulary for test words
                dictionary = BookDictionaryQuery()
                test_words = list(dictionary.vocabulary.keys())[:3]
                
                print(f"Testing words: {test_words}")
                
                for word in test_words:
                    print(f"\n📖 GraphRAG Query: '{word}'")
                    try:
                        result = rag.query(word, param=QueryParam(mode="local", top_k=3))
                        print(f"💡 Result: {result[:200]}...")
                    except Exception as e:
                        print(f"❌ Query failed: {e}")
                        
            except Exception as e:
                print(f"❌ GraphRAG not available: {e}")
                print("Using vocabulary-only mode...")
                dictionary = BookDictionaryQuery()
                test_words = list(dictionary.vocabulary.keys())[:3]
                
                for word in test_words:
                    result = dictionary.define_word(word)
                    print(f"\n{dictionary.format_definition(result)}")
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Usage:")
            print(f"  python {sys.argv[0]} build           - Build dictionary from book")
            print(f"  python {sys.argv[0]} query <word>    - Look up specific word")
            print(f"  python {sys.argv[0]} interactive     - Interactive query mode")
            print(f"  python {sys.argv[0]} test            - Test GraphRAG queries")
    
    else:
        # Default: show help
        print("📚 Complete Book Dictionary Builder")
        print("="*50)
        print("One file solution: Extract vocabulary, generate definitions, build GraphRAG")
        print("")
        print("Usage:")
        print(f"  python {sys.argv[0]} build           - Build dictionary from book2.txt/book.txt")
        print(f"  python {sys.argv[0]} query <word>    - Look up specific word")  
        print(f"  python {sys.argv[0]} interactive     - Interactive query mode")
        print(f"  python {sys.argv[0]} test            - Test GraphRAG functionality")
        print("")
        print("Features:")
        print("  ✅ Fixes scipy.linalg.triu issue with numpy patch")
        print("  ✅ Extracts vocabulary from book content")
        print("  ✅ Generates definitions using vLLM")
        print("  ✅ Builds GraphRAG knowledge graph")
        print("  ✅ Provides query interface")
        print("  ✅ Everything in one file!")
        print("")
        print("Requirements:")
        print("  - vLLM server running on localhost:8000")
        print("  - book2.txt or book.txt in current directory")
        print("  - Required packages: nano-graphrag, sentence-transformers, openai")

if __name__ == "__main__":
    main()
