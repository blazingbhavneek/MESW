# Book Dictionary Query Interface
# Query vocabulary and terms extracted from book content - No LLM required

import sys
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import faiss
import pickle
import os
import time
import re
from difflib import get_close_matches
from typing import List, Dict, Any, Optional, Tuple
import json
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)

WORKING_DIR = "./book_dictionary_cache"

# Embedding setup
EMBED_MODEL = SentenceTransformer("models/embeddings", local_files_only=True)

class BookDictionaryVectorStore:
    def __init__(self):
        self.embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []  # Store actual text chunks
        self.chunk_metadata = {}  # Store metadata for each chunk
        self.vocabulary = {}  # Store extracted vocabulary
        self.word_to_chunks = {}  # Map words to their chunk indices
        
        self._load_book_dictionary_data()
    
    def _load_book_dictionary_data(self):
        """Load vocabulary and chunks from book dictionary cache"""
        print("📚 Loading book dictionary data...")
        
        # Load extracted vocabulary
        vocab_path = os.path.join(WORKING_DIR, "extracted_vocabulary.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
            print(f"✅ Loaded {len(self.vocabulary)} vocabulary terms from book")
        
        # Load text chunks
        text_chunks_path = os.path.join(WORKING_DIR, "text_chunks.json")
        if os.path.exists(text_chunks_path):
            with open(text_chunks_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                self.chunks = [item['content'] for item in chunk_data]
                print(f"✅ Loaded {len(self.chunks)} text chunks from book")
        
        # Load vector data
        self._load_vector_data()
        self._create_word_mappings()
        
        if not self.vocabulary and not self.chunks:
            print("❌ No book dictionary data found. Please run the book dictionary builder first.")
    
    def _load_vector_data(self):
        """Load vector data from book dictionary storage"""
        storage_locations = [
            "book_faiss_entities.index",
            "book_faiss_relationships.index", 
            "book_faiss_communities.index",
            "faiss_entities.index",
        ]
        
        for storage_name in storage_locations:
            storage_path = os.path.join(WORKING_DIR, storage_name)
            if os.path.exists(storage_path):
                try:
                    self.index = faiss.read_index(storage_path)
                    print(f"✅ Loaded vector index: {storage_name} ({self.index.ntotal} vectors)")
                    
                    # Load metadata
                    meta_path = storage_path + '.meta'
                    if os.path.exists(meta_path):
                        with open(meta_path, 'rb') as f:
                            meta_data = pickle.load(f)
                            if len(meta_data) >= 4 and isinstance(meta_data[3], dict):
                                self.chunk_metadata = meta_data[3]
                                print(f"✅ Loaded metadata for {len(self.chunk_metadata)} chunks")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {storage_name}: {e}")
    
    def _create_word_mappings(self):
        """Create mappings from words to chunks containing them"""
        print("🔍 Creating word-to-chunk mappings...")
        
        # Map vocabulary words to chunks
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk.lower()))
            
            for word in chunk_words:
                if word not in self.word_to_chunks:
                    self.word_to_chunks[word] = []
                self.word_to_chunks[word].append(i)
        
        print(f"✅ Mapped {len(self.word_to_chunks)} unique words to chunks")
    
    def search_word_vector(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Search for word using vector similarity"""
        if self.index is None or not self.chunks:
            return []
        
        # Get query embedding
        query_embedding = EMBED_MODEL.encode([query], normalize_embeddings=True)
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in vector index
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vector, search_k)
        
        results = []
        seen_content = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            
            # Avoid very similar content
            chunk_key = chunk[:150]
            if chunk_key in seen_content:
                continue
            seen_content.add(chunk_key)
            
            # Determine chunk type
            chunk_type = "vocabulary" if chunk.startswith("**") else "context"
            
            results.append((chunk, float(score), chunk_type))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_word_direct(self, query: str) -> List[Tuple[str, str]]:
        """Direct search for word in vocabulary and chunks"""
        query_lower = query.lower()
        results = []
        
        # First check extracted vocabulary
        if query_lower in self.vocabulary:
            vocab_entry = self.vocabulary[query_lower]
            contexts = vocab_entry.get('contexts', [])
            word_type = vocab_entry.get('type', 'word')
            frequency = vocab_entry.get('frequency', 0)
            
            # Create a formatted vocabulary entry
            vocab_text = f"**{query.title()}** ({word_type}, appears {frequency} times in book)\n"
            if contexts:
                vocab_text += "Context examples from book:\n"
                for i, context in enumerate(contexts[:2], 1):
                    vocab_text += f"{i}. {context[:150]}...\n"
            
            results.append((vocab_text, "vocabulary"))
        
        # Then search in chunks
        if query_lower in self.word_to_chunks:
            chunk_indices = self.word_to_chunks[query_lower][:3]
            for idx in chunk_indices:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    # Skip if we already have this as vocabulary entry
                    if not any(result[1] == "vocabulary" for result in results):
                        results.append((chunk, "context"))
        
        return results

class BookDictionaryQuery:
    def __init__(self):
        self.vector_store = BookDictionaryVectorStore()
    
    def get_all_vocabulary_words(self) -> List[str]:
        """Get list of all vocabulary words extracted from book"""
        return sorted(list(self.vector_store.vocabulary.keys()))
    
    def suggest_words(self, query: str, n: int = 5) -> List[str]:
        """Suggest similar words from book vocabulary"""
        query = query.lower().strip()
        vocab_words = list(self.vector_store.vocabulary.keys())
        all_words = list(self.vector_store.word_to_chunks.keys())
        
        # Direct match in vocabulary
        if query in vocab_words:
            return [query]
        
        # Close matches in vocabulary first
        vocab_matches = get_close_matches(query, vocab_words, n=n, cutoff=0.6)
        
        # Then close matches in all words
        all_matches = get_close_matches(query, all_words, n=n, cutoff=0.6)
        
        # Prefix matches
        vocab_prefix = [w for w in vocab_words if w.startswith(query)][:n]
        all_prefix = [w for w in all_words if w.startswith(query)][:n]
        
        # Combine and prioritize vocabulary words
        combined = []
        combined.extend(vocab_matches)
        combined.extend([w for w in vocab_prefix if w not in combined])
        combined.extend([w for w in all_matches if w not in combined])
        combined.extend([w for w in all_prefix if w not in combined])
        
        return combined[:n]
    
    def get_word_info(self, word: str) -> Dict[str, Any]:
        """Get detailed information about a word from the book"""
        word_lower = word.lower().strip()
        
        if word_lower in self.vector_store.vocabulary:
            vocab_entry = self.vector_store.vocabulary[word_lower]
            return {
                'word': word,
                'found_in_vocabulary': True,
                'type': vocab_entry.get('type', 'unknown'),
                'frequency': vocab_entry.get('frequency', 0),
                'contexts': vocab_entry.get('contexts', []),
                'original_case': vocab_entry.get('original_case', word),
                'definition_source': 'extracted_from_book'
            }
        else:
            return {
                'word': word,
                'found_in_vocabulary': False,
                'type': 'unknown',
                'frequency': 0,
                'contexts': [],
                'definition_source': 'not_found'
            }
    
    def define_word(self, word: str, method: str = "hybrid", include_context: bool = True) -> Dict[str, Any]:
        """Define a word using book content - no LLM needed"""
        start_time = time.time()
        word_lower = word.lower().strip()
        
        definitions = []
        word_info = self.get_word_info(word)
        
        if method in ["direct", "hybrid"]:
            # Direct search in vocabulary and chunks
            direct_results = self.vector_store.search_word_direct(word)
            for content, source_type in direct_results:
                definitions.append({
                    'content': content,
                    'source': source_type,
                    'method': 'direct',
                    'score': 1.0 if source_type == 'vocabulary' else 0.8
                })
        
        if method in ["vector", "hybrid"] and self.vector_store.index:
            # Vector similarity search
            vector_results = self.vector_store.search_word_vector(word, top_k=3)
            for content, score, chunk_type in vector_results:
                # Avoid duplicates
                if not any(d['content'] == content for d in definitions):
                    definitions.append({
                        'content': content,
                        'source': chunk_type,
                        'method': 'vector',
                        'score': float(score)
                    })
        
        # Sort by score and source priority
        definitions.sort(key=lambda x: (x['score'], 1 if x['source'] == 'vocabulary' else 0), reverse=True)
        
        # Format the final definition
        if definitions:
            primary_def = definitions[0]['content']
            
            # Add additional context if requested
            additional_contexts = []
            if include_context and len(definitions) > 1:
                for d in definitions[1:3]:  # Up to 2 additional contexts
                    if d['content'] != primary_def:
                        content = d['content'][:300] + "..." if len(d['content']) > 300 else d['content']
                        additional_contexts.append(f"[{d['source'].title()}] {content}")
            
            final_definition = primary_def
            if additional_contexts:
                final_definition += "\n\nAdditional contexts:\n" + "\n\n".join(additional_contexts)
        else:
            final_definition = None
        
        end_time = time.time()
        
        return {
            'word': word,
            'definition': final_definition,
            'found': bool(definitions),
            'word_info': word_info,
            'num_sources': len(definitions),
            'query_time': end_time - start_time,
            'suggestions': self.suggest_words(word) if not definitions else [],
            'method': method
        }
    
    def batch_define(self, words: List[str], method: str = "hybrid") -> Dict[str, Dict[str, Any]]:
        """Define multiple words from book vocabulary"""
        print(f"📚 Looking up {len(words)} words in book dictionary...")
        results = {}
        
        for word in words:
            results[word.strip()] = self.define_word(word.strip(), method)
            time.sleep(0.05)  # Small delay
        
        return results
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the book vocabulary"""
        vocab = self.vector_store.vocabulary
        
        if not vocab:
            return {'total_words': 0}
        
        # Analyze vocabulary types
        type_counts = defaultdict(int)
        freq_distribution = []
        
        for word_info in vocab.values():
            word_type = word_info.get('type', 'unknown')
            frequency = word_info.get('frequency', 0)
            
            type_counts[word_type] += 1
            freq_distribution.append(frequency)
        
        return {
            'total_words': len(vocab),
            'word_types': dict(type_counts),
            'avg_frequency': sum(freq_distribution) / len(freq_distribution) if freq_distribution else 0,
            'max_frequency': max(freq_distribution) if freq_distribution else 0,
            'min_frequency': min(freq_distribution) if freq_distribution else 0,
            'total_chunks': len(self.vector_store.chunks),
            'vector_index_size': self.vector_store.index.ntotal if self.vector_store.index else 0
        }
    
    def search_by_type(self, word_type: str, limit: int = 20) -> List[Tuple[str, Dict]]:
        """Search words by type (proper_noun, technical_term, common_word)"""
        results = []
        
        for word, info in self.vector_store.vocabulary.items():
            if info.get('type') == word_type:
                results.append((word, info))
        
        # Sort by frequency
        results.sort(key=lambda x: x[1].get('frequency', 0), reverse=True)
        return results[:limit]
    
    def find_related_words(self, word: str, top_k: int = 8) -> List[str]:
        """Find words that appear in similar contexts in the book"""
        word_lower = word.lower().strip()
        related_words = set()
        
        # Get chunks containing the word
        if word_lower in self.vector_store.word_to_chunks:
            chunk_indices = self.vector_store.word_to_chunks[word_lower][:5]
            
            for idx in chunk_indices:
                if idx < len(self.vector_store.chunks):
                    chunk = self.vector_store.chunks[idx]
                    # Extract other significant words from these chunks
                    words_in_chunk = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.lower())
                    for w in words_in_chunk:
                        if w != word_lower and w in self.vector_store.vocabulary:
                            related_words.add(w)
        
        return list(related_words)[:top_k]
    
    def format_definition(self, result: Dict[str, Any], show_details: bool = True) -> str:
        """Format definition result for display"""
        word = result['word']
        definition = result['definition']
        word_info = result['word_info']
        suggestions = result.get('suggestions', [])
        query_time = result['query_time']
        method = result.get('method', 'unknown')
        num_sources = result.get('num_sources', 0)
        
        # Header
        output = f"📖 **{word.upper()}**"
        if word_info['found_in_vocabulary']:
            word_type = word_info['type']
            frequency = word_info['frequency']
            output += f" ({word_type}, appears {frequency} times in book)"
        
        if num_sources > 0:
            output += f" [{method} search, {num_sources} sources]"
        
        output += "\n" + "-" * (len(word) + 20) + "\n"
        
        # Definition content
        if definition:
            output += f"{definition}\n"
        else:
            output += f"❌ No definition found in book for '{word}'\n"
            if suggestions:
                output += f"\n💡 Similar words from book: {', '.join(suggestions[:5])}\n"
        
        # Additional details
        if show_details:
            output += f"\n⏱️ Query time: {query_time:.3f}s"
            if word_info['found_in_vocabulary'] and word_info['contexts']:
                output += f" | Found in {len(word_info['contexts'])} book contexts"
        
        return output
    
    async def interactive_book_dictionary(self):
        """Interactive book dictionary interface"""
        stats = self.get_vocabulary_stats()
        
        print(f"\n📚 Book Dictionary Interface")
        print("="*50)
        print(f"📊 Vocabulary: {stats['total_words']} words extracted from book")
        print(f"📄 Chunks: {stats.get('total_chunks', 0)} text segments")
        print("🚀 Pure vector/text search - No LLM required!")
        
        print("\nCommands:")
        print("  <word>              - Look up word from book")
        print("  vector <word>       - Vector similarity search")
        print("  direct <word>       - Direct vocabulary search") 
        print("  batch <w1,w2,w3>    - Look up multiple words")
        print("  related <word>      - Find related words from book")
        print("  type <type>         - Show words by type (proper_noun, technical_term, common_word)")
        print("  suggest <word>      - Get word suggestions")
        print("  vocab               - Show vocabulary sample")
        print("  stats               - Show detailed statistics")
        print("  quit/exit           - Exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🔍 Enter word or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Happy reading!")
                    break
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                query = parts[1] if len(parts) > 1 else ""
                
                if command in ['vector', 'direct'] and query:
                    result = self.define_word(query, method=command)
                    print(f"\n{self.format_definition(result)}")
                
                elif command == 'batch':
                    if not query:
                        print("❌ Please specify words: batch word1,word2,word3")
                        continue
                    words = [w.strip() for w in query.split(',')]
                    results = self.batch_define(words)
                    
                    print(f"\n📚 **BATCH LOOKUPS FROM BOOK**")
                    print("="*40)
                    for word, result in results.items():
                        print(f"\n{self.format_definition(result, show_details=False)}")
                
                elif command == 'related':
                    if not query:
                        print("❌ Please specify a word: related <word>")
                        continue
                    related = self.find_related_words(query)
                    if related:
                        print(f"\n🔗 **RELATED WORDS from book for '{query.upper()}':**")
                        for i, word in enumerate(related, 1):
                            word_info = self.get_word_info(word)
                            freq = word_info['frequency']
                            word_type = word_info['type']
                            print(f"  {i}. {word} ({word_type}, freq: {freq})")
                    else:
                        print(f"❌ No related words found for '{query}' in book")
                
                elif command == 'type':
                    if not query:
                        print("❌ Please specify type: type proper_noun|technical_term|common_word")
                        continue
                    words_by_type = self.search_by_type(query, limit=15)
                    if words_by_type:
                        print(f"\n📝 **{query.upper().replace('_', ' ')} WORDS from book:**")
                        for i, (word, info) in enumerate(words_by_type, 1):
                            freq = info['frequency']
                            print(f"  {i:2d}. {word} (frequency: {freq})")
                    else:
                        print(f"❌ No words of type '{query}' found")
                
                elif command == 'suggest':
                    if not query:
                        print("❌ Please specify a word: suggest <word>")
                        continue
                    suggestions = self.suggest_words(query, n=10)
                    if suggestions:
                        print(f"\n💡 **SUGGESTIONS for '{query}' from book vocabulary:**")
                        for i, suggestion in enumerate(suggestions, 1):
                            word_info = self.get_word_info(suggestion)
                            freq = word_info['frequency']
                            word_type = word_info['type']
                            print(f"  {i}. {suggestion} ({word_type}, freq: {freq})")
                    else:
                        print(f"❌ No suggestions found for '{query}'")
                
                elif command == 'vocab':
                    vocab_sample = sorted(list(self.vector_store.vocabulary.keys()))[:20]
                    print(f"\n📝 **VOCABULARY SAMPLE** (first 20 of {len(self.vector_store.vocabulary)}):")
                    for i, word in enumerate(vocab_sample, 1):
                        info = self.vector_store.vocabulary[word]
                        word_type = info['type']
                        freq = info['frequency']
                        print(f"  {i:2d}. {word} ({word_type}, freq: {freq})")
                    
                    if len(self.vector_store.vocabulary) > 20:
                        print(f"\n... and {len(self.vector_store.vocabulary) - 20} more words from the book")
                
                elif command == 'stats':
                    stats = self.get_vocabulary_stats()
                    print(f"\n📊 **BOOK DICTIONARY STATISTICS**")
                    print(f"  Total vocabulary words: {stats['total_words']}")
                    print(f"  Total text chunks: {stats['total_chunks']}")
                    print(f"  Vector index size: {stats['vector_index_size']}")
                    print(f"  Average word frequency: {stats['avg_frequency']:.1f}")
                    print(f"  Most frequent word appears: {stats['max_frequency']} times")
                    
                    print(f"\n  📂 Word types in book:")
                    for word_type, count in stats['word_types'].items():
                        print(f"    {word_type.replace('_', ' ').title()}: {count} words")
                    
                    # Cache size
                    cache_size = 0
                    if os.path.exists(WORKING_DIR):
                        for root, dirs, files in os.walk(WORKING_DIR):
                            for file in files:
                                filepath = os.path.join(root, file)
                                cache_size += os.path.getsize(filepath)
                        print(f"  Cache size: {cache_size / 1024 / 1024:.1f} MB")
                
                else:
                    # Default: look up word
                    word = query if query else command
                    result = self.define_word(word, method="hybrid")
                    print(f"\n{self.format_definition(result)}")
            
            except KeyboardInterrupt:
                print("\n👋 Happy reading!")
                break
            except Exception as e:
                print(f"❌ Error processing command: {e}")
    
    def quick_lookup(self, word: str, method: str = "hybrid"):
        """Quick word lookup for command line usage"""
        result = self.define_word(word, method=method)
        print(self.format_definition(result))

def main():
    """Main function"""
    print("📚 Book Dictionary Query System")
    print("="*55)
    print("🚀 Look up vocabulary extracted from your book!")
    
    # Check if book dictionary cache exists
    if not os.path.exists(WORKING_DIR):
        print(f"❌ Book dictionary cache not found: {WORKING_DIR}")
        print("Please run the book dictionary GraphRAG builder script first.")
        return
    
    # Initialize query interface
    dictionary = BookDictionaryQuery()
    
    if not dictionary.vector_store.vocabulary and not dictionary.vector_store.chunks:
        print("❌ No book dictionary data loaded. Please check your cache.")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            asyncio.run(dictionary.interactive_book_dictionary())
        elif sys.argv[1] == "batch":
            if len(sys.argv) > 2:
                words = sys.argv[2].split(',')
                results = dictionary.batch_define(words)
                for word, result in results.items():
                    print(f"\n{dictionary.format_definition(result)}")
            else:
                print("❌ Please provide comma-separated words for batch lookup")
        elif sys.argv[1] == "vocab":
            vocab_words = dictionary.get_all_vocabulary_words()[:30]
            print(f"\n📝 **BOOK VOCABULARY** (first 30 of {len(dictionary.get_all_vocabulary_words())}):")
            for i, word in enumerate(vocab_words, 1):
                info = dictionary.get_word_info(word)
                print(f"  {i:2d}. {word} ({info['type']}, freq: {info['frequency']})")
        elif sys.argv[1] == "stats":
            stats = dictionary.get_vocabulary_stats()
            print(f"\n📊 **BOOK DICTIONARY STATISTICS**")
            print(f"  Vocabulary extracted: {stats['total_words']} words")
            for word_type, count in stats['word_types'].items():
                print(f"  {word_type.replace('_', ' ').title()}: {count}")
        else:
            # Direct word lookup
            word = " ".join(sys.argv[1:])
            dictionary.quick_lookup(word)
    else:
        # Default to interactive mode
        asyncio.run(dictionary.interactive_book_dictionary())

if __name__ == "__main__":
    main()
