"""
Embedding Module
Tạo embeddings cho các n-gram và văn bản sử dụng Together API
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Data class for embedding results"""
    text: str
    embedding: List[float]
    model: str
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'embedding': self.embedding,
            'model': self.model,
            'timestamp': self.timestamp
        }

class EmbeddingGenerator:
    """Class để tạo embeddings sử dụng Together API"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
                 batch_size: int = 10,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize embedding generator
        
        Args:
            api_key: Together API key (if None, will try to get from environment)
            model: Model name for embeddings
            batch_size: Batch size for processing
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError("Together API key is required. Set TOGETHER_API_KEY environment variable.")
        
        # Import Together client (lazy import to handle missing dependency)
        try:
            from together import Together
            self.client = Together(api_key=self.api_key)
            logger.info(f"✅ Together API client initialized with model: {self.model}")
        except ImportError:
            logger.error("Together library not installed. Please install with: pip install together")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {e}")
            raise
    
    def _create_embedding_with_retry(self, text: str) -> Optional[List[float]]:
        """
        Create embedding with retry logic
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            return None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip()
                )
                
                if response.data and len(response.data) > 0:
                    return response.data[0].embedding
                else:
                    logger.warning(f"Empty response for text: {text[:50]}...")
                    return None
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for text '{text[:50]}...': {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for text: {text[:50]}...")
                    return None
        
        return None
    
    def create_embedding(self, text: str) -> Optional[EmbeddingResult]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult or None if failed
        """
        embedding = self._create_embedding_with_retry(text)
        
        if embedding is not None:
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                timestamp=time.time()
            )
        
        return None
    
    def create_embeddings_batch(self, texts: List[str], 
                              show_progress: bool = True) -> List[EmbeddingResult]:
        """
        Create embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress information
            
        Returns:
            List of EmbeddingResult objects (excludes failed embeddings)
        """
        if not texts:
            return []
        
        results = []
        total_texts = len(texts)
        
        if show_progress:
            logger.info(f"🔄 Creating embeddings for {total_texts} texts...")
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_start = i + 1
            batch_end = min(i + self.batch_size, total_texts)
            
            if show_progress:
                logger.info(f"Processing batch {batch_start}-{batch_end}/{total_texts}")
            
            # Use ThreadPoolExecutor for concurrent requests within batch
            with ThreadPoolExecutor(max_workers=min(5, len(batch))) as executor:
                # Submit all requests in the batch
                future_to_text = {
                    executor.submit(self.create_embedding, text): text 
                    for text in batch
                }
                
                # Collect results
                for future in as_completed(future_to_text):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
            # Add a small delay between batches to be respectful to the API
            if i + self.batch_size < total_texts:
                time.sleep(0.5)
        
        success_rate = len(results) / total_texts * 100
        if show_progress:
            logger.info(f"✅ Created {len(results)}/{total_texts} embeddings (success rate: {success_rate:.1f}%)")
        
        return results
    
    def create_ngram_embeddings(self, ngrams: List[Tuple[str, float]], 
                              show_progress: bool = True) -> List[Tuple[str, float, List[float]]]:
        """
        Create embeddings for n-grams with their scores
        
        Args:
            ngrams: List of (ngram, score) tuples
            show_progress: Whether to show progress
            
        Returns:
            List of (ngram, score, embedding) tuples
        """
        if not ngrams:
            return []
        
        # Extract just the n-gram texts
        ngram_texts = [ngram for ngram, _ in ngrams]
        
        # Create embeddings
        embedding_results = self.create_embeddings_batch(ngram_texts, show_progress)
        
        # Create mapping from text to embedding
        text_to_embedding = {result.text: result.embedding for result in embedding_results}
        
        # Combine with original scores
        result = []
        for ngram, score in ngrams:
            if ngram in text_to_embedding:
                result.append((ngram, score, text_to_embedding[ngram]))
        
        return result
    
    def get_embedding_stats(self, embeddings: List[EmbeddingResult]) -> Dict:
        """
        Get statistics about embeddings
        
        Args:
            embeddings: List of embedding results
            
        Returns:
            Dictionary with statistics
        """
        if not embeddings:
            return {"count": 0}
        
        # Extract embedding vectors
        vectors = [result.embedding for result in embeddings]
        embedding_matrix = np.array(vectors)
        
        stats = {
            "count": len(embeddings),
            "dimension": len(vectors[0]) if vectors else 0,
            "mean_magnitude": float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            "std_magnitude": float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            "mean_values": embedding_matrix.mean(axis=0).tolist(),
            "std_values": embedding_matrix.std(axis=0).tolist(),
            "model": embeddings[0].model if embeddings else None
        }
        
        return stats

# Convenience functions
def create_embeddings(texts: List[str], 
                     api_key: Optional[str] = None,
                     model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
                     batch_size: int = 10) -> List[EmbeddingResult]:
    """
    Quick function to create embeddings
    
    Args:
        texts: List of texts to embed
        api_key: Together API key
        model: Embedding model to use
        batch_size: Batch size for processing
        
    Returns:
        List of EmbeddingResult objects
    """
    generator = EmbeddingGenerator(api_key=api_key, model=model, batch_size=batch_size)
    return generator.create_embeddings_batch(texts)

def embed_ngrams(ngrams: List[Tuple[str, float]], 
                api_key: Optional[str] = None) -> List[Tuple[str, float, List[float]]]:
    """
    Create embeddings for n-grams
    
    Args:
        ngrams: List of (ngram, score) tuples
        api_key: Together API key
        
    Returns:
        List of (ngram, score, embedding) tuples
    """
    generator = EmbeddingGenerator(api_key=api_key)
    return generator.create_ngram_embeddings(ngrams)

# Example usage and testing
if __name__ == "__main__":
    # Test data
    sample_ngrams = [
        ("python developer", 10.5),
        ("machine learning", 8.3),
        ("data science", 7.2),
        ("javascript framework", 6.1),
        ("cloud computing", 5.9),
        ("backend engineer", 5.4),
        ("api development", 4.8),
        ("database design", 4.2),
        ("devops engineer", 3.9),
        ("artificial intelligence", 3.5)
    ]
    
    print("🧪 Testing Embedding Generator")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("⚠️  TOGETHER_API_KEY not found in environment variables")
        print("Set your API key to test the embedding functionality")
        print("Example: set TOGETHER_API_KEY=your_api_key_here")
    else:
        try:
            # Test embedding generation
            generator = EmbeddingGenerator()
            
            # Test single embedding
            result = generator.create_embedding("python machine learning")
            if result:
                print(f"✅ Single embedding created:")
                print(f"   Text: {result.text}")
                print(f"   Dimension: {len(result.embedding)}")
                print(f"   Model: {result.model}")
            
            print("\n" + "=" * 30)
            
            # Test batch embeddings
            sample_texts = [ngram for ngram, _ in sample_ngrams[:5]]
            batch_results = generator.create_embeddings_batch(sample_texts)
            
            print(f"✅ Batch embeddings created: {len(batch_results)}/{len(sample_texts)}")
            
            # Test n-gram embeddings
            ngram_embeddings = generator.create_ngram_embeddings(sample_ngrams[:3])
            print(f"✅ N-gram embeddings created: {len(ngram_embeddings)}")
            
            for ngram, score, embedding in ngram_embeddings:
                print(f"   - {ngram:<20} (score: {score:.1f}, dim: {len(embedding)})")
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            print("Make sure your Together API key is valid and you have internet connection")
