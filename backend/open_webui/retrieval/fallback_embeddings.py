import logging
import numpy as np
import requests
import hashlib
import json
import os
import sys
from typing import List, Union, Optional, Dict, Any

# Try to patch the _lzma import problem by providing a mock module
if '_lzma' not in sys.modules:
    try:
        import backports.lzma as _lzma
        sys.modules['_lzma'] = _lzma
        logging.info("Successfully loaded backports.lzma as _lzma")
    except ImportError:
        # Create a mock _lzma module to avoid import errors
        class MockLZMA:
            def __getattr__(self, name):
                logging.warning(f"Mock _lzma.{name} was called")
                return lambda *args, **kwargs: None
                
        sys.modules['_lzma'] = MockLZMA()
        logging.info("Created mock _lzma module")

log = logging.getLogger(__name__)

class FallbackEmbeddingModel:
    """
    A fallback embedding model that doesn't rely on sentence-transformers.
    It uses simple hashing-based encoding when external embedding services 
    are unavailable.
    """
    
    def __init__(self, model_name: str = "fallback/text-hash-embedding"):
        self.model_name = model_name
        self.embedding_size = 384  # Same as all-MiniLM-L6-v2
        log.info(f"Initialized FallbackEmbeddingModel: {model_name}")
        
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """
        Creates embeddings for the given texts.
        For fallback, we use a hash-based approach to generate pseudo-embeddings.
        """
        if isinstance(texts, str):
            return self._hash_to_vector(texts)
        else:
            return [self._hash_to_vector(text) for text in texts]
    
    def _hash_to_vector(self, text: str) -> List[float]:
        """
        Convert a text to a deterministic vector using hash functions.
        """
        # Create a deterministic seed from the text
        text_hash = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % 10**8
        
        # Use the hash as a seed for numpy's random generator
        rng = np.random.RandomState(text_hash)
        
        # Generate a random vector
        vector = rng.rand(self.embedding_size).astype(np.float32)
        
        # Normalize to unit length (cosine similarity)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()
    
    def __str__(self):
        return f"FallbackEmbeddingModel({self.model_name})"


def try_external_embeddings(text: Union[str, List[str]], api_url: Optional[str] = None) -> Optional[List[List[float]]]:
    """
    Try to get embeddings from external API if available.
    Returns None if not available or fails.
    """
    if not api_url:
        return None
        
    try:
        # Try to connect to an external API for embeddings
        headers = {"Content-Type": "application/json"}
        payload = {"texts": [text] if isinstance(text, str) else text}
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if "embeddings" in result:
            return result["embeddings"]
        
    except Exception as e:
        log.warning(f"External embedding API call failed: {e}")
        
    return None


def get_fallback_embedding_model() -> FallbackEmbeddingModel:
    """
    Returns a fallback embedding model that doesn't depend on SentenceTransformers.
    """
    return FallbackEmbeddingModel()
