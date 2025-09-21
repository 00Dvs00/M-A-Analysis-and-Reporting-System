"""API response caching implementation."""
import os
import json
import hashlib
import time
from typing import Any, Dict, Optional
from pathlib import Path

class APICache:
    """Cache API responses to disk for reproducibility and faster development."""
    
    def __init__(self, cache_dir: str = "cache/api", ttl: int = 86400):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files.
            ttl: Time-to-live for cache entries in seconds (default: 24 hours).
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        # Sort params to ensure consistent keys
        sorted_params = dict(sorted(params.items()))
        # Create a string representation
        cache_string = f"{endpoint}:{json.dumps(sorted_params)}"
        # Generate MD5 hash
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a cached response if it exists and is not expired."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with cache_path.open('r') as f:
                cached = json.load(f)
            
            # Check if cache is expired
            if time.time() - cached['timestamp'] > self.ttl:
                return None
            
            return cached['data']
        except (json.JSONDecodeError, KeyError):
            return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        """Cache an API response."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': time.time(),
            'endpoint': endpoint,
            'params': params,
            'data': data
        }
        
        with cache_path.open('w') as f:
            json.dump(cache_data, f, indent=2)
    
    def clear(self, older_than: Optional[int] = None):
        """Clear expired cache entries.
        
        Args:
            older_than: Optional age in seconds. If provided, clear entries older than this.
        """
        now = time.time()
        max_age = older_than if older_than is not None else self.ttl
        
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with cache_file.open('r') as f:
                    cached = json.load(f)
                if now - cached['timestamp'] > max_age:
                    cache_file.unlink()
            except (json.JSONDecodeError, KeyError, OSError):
                # If there's any error reading the cache file, remove it
                cache_file.unlink()