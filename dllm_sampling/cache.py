"""
Token caching system for DLLM inference acceleration.
"""

import time
from typing import Any, Dict, Optional, Tuple, List, Union
from collections import OrderedDict
import threading
import pickle
import hashlib
from dataclasses import dataclass
from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int


class TokenCache:
    """
    Thread-safe LRU cache optimized for token sequences and model outputs.
    Supports TTL (time-to-live) and memory-based eviction policies.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 500.0,
        ttl_seconds: float = 3600.0,
        enable_persistence: bool = False,
        cache_file: str = "token_cache.pkl"
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_file = cache_file
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_evictions": 0,
            "ttl_evictions": 0,
            "size_evictions": 0
        }
        
        # Load from persistence if enabled
        if self.enable_persistence:
            self._load_cache()
        
        logger.info(f"TokenCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _compute_key(self, key: Union[str, Tuple, List]) -> str:
        """Compute cache key hash."""
        if isinstance(key, str):
            return key
        
        # Convert to string and hash for complex keys
        key_str = str(key)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            return 1024  # Default estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds <= 0:
            return False
        
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
            self.stats["ttl_evictions"] += 1
            self.stats["evictions"] += 1
    
    def _evict_lru(self):
        """Remove least recently used entries."""
        while (len(self._cache) >= self.max_size or 
               self._current_memory >= self.max_memory_bytes):
            
            if not self._cache:
                break
            
            # Remove oldest entry
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size_bytes
            
            if len(self._cache) >= self.max_size:
                self.stats["size_evictions"] += 1
            else:
                self.stats["memory_evictions"] += 1
            
            self.stats["evictions"] += 1
    
    def get(self, key: Union[str, Tuple, List]) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._compute_key(key)
        
        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()
            
            if cache_key not in self._cache:
                self.stats["misses"] += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if self._is_expired(entry):
                del self._cache[cache_key]
                self._current_memory -= entry.size_bytes
                self.stats["misses"] += 1
                self.stats["ttl_evictions"] += 1
                self.stats["evictions"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.access_count += 1
            
            self.stats["hits"] += 1
            return entry.value
    
    def put(self, key: Union[str, Tuple, List], value: Any) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._compute_key(key)
        size_bytes = self._estimate_size(value)
        
        # Skip if value is too large
        if size_bytes > self.max_memory_bytes:
            logger.warning(f"Value too large to cache: {size_bytes} bytes")
            return False
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                old_entry = self._cache[cache_key]
                self._current_memory -= old_entry.size_bytes
            
            # Evict entries if necessary
            self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            self._cache[cache_key] = entry
            self._current_memory += size_bytes
            
            return True
    
    def contains(self, key: Union[str, Tuple, List]) -> bool:
        """Check if key exists in cache (without updating access)."""
        cache_key = self._compute_key(key)
        
        with self._lock:
            if cache_key not in self._cache:
                return False
            
            entry = self._cache[cache_key]
            return not self._is_expired(entry)
    
    def remove(self, key: Union[str, Tuple, List]) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if removed, False if not found
        """
        cache_key = self._compute_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache.pop(cache_key)
                self._current_memory -= entry.size_bytes
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self.stats,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "current_size": len(self._cache),
                "current_memory_mb": self._current_memory / (1024 * 1024),
                "memory_utilization": self._current_memory / self.max_memory_bytes,
                "size_utilization": len(self._cache) / self.max_size
            }
    
    def get_top_entries(self, n: int = 10) -> List[Tuple[str, int, float]]:
        """Get top N most accessed cache entries."""
        with self._lock:
            entries = [(key, entry.access_count, entry.timestamp) 
                      for key, entry in self._cache.items()]
            
            # Sort by access count descending
            entries.sort(key=lambda x: x[1], reverse=True)
            
            return entries[:n]
    
    def optimize(self):
        """Optimize cache by removing expired entries and defragmenting."""
        with self._lock:
            initial_size = len(self._cache)
            initial_memory = self._current_memory
            
            # Remove expired entries
            self._evict_expired()
            
            # Rebuild cache to optimize memory layout
            old_cache = self._cache
            self._cache = OrderedDict()
            self._current_memory = 0
            
            for key, entry in old_cache.items():
                if not self._is_expired(entry):
                    self._cache[key] = entry
                    self._current_memory += entry.size_bytes
            
            removed_entries = initial_size - len(self._cache)
            memory_freed = initial_memory - self._current_memory
            
            logger.info(f"Cache optimized: removed {removed_entries} entries, "
                       f"freed {memory_freed / (1024*1024):.1f}MB")
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.enable_persistence:
            return
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(dict(self._cache), f)
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self._cache = OrderedDict(cache_data)
            self._current_memory = sum(entry.size_bytes for entry in self._cache.values())
            
            logger.info(f"Cache loaded from {self.cache_file}: {len(self._cache)} entries")
        except FileNotFoundError:
            logger.info("No existing cache file found")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def save(self):
        """Manually save cache to disk."""
        self._save_cache()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.enable_persistence:
            self._save_cache()


class DistributedCache:
    """
    Distributed cache implementation for multi-node DLLM inference.
    Coordinates cache across multiple workers/nodes.
    """
    
    def __init__(
        self,
        local_cache: TokenCache,
        node_id: str,
        coordinator_address: Optional[str] = None
    ):
        self.local_cache = local_cache
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        
        # TODO: Implement distributed coordination
        # This would integrate with systems like Redis, etcd, or custom coordination
        logger.info(f"Distributed cache initialized for node {node_id}")
    
    def get(self, key: Union[str, Tuple, List]) -> Optional[Any]:
        """Get from local cache first, then distributed."""
        # Check local cache first
        result = self.local_cache.get(key)
        if result is not None:
            return result
        
        # TODO: Check distributed cache
        # Implementation would depend on coordination system
        
        return None
    
    def put(self, key: Union[str, Tuple, List], value: Any) -> bool:
        """Put to local cache and optionally distribute."""
        # Store locally
        success = self.local_cache.put(key, value)
        
        # TODO: Optionally replicate to other nodes
        # Implementation would depend on coordination system
        
        return success