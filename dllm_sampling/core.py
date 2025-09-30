"""
Core sampling algorithms and configuration for DLLM inference acceleration.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies in DLLM inference."""
    
    # Basic sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Advanced sampling options
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    
    # Acceleration specific parameters
    batch_size: int = 1
    max_length: int = 512
    min_length: int = 1
    
    # Memory optimization
    use_cache: bool = True
    cache_size: int = 10000
    low_memory: bool = False
    
    # Distributed inference
    num_workers: int = 1
    device_map: Optional[Dict[str, int]] = None
    
    # Performance monitoring
    track_metrics: bool = True
    log_level: str = "INFO"


class DLLMSampler:
    """
    High-performance sampling system optimized for DLLM (Distributed Large Language Model) inference.
    Specifically designed for LLaDA and similar architectures.
    """
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_logging()
        
        # Performance metrics
        self.metrics = {
            "total_tokens": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        logger.info(f"Initialized DLLM Sampler on device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        import sys
        logger.remove()
        logger.add(sys.stderr, level=self.config.log_level)
    
    def nucleus_sampling(
        self, 
        logits: torch.Tensor, 
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Optimized nucleus (top-p) sampling for DLLM inference.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            
        Returns:
            Sampled token indices [batch_size]
        """
        top_p = top_p or self.config.top_p
        temperature = temperature or self.config.temperature
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for nucleus
        nucleus_mask = cumsum_probs <= top_p
        
        # Ensure at least one token is selected
        nucleus_mask[..., 0] = True
        
        # Zero out probabilities outside nucleus
        sorted_probs = sorted_probs * nucleus_mask.float()
        
        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Sample from filtered distribution
        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)
        
        # Map back to original indices
        sampled_indices = torch.gather(sorted_indices, -1, sampled_sorted_indices)
        
        return sampled_indices.squeeze(-1)
    
    def top_k_sampling(
        self, 
        logits: torch.Tensor, 
        top_k: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Optimized top-k sampling for DLLM inference.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            top_k: Number of top tokens to consider
            temperature: Sampling temperature
            
        Returns:
            Sampled token indices [batch_size]
        """
        top_k = top_k or self.config.top_k
        temperature = temperature or self.config.temperature
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # Convert to probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Sample from top-k distribution
        sampled_k_indices = torch.multinomial(top_k_probs, num_samples=1)
        
        # Map back to original vocabulary indices
        sampled_indices = torch.gather(top_k_indices, -1, sampled_k_indices)
        
        return sampled_indices.squeeze(-1)
    
    def hybrid_sampling(
        self, 
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Hybrid top-k + nucleus sampling for optimal quality and efficiency.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            top_k: Number of top tokens to consider first
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            
        Returns:
            Sampled token indices [batch_size]
        """
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p
        temperature = temperature or self.config.temperature
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # First apply top-k filtering
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
        
        # Then apply nucleus sampling on top-k results
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Sort top-k probabilities
        sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create nucleus mask
        nucleus_mask = cumsum_probs <= top_p
        nucleus_mask[..., 0] = True  # Ensure at least one token
        
        # Apply nucleus filtering
        sorted_probs = sorted_probs * nucleus_mask.float()
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Sample from filtered distribution
        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)
        
        # Map back to top-k indices
        sampled_k_indices = torch.gather(sorted_indices, -1, sampled_sorted_indices)
        
        # Map back to original vocabulary
        sampled_indices = torch.gather(top_k_indices, -1, sampled_k_indices)
        
        return sampled_indices.squeeze(-1)
    
    def apply_repetition_penalty(
        self, 
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply repetition penalty to reduce repeated tokens.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            input_ids: Previous token sequence [batch_size, seq_len]
            penalty: Repetition penalty factor
            
        Returns:
            Modified logits with repetition penalty applied
        """
        penalty = penalty or self.config.repetition_penalty
        
        if penalty == 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        
        # Create penalty mask for tokens that appeared in input
        for i in range(batch_size):
            unique_tokens = torch.unique(input_ids[i])
            for token in unique_tokens:
                if token < vocab_size:
                    if logits[i, token] > 0:
                        logits[i, token] /= penalty
                    else:
                        logits[i, token] *= penalty
        
        return logits
    
    def sample(
        self, 
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        strategy: str = "hybrid"
    ) -> torch.Tensor:
        """
        Main sampling function with multiple strategies.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            input_ids: Previous token sequence for repetition penalty
            strategy: Sampling strategy ("nucleus", "top_k", "hybrid")
            
        Returns:
            Sampled token indices [batch_size]
        """
        # Apply repetition penalty if input_ids provided
        if input_ids is not None:
            logits = self.apply_repetition_penalty(logits, input_ids)
        
        # Choose sampling strategy
        if strategy == "nucleus":
            return self.nucleus_sampling(logits)
        elif strategy == "top_k":
            return self.top_k_sampling(logits)
        elif strategy == "hybrid":
            return self.hybrid_sampling(logits)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def update_metrics(self, tokens_generated: int, time_elapsed: float):
        """Update performance metrics."""
        self.metrics["total_tokens"] += tokens_generated
        self.metrics["total_time"] += time_elapsed
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if self.metrics["total_time"] > 0:
            tokens_per_second = self.metrics["total_tokens"] / self.metrics["total_time"]
        else:
            tokens_per_second = 0.0
        
        cache_hit_rate = 0.0
        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total_cache_requests
        
        return {
            "tokens_per_second": tokens_per_second,
            "total_tokens": self.metrics["total_tokens"],
            "total_time": self.metrics["total_time"],
            "cache_hit_rate": cache_hit_rate,
        }