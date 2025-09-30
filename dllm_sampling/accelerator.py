"""
Inference acceleration optimizations for DLLM models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import time
from dataclasses import dataclass
from loguru import logger
import gc

from .core import SamplingConfig, DLLMSampler
from .cache import TokenCache


@dataclass
class AccelerationConfig:
    """Configuration for inference acceleration techniques."""
    
    # Model optimization
    use_half_precision: bool = True
    use_flash_attention: bool = True
    enable_torch_compile: bool = False
    
    # Memory optimization
    gradient_checkpointing: bool = False
    offload_to_cpu: bool = False
    max_memory_per_gpu: Optional[float] = None
    
    # Batching optimization
    dynamic_batching: bool = True
    max_batch_size: int = 8
    batch_timeout_ms: int = 100
    
    # KV cache optimization
    use_kv_cache: bool = True
    kv_cache_max_length: int = 2048
    sliding_window_size: Optional[int] = None
    
    # Quantization
    use_int8: bool = False
    use_int4: bool = False
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = False


class InferenceAccelerator:
    """
    Comprehensive inference acceleration system for DLLM models.
    Implements various optimization techniques including memory management,
    batching, caching, and hardware-specific optimizations.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        sampler: DLLMSampler,
        config: AccelerationConfig,
        cache: Optional[TokenCache] = None
    ):
        self.model = model
        self.sampler = sampler
        self.config = config
        self.cache = cache or TokenCache()
        
        self.device = next(model.parameters()).device
        self.batch_queue = []
        self.last_batch_time = time.time()
        
        # Apply optimizations
        self._apply_model_optimizations()
        self._setup_profiling()
        
        logger.info("Inference accelerator initialized with optimizations")
    
    def _apply_model_optimizations(self):
        """Apply various model optimizations."""
        
        # Half precision
        if self.config.use_half_precision and self.device.type == "cuda":
            logger.info("Enabling half precision (FP16)")
            self.model = self.model.half()
        
        # Torch compile (PyTorch 2.0+)
        if self.config.enable_torch_compile:
            try:
                logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Gradient checkpointing for memory savings
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
        
        # Set model to eval mode
        self.model.eval()
    
    def _setup_profiling(self):
        """Setup profiling if enabled."""
        self.profiler = None
        if self.config.enable_profiling:
            from torch.profiler import profile, ProfilerActivity
            
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            self.profiler = profile(
                activities=activities,
                record_shapes=True,
                profile_memory=self.config.profile_memory,
                with_stack=True
            )
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if self.config.offload_to_cpu:
            # Move model to CPU when not in use
            self.model.cpu()
            logger.info("Model offloaded to CPU")
    
    def _prepare_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of requests for inference."""
        input_ids_list = []
        attention_mask_list = []
        max_length = 0
        
        for request in requests:
            input_ids = request["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            input_ids_list.append(input_ids)
            max_length = max(max_length, len(input_ids))
        
        # Pad sequences to same length
        batch_input_ids = []
        batch_attention_mask = []
        
        for input_ids in input_ids_list:
            # Pad sequence
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                padded_ids = torch.cat([
                    input_ids,
                    torch.zeros(pad_length, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    torch.ones(len(input_ids), dtype=torch.long),
                    torch.zeros(pad_length, dtype=torch.long)
                ])
            else:
                padded_ids = input_ids
                attention_mask = torch.ones(len(input_ids), dtype=torch.long)
            
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.stack(batch_input_ids).to(self.device),
            "attention_mask": torch.stack(batch_attention_mask).to(self.device)
        }
    
    def _should_process_batch(self) -> bool:
        """Determine if current batch should be processed."""
        if len(self.batch_queue) >= self.config.max_batch_size:
            return True
        
        if len(self.batch_queue) > 0:
            time_since_last = (time.time() - self.last_batch_time) * 1000
            if time_since_last >= self.config.batch_timeout_ms:
                return True
        
        return False
    
    def add_to_batch(self, request: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Add request to batch queue. Returns results if batch is ready to process.
        
        Args:
            request: Inference request with input_ids and generation parameters
            
        Returns:
            Batch results if batch is ready, None otherwise
        """
        self.batch_queue.append(request)
        
        if self._should_process_batch():
            return self.process_current_batch()
        
        return None
    
    def process_current_batch(self) -> List[Dict[str, Any]]:
        """Process the current batch queue."""
        if not self.batch_queue:
            return []
        
        batch_requests = self.batch_queue.copy()
        self.batch_queue.clear()
        self.last_batch_time = time.time()
        
        return self._process_batch(batch_requests)
    
    def _process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inference requests."""
        start_time = time.time()
        
        try:
            # Prepare batch
            batch_data = self._prepare_batch(requests)
            batch_size = len(requests)
            
            # Check cache for any complete matches
            cache_results = []
            uncached_indices = []
            uncached_requests = []
            
            for i, request in enumerate(requests):
                cache_key = tuple(request["input_ids"].tolist() if torch.is_tensor(request["input_ids"]) else request["input_ids"])
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    cache_results.append((i, cached_result))
                    self.sampler.metrics["cache_hits"] += 1
                else:
                    uncached_indices.append(i)
                    uncached_requests.append(request)
                    self.sampler.metrics["cache_misses"] += 1
            
            results = [None] * batch_size
            
            # Fill in cached results
            for i, cached_result in cache_results:
                results[i] = cached_result
            
            # Process uncached requests
            if uncached_requests:
                uncached_batch = self._prepare_batch(uncached_requests)
                uncached_results = self._forward_pass(uncached_batch, uncached_requests)
                
                # Fill in uncached results and update cache
                for idx, result in zip(uncached_indices, uncached_results):
                    results[idx] = result
                    
                    # Cache the result
                    original_request = requests[idx]
                    cache_key = tuple(original_request["input_ids"].tolist() if torch.is_tensor(original_request["input_ids"]) else original_request["input_ids"])
                    self.cache.put(cache_key, result)
            
            # Update metrics
            elapsed_time = time.time() - start_time
            total_tokens = sum(len(r.get("generated_tokens", [])) for r in results if r)
            self.sampler.update_metrics(total_tokens, elapsed_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [{"error": str(e)} for _ in requests]
    
    def _forward_pass(
        self, 
        batch_data: Dict[str, torch.Tensor], 
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform forward pass through the model."""
        
        if self.profiler:
            self.profiler.start()
        
        try:
            with torch.no_grad():
                # Move model back to GPU if offloaded
                if self.config.offload_to_cpu and self.model.device.type == "cpu":
                    self.model.to(self.device)
                
                # Forward pass
                outputs = self.model(**batch_data, use_cache=self.config.use_kv_cache)
                logits = outputs.logits[:, -1, :]  # Get last token logits
                
                # Sample next tokens
                sampled_tokens = self.sampler.sample(
                    logits, 
                    batch_data["input_ids"],
                    strategy="hybrid"
                )
                
                # Prepare results
                results = []
                for i, request in enumerate(requests):
                    result = {
                        "generated_tokens": [sampled_tokens[i].item()],
                        "logits": logits[i].cpu() if request.get("return_logits", False) else None,
                        "attention_weights": None,  # Could add attention weights if needed
                    }
                    results.append(result)
                
                return results
                
        finally:
            if self.profiler:
                self.profiler.stop()
            
            # Offload model if configured
            if self.config.offload_to_cpu:
                self.model.cpu()
    
    def generate(
        self, 
        input_ids: Union[torch.Tensor, List[int]],
        max_length: int = 50,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the accelerated inference pipeline.
        
        Args:
            input_ids: Input token sequence
            max_length: Maximum generation length
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generation result with tokens and metadata
        """
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        generated_tokens = []
        current_ids = input_ids.clone()
        
        start_time = time.time()
        
        for step in range(max_length):
            # Create request
            request = {
                "input_ids": current_ids,
                "step": step,
                **generation_kwargs
            }
            
            # Process through batch system (even for single request)
            batch_results = self._process_batch([request])
            
            if not batch_results or "error" in batch_results[0]:
                break
            
            # Get generated token
            next_token = batch_results[0]["generated_tokens"][0]
            generated_tokens.append(next_token)
            
            # Update current sequence
            current_ids = torch.cat([current_ids, torch.tensor([next_token], dtype=torch.long)])
            
            # Check for stop conditions
            if generation_kwargs.get("eos_token_id") and next_token == generation_kwargs["eos_token_id"]:
                break
        
        generation_time = time.time() - start_time
        
        return {
            "generated_tokens": generated_tokens,
            "full_sequence": current_ids.tolist(),
            "generation_time": generation_time,
            "tokens_per_second": len(generated_tokens) / generation_time if generation_time > 0 else 0,
            "steps": len(generated_tokens)
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {"cpu_memory_mb": 0, "gpu_memory_mb": 0}
        
        try:
            import psutil
            process = psutil.Process()
            stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            stats["gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return stats
    
    def export_profile(self, filename: str = "profile_trace.json"):
        """Export profiling results."""
        if self.profiler:
            self.profiler.export_chrome_trace(filename)
            logger.info(f"Profile exported to {filename}")
    
    def cleanup(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(self, 'profiler') and self.profiler:
            self.profiler.stop()
        
        self.batch_queue.clear()
        logger.info("Accelerator cleanup completed")