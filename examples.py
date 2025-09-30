"""
Example usage of DLLM sampling system.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dllm_sampling import (
    DLLMSampler, 
    SamplingConfig, 
    InferenceAccelerator, 
    AccelerationConfig,
    TokenCache
)


def basic_sampling_example():
    """Basic sampling example without model."""
    print("=== Basic Sampling Example ===")
    
    # Create configuration
    config = SamplingConfig(
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # Create sampler
    sampler = DLLMSampler(config)
    
    # Create fake logits (in real usage, these come from your model)
    torch.manual_seed(42)
    batch_size = 2
    vocab_size = 1000
    logits = torch.randn(batch_size, vocab_size)
    
    # Sample using different strategies
    print("Nucleus sampling:")
    nucleus_tokens = sampler.nucleus_sampling(logits)
    print(f"Sampled tokens: {nucleus_tokens.tolist()}")
    
    print("\nTop-k sampling:")
    topk_tokens = sampler.top_k_sampling(logits)
    print(f"Sampled tokens: {topk_tokens.tolist()}")
    
    print("\nHybrid sampling:")
    hybrid_tokens = sampler.hybrid_sampling(logits)
    print(f"Sampled tokens: {hybrid_tokens.tolist()}")
    
    # Performance stats
    sampler.update_metrics(tokens_generated=6, time_elapsed=0.1)
    stats = sampler.get_performance_stats()
    print(f"\nPerformance stats: {stats}")


def cache_example():
    """Token cache usage example."""
    print("\n=== Cache Example ===")
    
    # Create cache
    cache = TokenCache(max_size=1000, max_memory_mb=10.0)
    
    # Store some token sequences
    sequences = [
        ([1, 2, 3, 4], "output_1"),
        ([5, 6, 7, 8], "output_2"),
        ([1, 2, 3, 4], "output_1_duplicate"),  # Duplicate key
        ([9, 10, 11, 12], "output_3")
    ]
    
    for seq, output in sequences:
        success = cache.put(tuple(seq), output)
        print(f"Cached {seq} -> {output}: {success}")
    
    # Retrieve cached results
    test_sequences = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [99, 100]  # Not in cache
    ]
    
    for seq in test_sequences:
        result = cache.get(tuple(seq))
        print(f"Get {seq}: {result}")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")


def accelerator_example():
    """Inference accelerator example with dummy model."""
    print("\n=== Accelerator Example ===")
    
    # Create a simple dummy model for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.linear = torch.nn.Linear(512, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            # Dummy forward pass
            batch_size, seq_len = input_ids.shape
            hidden = torch.randn(batch_size, seq_len, 512)
            logits = self.linear(hidden)
            
            # Return structure similar to HuggingFace models
            class Output:
                def __init__(self, logits):
                    self.logits = logits
            
            return Output(logits)
    
    # Create model and configurations
    model = DummyModel()
    
    sampling_config = SamplingConfig(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        batch_size=4
    )
    
    acceleration_config = AccelerationConfig(
        use_half_precision=False,  # Keep as float32 for CPU
        max_batch_size=4,
        dynamic_batching=True,
        use_kv_cache=True
    )
    
    # Create accelerator
    sampler = DLLMSampler(sampling_config)
    cache = TokenCache()
    accelerator = InferenceAccelerator(model, sampler, acceleration_config, cache)
    
    # Generate text
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    
    print("Generating text...")
    result = accelerator.generate(
        input_ids=input_ids,
        max_length=10,
        eos_token_id=0  # Dummy EOS token
    )
    
    print(f"Generation result: {result}")
    
    # Show memory stats
    memory_stats = accelerator.get_memory_stats()
    print(f"Memory stats: {memory_stats}")


def batch_processing_example():
    """Batch processing example."""
    print("\n=== Batch Processing Example ===")
    
    # Create dummy model (same as above)
    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.linear = torch.nn.Linear(512, vocab_size)
            
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            batch_size, seq_len = input_ids.shape
            hidden = torch.randn(batch_size, seq_len, 512)
            logits = self.linear(hidden)
            
            class Output:
                def __init__(self, logits):
                    self.logits = logits
            
            return Output(logits)
    
    model = DummyModel()
    
    # Create accelerator
    sampling_config = SamplingConfig(batch_size=8)
    acceleration_config = AccelerationConfig(
        max_batch_size=8,
        dynamic_batching=True,
        batch_timeout_ms=50
    )
    
    sampler = DLLMSampler(sampling_config)
    accelerator = InferenceAccelerator(model, sampler, acceleration_config)
    
    # Create batch requests
    requests = [
        {"input_ids": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6, 7])},
        {"input_ids": torch.tensor([8, 9])},
        {"input_ids": torch.tensor([10, 11, 12, 13, 14])}
    ]
    
    print(f"Processing batch of {len(requests)} requests...")
    
    # Add requests to batch queue
    for i, request in enumerate(requests):
        result = accelerator.add_to_batch(request)
        if result:
            print(f"Batch processed after request {i+1}")
            print(f"Results: {len(result)} items")
    
    # Process any remaining requests
    remaining_results = accelerator.process_current_batch()
    if remaining_results:
        print(f"Processed remaining {len(remaining_results)} requests")


def performance_monitoring_example():
    """Performance monitoring example."""
    print("\n=== Performance Monitoring Example ===")
    
    # Create sampler with metrics tracking
    config = SamplingConfig(track_metrics=True)
    sampler = DLLMSampler(config)
    
    # Simulate some sampling operations
    torch.manual_seed(42)
    
    for i in range(5):
        logits = torch.randn(2, 1000)
        
        start_time = time.time()
        tokens = sampler.sample(logits, strategy="hybrid")
        elapsed = time.time() - start_time
        
        sampler.update_metrics(tokens_generated=2, time_elapsed=elapsed)
        
        print(f"Iteration {i+1}: Generated tokens {tokens.tolist()} in {elapsed:.4f}s")
    
    # Show final performance stats
    final_stats = sampler.get_performance_stats()
    print(f"\nFinal performance stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


def configuration_example():
    """Configuration management example."""
    print("\n=== Configuration Example ===")
    
    # Different configuration scenarios
    configs = {
        "fast_sampling": SamplingConfig(
            temperature=1.0,
            top_k=20,
            top_p=0.8,
            batch_size=16,
            use_cache=True,
            cache_size=5000
        ),
        
        "quality_sampling": SamplingConfig(
            temperature=0.7,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.2,
            batch_size=4,
            use_cache=True,
            cache_size=10000
        ),
        
        "creative_sampling": SamplingConfig(
            temperature=1.2,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            batch_size=8
        )
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-k: {config.top_k}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Cache enabled: {config.use_cache}")
        
        # Create sampler with this config
        sampler = DLLMSampler(config)
        
        # Test sampling
        logits = torch.randn(1, 1000)
        sample = sampler.sample(logits)
        print(f"  Sample token: {sample.item()}")


if __name__ == "__main__":
    import time
    
    print("DLLM Sampling System Examples")
    print("=" * 50)
    
    # Run all examples
    basic_sampling_example()
    cache_example()
    accelerator_example()
    batch_processing_example()
    performance_monitoring_example()
    configuration_example()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")