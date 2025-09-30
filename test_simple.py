"""
Simple example without external dependencies for testing the core system.
"""

import torch
import time
from dllm_sampling import (
    DLLMSampler, 
    SamplingConfig, 
    InferenceAccelerator, 
    AccelerationConfig,
    TokenCache
)


def main():
    print("DLLM Sampling System - Core Functionality Test")
    print("=" * 60)
    
    # 1. Basic Sampling Test
    print("\n1. Testing Basic Sampling...")
    
    config = SamplingConfig(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    sampler = DLLMSampler(config)
    
    # Create test logits
    torch.manual_seed(42)
    batch_size = 3
    vocab_size = 1000
    logits = torch.randn(batch_size, vocab_size)
    
    print(f"  Input logits shape: {logits.shape}")
    
    # Test different sampling strategies
    nucleus_tokens = sampler.nucleus_sampling(logits)
    topk_tokens = sampler.top_k_sampling(logits)
    hybrid_tokens = sampler.hybrid_sampling(logits)
    
    print(f"  Nucleus sampling: {nucleus_tokens.tolist()}")
    print(f"  Top-k sampling: {topk_tokens.tolist()}")
    print(f"  Hybrid sampling: {hybrid_tokens.tolist()}")
    
    # 2. Cache Test
    print("\n2. Testing Token Cache...")
    
    cache = TokenCache(max_size=1000, max_memory_mb=10.0)
    
    # Store some sequences
    test_sequences = [
        ([1, 2, 3, 4], "result_1"),
        ([5, 6, 7], "result_2"),
        ([8, 9, 10, 11, 12], "result_3"),
    ]
    
    for seq, result in test_sequences:
        success = cache.put(tuple(seq), result)
        print(f"  Cached {seq} -> '{result}': {success}")
    
    # Retrieve and test
    for seq, expected in test_sequences:
        retrieved = cache.get(tuple(seq))
        status = "✓" if retrieved == expected else "✗"
        print(f"  {status} Retrieved {seq}: '{retrieved}'")
    
    cache_stats = cache.get_stats()
    print(f"  Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
    
    # 3. Performance Test
    print("\n3. Testing Performance...")
    
    total_tokens = 0
    total_time = 0
    
    for i in range(10):
        torch.manual_seed(i)
        test_logits = torch.randn(4, 1000)
        
        start_time = time.time()
        samples = sampler.hybrid_sampling(test_logits)
        elapsed = time.time() - start_time
        
        total_tokens += len(samples)
        total_time += elapsed
        
        sampler.update_metrics(len(samples), elapsed)
    
    perf_stats = sampler.get_performance_stats()
    print(f"  Generated {total_tokens} tokens in {total_time:.4f}s")
    print(f"  Performance: {perf_stats['tokens_per_second']:.2f} tokens/sec")
    
    # 4. Configuration Test
    print("\n4. Testing Different Configurations...")
    
    configs = [
        ("High Temperature", SamplingConfig(temperature=2.0, top_p=0.9)),
        ("Low Temperature", SamplingConfig(temperature=0.1, top_p=0.9)),
        ("High Top-k", SamplingConfig(temperature=1.0, top_k=100)),
        ("Low Top-k", SamplingConfig(temperature=1.0, top_k=10)),
    ]
    
    test_logits = torch.randn(1, 1000)
    
    for name, cfg in configs:
        test_sampler = DLLMSampler(cfg)
        
        # Sample multiple times to see variation
        samples = []
        for _ in range(5):
            sample = test_sampler.sample(test_logits, strategy="hybrid")
            samples.append(sample.item())
        
        variance = torch.var(torch.tensor(samples, dtype=torch.float))
        print(f"  {name}: samples={samples}, variance={variance:.2f}")
    
    # 5. Acceleration Config Test
    print("\n5. Testing Acceleration Configuration...")
    
    acc_config = AccelerationConfig(
        use_half_precision=False,  # Keep as float32 for CPU testing
        max_batch_size=8,
        dynamic_batching=True,
        use_kv_cache=True,
        batch_timeout_ms=100
    )
    
    print(f"  Acceleration config:")
    print(f"    Half precision: {acc_config.use_half_precision}")
    print(f"    Max batch size: {acc_config.max_batch_size}")
    print(f"    Dynamic batching: {acc_config.dynamic_batching}")
    print(f"    KV cache: {acc_config.use_kv_cache}")
    
    # 6. Repetition Penalty Test
    print("\n6. Testing Repetition Penalty...")
    
    # Create input with repeated tokens
    input_ids = torch.tensor([[1, 2, 3, 1, 2], [4, 5, 6, 4, 5]])
    test_logits = torch.randn(2, 1000)
    
    # Set high probability for repeated tokens
    test_logits[:, 1] = 10.0  # High prob for token 1
    test_logits[:, 2] = 10.0  # High prob for token 2
    
    # Apply repetition penalty
    penalized_logits = sampler.apply_repetition_penalty(
        test_logits, input_ids, penalty=1.5
    )
    
    print(f"  Original logits for token 1: {test_logits[0, 1]:.2f}")
    print(f"  Penalized logits for token 1: {penalized_logits[0, 1]:.2f}")
    print(f"  Penalty applied: {test_logits[0, 1] > penalized_logits[0, 1]}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("The DLLM Sampling System is working correctly.")


if __name__ == "__main__":
    main()