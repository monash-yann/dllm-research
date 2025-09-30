#!/usr/bin/env python3
"""
Comprehensive benchmark script for DLLM sampling system.
Tests various configurations and generates performance reports.
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

# Import our sampling system
from dllm_sampling import (
    DLLMSampler, 
    SamplingConfig, 
    InferenceAccelerator, 
    AccelerationConfig,
    TokenCache,
    DistributedInference,
    DistributedConfig
)


class DummyModel(torch.nn.Module):
    """Dummy model for benchmarking without requiring actual LLM."""
    
    def __init__(self, vocab_size=50000, hidden_size=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_layer = torch.nn.Linear(hidden_size, vocab_size)
        
        # Initialize with reasonable weights
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        
        # Simulate transformer output
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size, 
                                   device=input_ids.device, dtype=self.output_layer.weight.dtype)
        
        # Add some position dependency
        for i in range(seq_len):
            hidden_states[:, i, :] += i * 0.01
        
        logits = self.output_layer(hidden_states)
        
        # Return object similar to HuggingFace models
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits)


def create_test_requests(num_requests: int, seq_lengths: List[int]) -> List[Dict]:
    """Create test requests with varying sequence lengths."""
    requests = []
    
    for i in range(num_requests):
        seq_len = np.random.choice(seq_lengths)
        input_ids = torch.randint(1, 1000, (seq_len,))
        
        requests.append({
            "input_ids": input_ids,
            "max_length": np.random.randint(10, 100),
            "request_id": f"req_{i}"
        })
    
    return requests


def benchmark_sampling_strategies(config: Dict) -> Dict[str, Any]:
    """Benchmark different sampling strategies."""
    print("Benchmarking sampling strategies...")
    
    results = {}
    vocab_size = config.get("vocab_size", 50000)
    batch_size = config.get("batch_size", 8)
    num_iterations = config.get("num_iterations", 100)
    
    # Create test data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size)
    
    strategies = ["nucleus", "top_k", "hybrid"]
    
    for strategy in strategies:
        print(f"  Testing {strategy} sampling...")
        
        sampling_config = SamplingConfig(
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            batch_size=batch_size
        )
        
        sampler = DLLMSampler(sampling_config)
        
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            samples = sampler.sample(logits, strategy=strategy)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        results[strategy] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "tokens_per_second": batch_size / np.mean(times)
        }
    
    return results


def benchmark_cache_performance(config: Dict) -> Dict[str, Any]:
    """Benchmark cache performance with different configurations."""
    print("Benchmarking cache performance...")
    
    results = {}
    num_operations = config.get("num_operations", 10000)
    
    cache_configs = [
        {"max_size": 1000, "max_memory_mb": 10},
        {"max_size": 10000, "max_memory_mb": 100},
        {"max_size": 100000, "max_memory_mb": 1000},
    ]
    
    for i, cache_config in enumerate(cache_configs):
        print(f"  Testing cache config {i+1}...")
        
        cache = TokenCache(**cache_config)
        
        # Generate test data
        test_keys = []
        test_values = []
        for j in range(num_operations):
            key = tuple(np.random.randint(0, 1000, np.random.randint(5, 20)))
            value = f"result_{j}"
            test_keys.append(key)
            test_values.append(value)
        
        # Benchmark put operations
        start_time = time.time()
        for key, value in zip(test_keys, test_values):
            cache.put(key, value)
        put_time = time.time() - start_time
        
        # Benchmark get operations
        start_time = time.time()
        hits = 0
        for key in test_keys:
            if cache.get(key) is not None:
                hits += 1
        get_time = time.time() - start_time
        
        cache_stats = cache.get_stats()
        
        results[f"config_{i+1}"] = {
            "cache_config": cache_config,
            "put_time": put_time,
            "get_time": get_time,
            "put_ops_per_second": num_operations / put_time,
            "get_ops_per_second": num_operations / get_time,
            "hit_rate": cache_stats["hit_rate"],
            "final_size": cache_stats["current_size"],
            "memory_usage_mb": cache_stats["current_memory_mb"]
        }
    
    return results


def benchmark_batch_processing(config: Dict) -> Dict[str, Any]:
    """Benchmark batch processing performance."""
    print("Benchmarking batch processing...")
    
    # Create dummy model
    model = DummyModel()
    
    # Configure system
    sampling_config = SamplingConfig(batch_size=config.get("max_batch_size", 8))
    acceleration_config = AccelerationConfig(
        use_half_precision=False,  # Keep FP32 for CPU
        max_batch_size=config.get("max_batch_size", 8),
        dynamic_batching=True,
        batch_timeout_ms=config.get("batch_timeout_ms", 100)
    )
    
    sampler = DLLMSampler(sampling_config)
    cache = TokenCache()
    accelerator = InferenceAccelerator(model, sampler, acceleration_config, cache)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > config.get("max_batch_size", 8):
            continue
            
        print(f"  Testing batch size {batch_size}...")
        
        # Create test requests
        requests = create_test_requests(batch_size, [10, 20, 30, 40])
        
        # Benchmark processing
        times = []
        for _ in range(10):  # Multiple runs
            start_time = time.time()
            
            # Process batch
            batch_results = accelerator._process_batch(requests)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        total_tokens = sum(len(req["input_ids"]) for req in requests)
        avg_time = np.mean(times)
        
        results[f"batch_size_{batch_size}"] = {
            "batch_size": batch_size,
            "avg_time": avg_time,
            "std_time": np.std(times),
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / avg_time,
            "requests_per_second": batch_size / avg_time
        }
    
    return results


def benchmark_memory_usage(config: Dict) -> Dict[str, Any]:
    """Benchmark memory usage with different configurations."""
    print("Benchmarking memory usage...")
    
    results = {}
    
    # Test different model sizes
    model_configs = [
        {"vocab_size": 10000, "hidden_size": 1024},
        {"vocab_size": 30000, "hidden_size": 2048},
        {"vocab_size": 50000, "hidden_size": 4096},
    ]
    
    for i, model_config in enumerate(model_configs):
        print(f"  Testing model config {i+1}...")
        
        # Create model
        model = DummyModel(**model_config)
        
        # Measure initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        # Configure acceleration
        acceleration_config = AccelerationConfig(
            use_half_precision=config.get("use_half_precision", False),
            max_batch_size=8
        )
        
        sampling_config = SamplingConfig()
        sampler = DLLMSampler(sampling_config)
        accelerator = InferenceAccelerator(model, sampler, acceleration_config)
        
        # Measure memory after setup
        if torch.cuda.is_available():
            setup_memory = torch.cuda.memory_allocated()
        else:
            setup_memory = 0
        
        # Run inference to measure peak memory
        test_input = torch.randint(0, model_config["vocab_size"], (4, 20))
        
        peak_memory = setup_memory
        for _ in range(10):
            result = accelerator._forward_pass(
                {"input_ids": test_input, "attention_mask": torch.ones_like(test_input)},
                [{"input_ids": test_input[0]}] * 4
            )
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                peak_memory = max(peak_memory, current_memory)
        
        memory_stats = accelerator.get_memory_stats()
        
        results[f"model_{i+1}"] = {
            "model_config": model_config,
            "initial_memory_mb": initial_memory / (1024 * 1024),
            "setup_memory_mb": setup_memory / (1024 * 1024),
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "memory_stats": memory_stats
        }
    
    return results


def run_comprehensive_benchmark(config_file: str = None) -> Dict[str, Any]:
    """Run comprehensive benchmark suite."""
    print("Starting comprehensive DLLM sampling system benchmark...")
    print("=" * 60)
    
    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "vocab_size": 50000,
            "batch_size": 8,
            "max_batch_size": 16,
            "num_iterations": 100,
            "num_operations": 10000,
            "use_half_precision": False,
            "batch_timeout_ms": 100
        }
    
    results = {
        "timestamp": time.time(),
        "config": config,
        "system_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    # Run benchmarks
    try:
        results["sampling_strategies"] = benchmark_sampling_strategies(config)
    except Exception as e:
        print(f"Sampling strategies benchmark failed: {e}")
        results["sampling_strategies"] = {"error": str(e)}
    
    try:
        results["cache_performance"] = benchmark_cache_performance(config)
    except Exception as e:
        print(f"Cache performance benchmark failed: {e}")
        results["cache_performance"] = {"error": str(e)}
    
    try:
        results["batch_processing"] = benchmark_batch_processing(config)
    except Exception as e:
        print(f"Batch processing benchmark failed: {e}")
        results["batch_processing"] = {"error": str(e)}
    
    try:
        results["memory_usage"] = benchmark_memory_usage(config)
    except Exception as e:
        print(f"Memory usage benchmark failed: {e}")
        results["memory_usage"] = {"error": str(e)}
    
    return results


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # System info
    print(f"PyTorch version: {results['system_info']['torch_version']}")
    print(f"CUDA available: {results['system_info']['cuda_available']}")
    
    # Sampling strategies
    if "sampling_strategies" in results and "error" not in results["sampling_strategies"]:
        print("\nSampling Strategies Performance:")
        for strategy, stats in results["sampling_strategies"].items():
            print(f"  {strategy.capitalize()}: {stats['tokens_per_second']:.1f} tokens/sec")
    
    # Cache performance
    if "cache_performance" in results and "error" not in results["cache_performance"]:
        print("\nCache Performance:")
        for config_name, stats in results["cache_performance"].items():
            if "error" not in stats:
                print(f"  {config_name}: {stats['get_ops_per_second']:.1f} get ops/sec, "
                      f"{stats['hit_rate']:.2f} hit rate")
    
    # Batch processing
    if "batch_processing" in results and "error" not in results["batch_processing"]:
        print("\nBatch Processing Performance:")
        for batch_name, stats in results["batch_processing"].items():
            if "error" not in stats:
                print(f"  {batch_name}: {stats['tokens_per_second']:.1f} tokens/sec, "
                      f"{stats['requests_per_second']:.1f} req/sec")
    
    # Memory usage
    if "memory_usage" in results and "error" not in results["memory_usage"]:
        print("\nMemory Usage:")
        for model_name, stats in results["memory_usage"].items():
            if "error" not in stats:
                print(f"  {model_name}: Peak {stats['peak_memory_mb']:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="DLLM Sampling System Benchmark")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file path")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_benchmark(args.config)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nBenchmark results saved to {args.output}")
    
    # Print summary
    if args.summary or True:  # Always print summary
        print_benchmark_summary(results)


if __name__ == "__main__":
    main()