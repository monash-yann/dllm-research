"""
Command-line interface for DLLM sampling system.
"""

import click
import yaml
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from .core import SamplingConfig, DLLMSampler
from .accelerator import AccelerationConfig, InferenceAccelerator
from .distributed import DistributedConfig, DistributedInference
from .cache import TokenCache


@click.group()
@click.version_option()
def main():
    """DLLM Sampling System - Acceleration for DLLM like LLaDA"""
    pass


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--model-path', '-m', required=True, help='Path to model')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--device', default='auto', help='Device to use (cuda/cpu/auto)')
@click.option('--batch-size', default=8, help='Maximum batch size')
@click.option('--cache-size', default=10000, help='Cache size')
@click.option('--half-precision', is_flag=True, help='Use half precision')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def serve(config, model_path, host, port, workers, device, batch_size, cache_size, half_precision, verbose):
    """Start DLLM inference server."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level=log_level)
    
    # Load configuration
    if config:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
    else:
        config_data = {}
    
    # Override with CLI arguments
    config_data.update({
        'model_path': model_path,
        'host': host,
        'port': port,
        'workers': workers,
        'device': device,
        'batch_size': batch_size,
        'cache_size': cache_size,
        'half_precision': half_precision
    })
    
    # Create configurations
    sampling_config = SamplingConfig(**config_data.get('sampling', {}))
    acceleration_config = AccelerationConfig(
        use_half_precision=half_precision,
        max_batch_size=batch_size,
        **config_data.get('acceleration', {})
    )
    distributed_config = DistributedConfig(
        world_size=workers,
        **config_data.get('distributed', {})
    )
    
    # Start server
    try:
        _start_server(model_path, sampling_config, acceleration_config, distributed_config, host, port)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def _start_server(model_path, sampling_config, acceleration_config, distributed_config, host, port):
    """Start the inference server."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import flask
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if acceleration_config.use_half_precision else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Create sampling system
    sampler = DLLMSampler(sampling_config)
    cache = TokenCache(max_size=acceleration_config.max_batch_size * 100)
    accelerator = InferenceAccelerator(model, sampler, acceleration_config, cache)
    
    # Setup distributed inference if multiple workers
    if distributed_config.world_size > 1:
        dist_inference = DistributedInference(distributed_config, sampling_config, acceleration_config)
        dist_inference.setup_workers(model)
        dist_inference.start()
        inference_engine = dist_inference
    else:
        inference_engine = accelerator
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "timestamp": time.time()})
    
    @app.route('/generate', methods=['POST'])
    def generate():
        """Text generation endpoint."""
        try:
            data = request.json
            
            # Validate input
            if 'text' not in data and 'input_ids' not in data:
                return jsonify({"error": "Either 'text' or 'input_ids' required"}), 400
            
            # Tokenize if text provided
            if 'text' in data:
                inputs = tokenizer(data['text'], return_tensors="pt")
                input_ids = inputs.input_ids[0]
            else:
                input_ids = data['input_ids']
            
            # Generation parameters
            generation_params = {
                'max_length': data.get('max_length', 50),
                'temperature': data.get('temperature', sampling_config.temperature),
                'top_k': data.get('top_k', sampling_config.top_k),
                'top_p': data.get('top_p', sampling_config.top_p),
                'repetition_penalty': data.get('repetition_penalty', sampling_config.repetition_penalty)
            }
            
            # Generate
            if hasattr(inference_engine, 'generate'):
                result = inference_engine.generate(input_ids, **generation_params)
            else:
                # Fallback for distributed inference
                result = inference_engine.generate(input_ids, **generation_params)
            
            # Decode generated tokens
            if 'generated_tokens' in result:
                generated_text = tokenizer.decode(result['generated_tokens'], skip_special_tokens=True)
                result['generated_text'] = generated_text
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch_generate', methods=['POST'])
    def batch_generate():
        """Batch text generation endpoint."""
        try:
            data = request.json
            requests = data.get('requests', [])
            
            if not requests:
                return jsonify({"error": "No requests provided"}), 400
            
            # Process batch
            if hasattr(inference_engine, 'generate_batch'):
                results = inference_engine.generate_batch(requests)
            else:
                # Process individually for single worker
                results = []
                for req in requests:
                    if 'text' in req:
                        inputs = tokenizer(req['text'], return_tensors="pt")
                        input_ids = inputs.input_ids[0]
                    else:
                        input_ids = req['input_ids']
                    
                    result = inference_engine.generate(input_ids, **req.get('generation_params', {}))
                    results.append(result)
            
            return jsonify({"results": results})
            
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/stats', methods=['GET'])
    def stats():
        """Get system statistics."""
        try:
            if hasattr(inference_engine, 'get_system_stats'):
                stats = inference_engine.get_system_stats()
            else:
                stats = {
                    "performance": sampler.get_performance_stats(),
                    "memory": accelerator.get_memory_stats(),
                    "cache": cache.get_stats()
                }
            
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/optimize', methods=['POST'])
    def optimize():
        """Trigger system optimization."""
        try:
            if hasattr(inference_engine, 'optimize_workers'):
                inference_engine.optimize_workers()
            else:
                accelerator.optimize_memory()
                cache.optimize()
            
            return jsonify({"status": "optimization completed"})
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Start server
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


@main.command()
@click.option('--text', '-t', required=True, help='Text to generate from')
@click.option('--model-path', '-m', required=True, help='Path to model')
@click.option('--max-length', default=50, help='Maximum generation length')
@click.option('--temperature', default=1.0, help='Sampling temperature')
@click.option('--top-k', default=50, help='Top-k sampling')
@click.option('--top-p', default=0.9, help='Top-p (nucleus) sampling')
@click.option('--strategy', default='hybrid', help='Sampling strategy')
@click.option('--device', default='auto', help='Device to use')
@click.option('--half-precision', is_flag=True, help='Use half precision')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def generate(text, model_path, max_length, temperature, top_k, top_p, strategy, device, half_precision, verbose):
    """Generate text using DLLM sampling system."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level=log_level)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if half_precision else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Create configurations
        sampling_config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            log_level=log_level
        )
        
        acceleration_config = AccelerationConfig(
            use_half_precision=half_precision
        )
        
        # Create sampling system
        sampler = DLLMSampler(sampling_config)
        cache = TokenCache()
        accelerator = InferenceAccelerator(model, sampler, acceleration_config, cache)
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids[0]
        
        # Generate
        logger.info("Generating text...")
        start_time = time.time()
        
        result = accelerator.generate(
            input_ids=input_ids,
            max_length=max_length,
            strategy=strategy
        )
        
        generation_time = time.time() - start_time
        
        # Decode result
        generated_tokens = result.get('generated_tokens', [])
        if generated_tokens:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_text = text + generated_text
        else:
            full_text = text
        
        # Output results
        click.echo("\n" + "="*50)
        click.echo("GENERATED TEXT:")
        click.echo("="*50)
        click.echo(full_text)
        click.echo("\n" + "="*50)
        click.echo("STATISTICS:")
        click.echo("="*50)
        click.echo(f"Generated tokens: {len(generated_tokens)}")
        click.echo(f"Generation time: {generation_time:.2f}s")
        click.echo(f"Tokens/second: {result.get('tokens_per_second', 0):.2f}")
        click.echo(f"Performance stats: {sampler.get_performance_stats()}")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


@main.command()
@click.option('--model-path', '-m', required=True, help='Path to model')
@click.option('--num-requests', default=100, help='Number of requests to benchmark')
@click.option('--batch-size', default=8, help='Batch size')
@click.option('--max-length', default=50, help='Maximum generation length')
@click.option('--workers', default=1, help='Number of workers')
@click.option('--device', default='auto', help='Device to use')
@click.option('--output', '-o', help='Output file for results')
def benchmark(model_path, num_requests, batch_size, max_length, workers, device, output):
    """Benchmark DLLM sampling system performance."""
    
    logger.info("Starting benchmark...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import random
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Create configurations
        sampling_config = SamplingConfig(batch_size=batch_size)
        acceleration_config = AccelerationConfig(
            use_half_precision=True,
            max_batch_size=batch_size,
            dynamic_batching=True
        )
        distributed_config = DistributedConfig(world_size=workers)
        
        # Create system
        sampler = DLLMSampler(sampling_config)
        cache = TokenCache()
        accelerator = InferenceAccelerator(model, sampler, acceleration_config, cache)
        
        if workers > 1:
            dist_inference = DistributedInference(distributed_config, sampling_config, acceleration_config)
            dist_inference.setup_workers(model)
            dist_inference.start()
            inference_engine = dist_inference
        else:
            inference_engine = accelerator
        
        # Generate test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The quick brown fox jumps over",
            "In the beginning was the word",
            "Technology has revolutionized the way we"
        ]
        
        # Run benchmark
        results = []
        total_start_time = time.time()
        
        for i in range(num_requests):
            prompt = random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids[0]
            
            start_time = time.time()
            
            if hasattr(inference_engine, 'generate'):
                result = inference_engine.generate(
                    input_ids=input_ids,
                    max_length=max_length
                )
            else:
                result = {"error": "Generation failed"}
            
            end_time = time.time()
            
            results.append({
                "request_id": i,
                "prompt": prompt,
                "generation_time": end_time - start_time,
                "tokens_generated": len(result.get('generated_tokens', [])),
                "success": "error" not in result
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_requests} requests")
        
        total_time = time.time() - total_start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        total_tokens = sum(r["tokens_generated"] for r in successful_requests)
        avg_latency = sum(r["generation_time"] for r in successful_requests) / len(successful_requests)
        throughput = len(successful_requests) / total_time
        tokens_per_second = total_tokens / total_time
        
        # Benchmark results
        benchmark_stats = {
            "total_requests": num_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": num_requests - len(successful_requests),
            "total_time": total_time,
            "total_tokens": total_tokens,
            "average_latency": avg_latency,
            "throughput_requests_per_second": throughput,
            "throughput_tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
            "workers": workers,
            "model_path": model_path,
            "results": results
        }
        
        # Output results
        click.echo("\n" + "="*50)
        click.echo("BENCHMARK RESULTS:")
        click.echo("="*50)
        click.echo(f"Total requests: {num_requests}")
        click.echo(f"Successful requests: {len(successful_requests)}")
        click.echo(f"Total time: {total_time:.2f}s")
        click.echo(f"Average latency: {avg_latency:.3f}s")
        click.echo(f"Throughput: {throughput:.2f} requests/second")
        click.echo(f"Token throughput: {tokens_per_second:.2f} tokens/second")
        click.echo(f"Total tokens generated: {total_tokens}")
        
        # Save to file if specified
        if output:
            with open(output, 'w') as f:
                json.dump(benchmark_stats, f, indent=2)
            logger.info(f"Benchmark results saved to {output}")
        
        # Cleanup
        if workers > 1:
            dist_inference.stop()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


@main.command()
@click.option('--config-template', default='config_template.yaml', help='Output config template file')
def init_config(config_template):
    """Initialize configuration template."""
    
    config = {
        'sampling': {
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.0,
            'batch_size': 8,
            'max_length': 512,
            'use_cache': True,
            'cache_size': 10000
        },
        'acceleration': {
            'use_half_precision': True,
            'use_flash_attention': True,
            'enable_torch_compile': False,
            'dynamic_batching': True,
            'max_batch_size': 8,
            'use_kv_cache': True,
            'kv_cache_max_length': 2048
        },
        'distributed': {
            'world_size': 1,
            'backend': 'nccl',
            'load_balancing_strategy': 'round_robin',
            'request_timeout': 30.0
        },
        'server': {
            'host': 'localhost',
            'port': 8000,
            'workers': 1
        }
    }
    
    with open(config_template, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration template created: {config_template}")
    click.echo("Edit the template and use it with --config option")


if __name__ == '__main__':
    main()