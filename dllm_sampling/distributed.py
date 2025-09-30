"""
Distributed inference system for DLLM models across multiple nodes/GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import socket
import threading
from dataclasses import dataclass
from loguru import logger
import queue
import json

from .core import SamplingConfig, DLLMSampler
from .accelerator import InferenceAccelerator, AccelerationConfig
from .cache import TokenCache, DistributedCache


@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    
    # Distributed setup
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    
    # Node configuration
    master_addr: str = "localhost"
    master_port: int = 29500
    local_rank: int = 0
    
    # Load balancing
    load_balancing_strategy: str = "round_robin"  # round_robin, least_loaded, hash_based
    request_timeout: float = 30.0
    
    # Communication
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    heartbeat_interval: float = 5.0
    
    # Model parallelism
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Memory management
    offload_activations: bool = False
    use_zero_optimizer: bool = False


class WorkerNode:
    """Individual worker node in the distributed system."""
    
    def __init__(
        self,
        rank: int,
        config: DistributedConfig,
        model: torch.nn.Module,
        accelerator: InferenceAccelerator
    ):
        self.rank = rank
        self.config = config
        self.model = model
        self.accelerator = accelerator
        
        self.is_running = False
        self.request_queue = queue.Queue()
        self.result_cache = {}
        
        # Performance tracking
        self.processed_requests = 0
        self.total_processing_time = 0.0
        self.current_load = 0
        
        logger.info(f"Worker node {rank} initialized")
    
    def start(self):
        """Start the worker node."""
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_requests)
        self.processing_thread.start()
        
        logger.info(f"Worker node {self.rank} started")
    
    def stop(self):
        """Stop the worker node."""
        self.is_running = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        logger.info(f"Worker node {self.rank} stopped")
    
    def _process_requests(self):
        """Main request processing loop."""
        while self.is_running:
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=1.0)
                
                if request is None:  # Poison pill
                    break
                
                self.current_load += 1
                start_time = time.time()
                
                # Process request
                result = self._handle_request(request)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.processed_requests += 1
                self.total_processing_time += processing_time
                self.current_load -= 1
                
                # Store result
                self.result_cache[request["request_id"]] = result
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.current_load = max(0, self.current_load - 1)
    
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single inference request."""
        try:
            request_type = request.get("type", "generate")
            
            if request_type == "generate":
                return self._handle_generate_request(request)
            elif request_type == "batch":
                return self._handle_batch_request(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _handle_generate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text generation request."""
        input_ids = request["input_ids"]
        generation_params = request.get("generation_params", {})
        
        result = self.accelerator.generate(
            input_ids=input_ids,
            **generation_params
        )
        
        result["worker_rank"] = self.rank
        result["request_id"] = request["request_id"]
        
        return result
    
    def _handle_batch_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch inference request."""
        batch_requests = request["requests"]
        
        # Add to accelerator batch queue
        for req in batch_requests:
            self.accelerator.add_to_batch(req)
        
        # Process the batch
        results = self.accelerator.process_current_batch()
        
        return {
            "results": results,
            "worker_rank": self.rank,
            "request_id": request["request_id"]
        }
    
    def add_request(self, request: Dict[str, Any]) -> bool:
        """Add request to processing queue."""
        try:
            self.request_queue.put(request, timeout=1.0)
            return True
        except queue.Full:
            return False
    
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a request ID."""
        return self.result_cache.pop(request_id, None)
    
    def get_load_info(self) -> Dict[str, Any]:
        """Get current load information."""
        avg_processing_time = 0.0
        if self.processed_requests > 0:
            avg_processing_time = self.total_processing_time / self.processed_requests
        
        return {
            "rank": self.rank,
            "current_load": self.current_load,
            "queue_size": self.request_queue.qsize(),
            "processed_requests": self.processed_requests,
            "avg_processing_time": avg_processing_time,
            "memory_stats": self.accelerator.get_memory_stats()
        }


class LoadBalancer:
    """Load balancer for distributing requests across worker nodes."""
    
    def __init__(self, workers: List[WorkerNode], strategy: str = "round_robin"):
        self.workers = workers
        self.strategy = strategy
        self.current_worker_idx = 0
        
        logger.info(f"Load balancer initialized with {len(workers)} workers, strategy: {strategy}")
    
    def select_worker(self, request: Dict[str, Any]) -> Optional[WorkerNode]:
        """Select the best worker for the request."""
        if not self.workers:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection()
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection()
        elif self.strategy == "hash_based":
            return self._hash_based_selection(request)
        else:
            return self.workers[0]  # Fallback
    
    def _round_robin_selection(self) -> WorkerNode:
        """Round-robin worker selection."""
        worker = self.workers[self.current_worker_idx]
        self.current_worker_idx = (self.current_worker_idx + 1) % len(self.workers)
        return worker
    
    def _least_loaded_selection(self) -> WorkerNode:
        """Select worker with least current load."""
        return min(self.workers, key=lambda w: w.current_load + w.request_queue.qsize())
    
    def _hash_based_selection(self, request: Dict[str, Any]) -> WorkerNode:
        """Hash-based worker selection for consistent routing."""
        # Use input_ids hash for consistent routing
        input_ids = request.get("input_ids", [])
        hash_val = hash(tuple(input_ids))
        worker_idx = hash_val % len(self.workers)
        return self.workers[worker_idx]


class DistributedInference:
    """
    Main distributed inference coordinator.
    Manages multiple worker nodes and handles request routing.
    """
    
    def __init__(
        self,
        config: DistributedConfig,
        sampling_config: SamplingConfig,
        acceleration_config: AccelerationConfig
    ):
        self.config = config
        self.sampling_config = sampling_config
        self.acceleration_config = acceleration_config
        
        self.workers: List[WorkerNode] = []
        self.load_balancer: Optional[LoadBalancer] = None
        self.is_running = False
        
        # Request tracking
        self.pending_requests = {}
        self.request_counter = 0
        
        logger.info("Distributed inference system initialized")
    
    def initialize_distributed(self):
        """Initialize distributed training/inference setup."""
        if self.config.world_size > 1:
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            logger.info(f"Distributed process group initialized: rank {self.config.rank}/{self.config.world_size}")
    
    def setup_workers(self, model: torch.nn.Module):
        """Setup worker nodes."""
        # Create sampler
        sampler = DLLMSampler(self.sampling_config)
        
        # Create cache
        cache = TokenCache()
        
        # Create workers
        for i in range(self.config.world_size):
            # Clone model for each worker (in practice, might use model parallelism)
            worker_model = model  # Could implement model parallelism here
            
            # Create accelerator for this worker
            accelerator = InferenceAccelerator(
                model=worker_model,
                sampler=sampler,
                config=self.acceleration_config,
                cache=cache
            )
            
            # Create worker
            worker = WorkerNode(
                rank=i,
                config=self.config,
                model=worker_model,
                accelerator=accelerator
            )
            
            self.workers.append(worker)
        
        # Create load balancer
        self.load_balancer = LoadBalancer(self.workers, self.config.load_balancing_strategy)
        
        logger.info(f"Setup {len(self.workers)} worker nodes")
    
    def start(self):
        """Start the distributed inference system."""
        if not self.workers:
            raise RuntimeError("No workers configured. Call setup_workers() first.")
        
        self.is_running = True
        
        # Start all workers
        for worker in self.workers:
            worker.start()
        
        logger.info("Distributed inference system started")
    
    def stop(self):
        """Stop the distributed inference system."""
        self.is_running = False
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()
        
        logger.info("Distributed inference system stopped")
    
    def generate(
        self, 
        input_ids: Union[torch.Tensor, List[int]],
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate text using distributed inference.
        
        Args:
            input_ids: Input token sequence
            **generation_params: Generation parameters
            
        Returns:
            Generation result
        """
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        request = {
            "request_id": request_id,
            "type": "generate",
            "input_ids": input_ids,
            "generation_params": generation_params,
            "timestamp": time.time()
        }
        
        # Select worker
        worker = self.load_balancer.select_worker(request)
        if not worker:
            return {"error": "No available workers"}
        
        # Submit request
        if not worker.add_request(request):
            return {"error": "Worker queue full"}
        
        # Wait for result
        start_time = time.time()
        while time.time() - start_time < self.config.request_timeout:
            result = worker.get_result(request_id)
            if result is not None:
                return result
            
            time.sleep(0.01)  # Small delay
        
        return {"error": "Request timeout"}
    
    def generate_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple requests in parallel.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation results
        """
        # Distribute requests across workers
        worker_requests = {worker: [] for worker in self.workers}
        
        for i, request in enumerate(requests):
            request["request_id"] = f"batch_req_{self.request_counter}_{i}"
            worker = self.load_balancer.select_worker(request)
            worker_requests[worker].append(request)
        
        self.request_counter += 1
        
        # Submit batch requests
        submitted_requests = []
        for worker, worker_reqs in worker_requests.items():
            if worker_reqs:
                batch_request = {
                    "request_id": f"batch_{self.request_counter}",
                    "type": "batch",
                    "requests": worker_reqs,
                    "timestamp": time.time()
                }
                
                if worker.add_request(batch_request):
                    submitted_requests.append((worker, batch_request["request_id"]))
        
        # Collect results
        results = []
        start_time = time.time()
        
        while (len(results) < len(submitted_requests) and 
               time.time() - start_time < self.config.request_timeout):
            
            for worker, request_id in submitted_requests:
                result = worker.get_result(request_id)
                if result is not None and result not in results:
                    if "results" in result:
                        results.extend(result["results"])
                    else:
                        results.append(result)
            
            time.sleep(0.01)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        worker_stats = [worker.get_load_info() for worker in self.workers]
        
        total_requests = sum(w["processed_requests"] for w in worker_stats)
        total_queue_size = sum(w["queue_size"] for w in worker_stats)
        avg_load = sum(w["current_load"] for w in worker_stats) / len(worker_stats)
        
        return {
            "num_workers": len(self.workers),
            "total_processed_requests": total_requests,
            "total_queue_size": total_queue_size,
            "average_load": avg_load,
            "worker_stats": worker_stats,
            "config": {
                "world_size": self.config.world_size,
                "load_balancing_strategy": self.config.load_balancing_strategy,
                "backend": self.config.backend
            }
        }
    
    def optimize_workers(self):
        """Optimize worker performance and memory usage."""
        for worker in self.workers:
            worker.accelerator.optimize_memory()
            if hasattr(worker.accelerator, 'cache'):
                worker.accelerator.cache.optimize()
        
        logger.info("Worker optimization completed")


def launch_distributed_inference(
    rank: int,
    world_size: int,
    model_factory,
    config: DistributedConfig,
    sampling_config: SamplingConfig,
    acceleration_config: AccelerationConfig
):
    """
    Launch distributed inference process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model_factory: Function to create model instance
        config: Distributed configuration
        sampling_config: Sampling configuration
        acceleration_config: Acceleration configuration
    """
    # Update config
    config.rank = rank
    config.world_size = world_size
    
    # Create distributed inference system
    dist_inference = DistributedInference(config, sampling_config, acceleration_config)
    
    # Initialize distributed backend
    dist_inference.initialize_distributed()
    
    # Create model
    model = model_factory()
    
    # Setup workers
    dist_inference.setup_workers(model)
    
    # Start system
    dist_inference.start()
    
    try:
        # Keep process alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down distributed inference")
    
    finally:
        dist_inference.stop()