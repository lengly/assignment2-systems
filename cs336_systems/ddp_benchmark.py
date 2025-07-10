#!/usr/bin/env python3
"""
Benchmark script for naive DDP implementation.
Measures training time and communication overhead for distributed data parallel training.
"""

import argparse
from time import sleep
import timeit
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import os
import sys
from typing import Dict, Any, List
import numpy as np

# Add the cs336-basics directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))
from cs336_basics.model import BasicsTransformerLM

# Set multiprocessing start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)

# # Disable NCCL P2P and shared memory for single-node testing
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_SHM_DISABLE"] = "1"

# 'naive'  'flatten'  'overlap_individual'  'overlap_bucketed'
DDP_TYPE_1 = 'naive'
DDP_TYPE_2 = 'flatten'
DDP_TYPE_3 = 'overlap_individual'
DDP_TYPE_4 = 'overlap_bucketed'

class Bucket:
    def __init__(self, module: nn.Module, bucket_size_mb: float, world_size: int):
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.bucket = []
        self.bucket_count = []
        self.param_to_bucket = {}
        self.bucket_ready_count = []
        self.world_size = world_size
        self.bucketize_parameters(module)
        self.handles = []
        self.bucket_handles = {}  # 存储异步 all_reduce 的 handles
    
    def post_accumulate_grad_hook(self, param):
        bucket_index = self.param_to_bucket[param]
        self.bucket_ready_count[bucket_index] += 1
        if self.bucket_ready_count[bucket_index] == self.bucket_count[bucket_index]:
            self.all_reduce_bucket(bucket_index)

    def bucketize_parameters(self, module: nn.Module):
        current_bucket = []
        cur_size = 0
        for param in reversed(list(module.parameters())):
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.post_accumulate_grad_hook)
                param_size = param.numel() * param.element_size()
                if cur_size + param_size > self.bucket_size_bytes:
                    self.bucket.append(current_bucket)
                    current_bucket = []
                    cur_size = 0
                current_bucket.append(param)
                cur_size += param_size
                self.param_to_bucket[param] = len(self.bucket)
        if cur_size > 0:
            self.bucket.append(current_bucket)
        self.bucket_count = [len(bucket) for bucket in self.bucket]
        self.bucket_ready_count = [0 for _ in range(len(self.bucket))]
        print("Num of buckets: ", len(self.bucket))
        print("Bucket sizes: ", [len(bucket) for bucket in self.bucket])

    def all_reduce_bucket(self, bucket_index: int):
        # print("All reduce bucket: ", bucket_index)
        grads = [p.grad for p in self.bucket[bucket_index]]
        flattened_grads = _flatten_dense_tensors(grads)
        handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
        self.bucket_handles[bucket_index] = {
            'handle': handle,
            'flattened_grads': flattened_grads,
            'grads': grads,
        }
        self.bucket_ready_count[bucket_index] = 0
    
    def finish_gradient_synchronization(self):
        for bucket_index in range(len(self.bucket)):
            self.wait_bucket(bucket_index)
    
    def wait_bucket(self, bucket_index: int):
        if bucket_index not in self.bucket_handles:
            return
        handle_info = self.bucket_handles[bucket_index]
        handle_info['handle'].wait()  # 等待异步操作完成
        unflattened_grads = _unflatten_dense_tensors(
            handle_info['flattened_grads'], 
            handle_info['grads']
        )
        for p, g in zip(self.bucket[bucket_index], unflattened_grads):
            p.grad = g / self.world_size
        del self.bucket_handles[bucket_index]


class NaiveDDP(nn.Module):
    """Naive Distributed Data Parallel implementation that all-reduces gradients after backward pass."""
    
    def __init__(self, module: nn.Module, bucket_size_mb: float = 20, ddp_type: str = DDP_TYPE_4):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for name, param in self.module.named_parameters():
            dist.broadcast(param.data, src=0)
        self.ddp_type = ddp_type
        if self.ddp_type == DDP_TYPE_3:
            for param in self.module.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self.post_accumulate_grad_hook)
            self.handles = []
        if self.ddp_type == DDP_TYPE_4:
            self.bucket = Bucket(self.module, bucket_size_mb, self.world_size)
    
    def post_accumulate_grad_hook(self, param):
        param.grad /= self.world_size
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if self.ddp_type == DDP_TYPE_4:
            self.bucket.finish_gradient_synchronization()
            
        if self.ddp_type == DDP_TYPE_3:
            for handle in self.handles:
                handle.wait()
            self.handles = []

        # ================================================================================
        # Naive DDP Implementation with Reducing the Number of Communication Calls
        # ================================================================================
        if self.ddp_type == DDP_TYPE_2:
            params = []
            param_list = []
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    params.append(param.grad)
                    param_list.append(param)
            flattened_params = _flatten_dense_tensors(params)
            dist.all_reduce(flattened_params, op=dist.ReduceOp.SUM)
            unflattened_params = _unflatten_dense_tensors(flattened_params, params)
            for param, unflattened_grad in zip(param_list, unflattened_params):
                param.grad = unflattened_grad / self.world_size

        # ================================================================================
        # Naive DDP Implementation
        # ================================================================================
        if self.ddp_type == DDP_TYPE_1:
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size



def setup_process_group(rank: int, world_size: int):
    """Setup distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set device for GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = 'nccl'  # Use NCCL for GPU
    else:
        device = 'cpu'
        backend = 'gloo'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group():
    """Cleanup distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def create_model(config: Dict[str, Any], device: str) -> BasicsTransformerLM:
    """Initialize a Transformer model with given hyperparameters."""
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta']
    )
    model = model.to(device)
    return model


def generate_random_batch(batch_size: int, seq_len: int, vocab_size: int, device: str) -> torch.Tensor:
    """Generate a random batch of token IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def benchmark_ddp_training(
    rank: int,
    world_size: int,
    config: Dict[str, Any],
    batch_size: int,
    seq_len: int,
    num_warmup: int,
    num_steps: int
) -> Dict[str, float]:
    """
    Benchmark DDP training performance.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        config: Model configuration
        batch_size: Total batch size across all processes
        seq_len: Sequence length
        num_warmup: Number of warm-up steps
        num_steps: Number of benchmark steps
    
    Returns:
        Dictionary with timing results
    """
    device = setup_process_group(rank, world_size)
    
    # Create model and wrap with DDP
    model = create_model(config, device)
    ddp_model = NaiveDDP(model)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    # Generate data for this rank
    local_batch_size = batch_size // world_size
    batch = generate_random_batch(local_batch_size, seq_len, config['vocab_size'], device)
    targets = torch.randint(0, config['vocab_size'], (local_batch_size, seq_len), device=device)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Warm-up phase
    print(f"Rank {rank}: Running {num_warmup} warm-up steps...")
    for _ in range(num_warmup):
        optimizer.zero_grad()
        logits = ddp_model(batch)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()
    
    # Benchmarking phase
    print(f"Rank {rank}: Running {num_steps} benchmark steps...")
    forward_times = []
    backward_times = []
    communication_times = []
    total_times = []
    
    for step in range(num_steps):
        # Forward pass timing
        torch.cuda.synchronize()
        start_forward = timeit.default_timer()
        
        optimizer.zero_grad()
        logits = ddp_model(batch)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        torch.cuda.synchronize()
        end_forward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)
        
        # Backward pass timing
        start_backward = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end_backward = timeit.default_timer()
        backward_times.append(end_backward - start_backward)
        
        # Communication timing
        start_comm = timeit.default_timer()
        ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        end_comm = timeit.default_timer()
        communication_times.append(end_comm - start_comm)
        
        # Optimizer step
        optimizer.step()
        
        # Total time for this step
        total_times.append(end_comm - start_forward)
        
        if (step + 1) % 10 == 0:
            print(f"Rank {rank}: Completed {step + 1}/{num_steps} steps")
    
    # Calculate statistics
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    communication_times = torch.tensor(communication_times)
    total_times = torch.tensor(total_times)
    
    results = {
        'forward_mean': forward_times.mean().item(),
        'forward_std': forward_times.std().item(),
        'backward_mean': backward_times.mean().item(),
        'backward_std': backward_times.std().item(),
        'communication_mean': communication_times.mean().item(),
        'communication_std': communication_times.std().item(),
        'total_mean': total_times.mean().item(),
        'total_std': total_times.std().item(),
        'total_time': total_times.sum().item(),
        'throughput': num_steps / total_times.sum().item() if total_times.sum().item() > 0 else 0
    }
    
    cleanup_process_group()
    return results


def benchmark_ddp_worker(rank: int, world_size: int, config: Dict[str, Any], 
                        batch_size: int, seq_len: int, num_warmup: int, num_steps: int,
                        results_queue: mp.Queue):
    """Worker function for DDP benchmarking that can be pickled."""
    try:
        results = benchmark_ddp_training(
            rank=rank,
            world_size=world_size,
            config=config,
            batch_size=batch_size,
            seq_len=seq_len,
            num_warmup=num_warmup,
            num_steps=num_steps
        )
        results_queue.put((rank, results))
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        results_queue.put((rank, None))


def benchmark_single_process(
    config: Dict[str, Any],
    batch_size: int,
    seq_len: int,
    num_warmup: int,
    num_steps: int
) -> Dict[str, float]:
    """
    Benchmark single process training for comparison.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Generate data
    batch = generate_random_batch(batch_size, seq_len, config['vocab_size'], device)
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Warm-up phase
    print(f"Single process: Running {num_warmup} warm-up steps...")
    for _ in range(num_warmup):
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    
    # Benchmarking phase
    print(f"Single process: Running {num_steps} benchmark steps...")
    forward_times = []
    backward_times = []
    total_times = []
    
    for step in range(num_steps):
        # Forward pass timing
        torch.cuda.synchronize()
        start_forward = timeit.default_timer()
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        torch.cuda.synchronize()
        end_forward = timeit.default_timer()
        forward_times.append(end_forward - start_forward)
        
        # Backward pass timing
        start_backward = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end_backward = timeit.default_timer()
        backward_times.append(end_backward - start_backward)
        
        # Optimizer step
        optimizer.step()
        
        # Total time for this step
        total_times.append(end_backward - start_forward)
        
        if (step + 1) % 10 == 0:
            print(f"Single process: Completed {step + 1}/{num_steps} steps")
    
    # Calculate statistics
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    total_times = torch.tensor(total_times)
    
    results = {
        'forward_mean': forward_times.mean().item(),
        'forward_std': forward_times.std().item(),
        'backward_mean': backward_times.mean().item(),
        'backward_std': backward_times.std().item(),
        'total_mean': total_times.mean().item(),
        'total_std': total_times.std().item(),
        'total_time': total_times.sum().item(),
        'throughput': num_steps / total_times.sum().item() if total_times.sum().item() > 0 else 0
    }
    
    return results


def print_results(
    single_results: Dict[str, float],
    ddp_results: Dict[str, float],
    config: Dict[str, Any],
    batch_size: int,
    world_size: int,
    model_type: str
):
    """Print benchmarking results in a formatted way."""
    print("\n" + "="*80)
    print("DDP BENCHMARKING RESULTS")
    print("="*80)
    
    # Model configuration
    print(f"Model Configuration ({model_type}):")
    print(f"  - Vocab size: {config['vocab_size']:,}")
    print(f"  - Context length: {config['context_length']:,}")
    print(f"  - d_model: {config['d_model']:,}")
    print(f"  - Num layers: {config['num_layers']:,}")
    print(f"  - Num heads: {config['num_heads']:,}")
    print(f"  - d_ff: {config['d_ff']:,}")
    print(f"  - Rope theta: {config['rope_theta']}")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  - Total batch size: {batch_size}")
    print(f"  - World size: {world_size}")
    print(f"  - Local batch size: {batch_size // world_size}")
    
    # Single process results
    print(f"\nSingle Process Results:")
    print(f"  - Forward pass: {single_results['forward_mean']*1000:.3f} ± {single_results['forward_std']*1000:.3f} ms")
    print(f"  - Backward pass: {single_results['backward_mean']*1000:.3f} ± {single_results['backward_std']*1000:.3f} ms")
    print(f"  - Total time per step: {single_results['total_mean']*1000:.3f} ± {single_results['total_std']*1000:.3f} ms")
    print(f"  - Throughput: {single_results['throughput']:.2f} steps/s")
    
    # DDP results
    print(f"\nDDP Results (2 GPUs):")
    print(f"  - Forward pass: {ddp_results['forward_mean']*1000:.3f} ± {ddp_results['forward_std']*1000:.3f} ms")
    print(f"  - Backward pass: {ddp_results['backward_mean']*1000:.3f} ± {ddp_results['backward_std']*1000:.3f} ms")
    print(f"  - Communication: {ddp_results['communication_mean']*1000:.3f} ± {ddp_results['communication_std']*1000:.3f} ms")
    print(f"  - Total time per step: {ddp_results['total_mean']*1000:.3f} ± {ddp_results['total_std']*1000:.3f} ms")
    print(f"  - Throughput: {ddp_results['throughput']:.2f} steps/s")
    
    # Communication overhead analysis
    comm_overhead = ddp_results['communication_mean'] / ddp_results['total_mean'] * 100
    speedup = single_results['total_mean'] / ddp_results['total_mean']
    efficiency = speedup / world_size * 100
    
    print(f"\nCommunication Overhead Analysis:")
    print(f"  - Communication time: {ddp_results['communication_mean']*1000:.3f} ms")
    print(f"  - Communication overhead: {comm_overhead:.2f}%")
    print(f"  - Speedup vs single GPU: {speedup:.2f}x")
    print(f"  - Scaling efficiency: {efficiency:.2f}%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark naive DDP implementation')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='gpt2', 
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT-2 model size to benchmark')
    
    # Benchmarking parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Total batch size across all processes')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of benchmark steps')
    parser.add_argument('--world_size', type=int, default=2, help='Number of processes/GPUs')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return
    
    # GPT-2 model configurations
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[args.model_type]
    
    # Create model configuration
    config = {
        'vocab_size': 50257,  # GPT-2 vocabulary size
        'context_length': args.seq_len,
        'd_model': config_args['n_embd'],
        'num_layers': config_args['n_layer'],
        'num_heads': config_args['n_head'],
        'd_ff': config_args['n_embd'] * 4,  # Standard GPT-2 FF dimension
        'rope_theta': 10000.0
    }
    
    print("Benchmarking Setup:")
    print(f"  - Model: {args.model_type} ({config['d_model']} dim, {config['num_layers']} layers)")
    print(f"  - Batch size: {args.batch_size} (total across {args.world_size} GPUs)")
    print(f"  - Sequence length: {args.seq_len}")
    print(f"  - Warm-up steps: {args.num_warmup}")
    print(f"  - Benchmark steps: {args.num_steps}")
    
    # Benchmark single process training
    print("\nBenchmarking single process training...")
    single_results = benchmark_single_process(
        config=config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_warmup=args.num_warmup,
        num_steps=args.num_steps
    )
    
    # Benchmark DDP training
    print("\nBenchmarking DDP training...")
    processes = []
    results_queue = mp.Queue()
    
    for rank in range(args.world_size):
        p = mp.Process(
            target=benchmark_ddp_worker,
            args=(rank, args.world_size, config, args.batch_size, args.seq_len, 
                  args.num_warmup, args.num_steps, results_queue)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Collect results from queue
    ddp_results_list = []
    for _ in range(args.world_size):
        rank, results = results_queue.get()
        if results is not None:
            ddp_results_list.append(results)
    
    if not ddp_results_list:
        print("Error: No valid results from DDP processes")
        return
    
    # Average results across all processes
    ddp_results = {}
    for key in ddp_results_list[0].keys():
        values = [results[key] for results in ddp_results_list]
        ddp_results[key] = np.mean(values)
    
    # Print results
    print_results(single_results, ddp_results, config, args.batch_size, args.world_size, args.model_type)


if __name__ == '__main__':
    main() 



# ================================================================================
# DDP BENCHMARKING RESULTS @ Nvidia RTX 4090*2 (24GB)
# ================================================================================
# Model Configuration (gpt2):
#   - Vocab size: 50,257
#   - Context length: 512
#   - d_model: 768
#   - Num layers: 12
#   - Num heads: 12
#   - d_ff: 3,072
#   - Rope theta: 10000.0

# Training Configuration:
#   - Total batch size: 8
#   - World size: 2
#   - Local batch size: 4

# ================================================================================
# Baseline: Single Process Training
# ================================================================================

# Single Process Results:
#   - Forward pass: 66.530 ± 0.307 ms
#   - Backward pass: 128.165 ± 0.284 ms
#   - Total time per step: 194.696 ± 0.500 ms
#   - Throughput: 5.14 steps/s

# ================================================================================
# Naive DDP Implementation
# ================================================================================

# DDP Results (2 GPUs):
#   - Forward pass: 35.646 ± 2.183 ms
#   - Backward pass: 65.135 ± 0.291 ms
#   - Communication: 159.884 ± 1.870 ms
#   - Total time per step: 260.667 ± 2.715 ms
#   - Throughput: 3.84 steps/s

# Communication Overhead Analysis:
#   - Communication time: 159.884 ms
#   - Communication overhead: 61.34%
#   - Speedup vs single GPU: 0.75x
#   - Scaling efficiency: 37.35%

# ================================================================================
# Naive DDP Implementation with Reducing the Number of Communication Calls
# ================================================================================

# DDP Results (2 GPUs):
#   - Forward pass: 39.623 ± 5.194 ms
#   - Backward pass: 65.417 ± 0.721 ms
#   - Communication: 145.125 ± 3.153 ms
#   - Total time per step: 250.167 ± 5.297 ms
#   - Throughput: 4.00 steps/s

# Communication Overhead Analysis:
#   - Communication time: 145.125 ms
#   - Communication overhead: 58.01%
#   - Speedup vs single GPU: 0.78x
#   - Scaling efficiency: 38.99%
