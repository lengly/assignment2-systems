#!/usr/bin/env python3
"""
Benchmark script for naive DDP implementation.
Measures training time and communication overhead for distributed data parallel training.
"""

import argparse
import timeit
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import sys
from typing import Dict, Any, List
import numpy as np

# Add the cs336-basics directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cs336-basics'))
from cs336_basics.model import BasicsTransformerLM

# Set multiprocessing start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Disable NCCL P2P and shared memory for single-node testing
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"


class NaiveDDP(nn.Module):
    """Naive Distributed Data Parallel implementation that all-reduces gradients after backward pass."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def all_reduce_gradients(self):
        """All-reduce gradients for all parameters that require gradients."""
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size


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
        ddp_model.all_reduce_gradients()
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
        ddp_model.all_reduce_gradients()
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
    world_size: int
):
    """Print benchmarking results in a formatted way."""
    print("\n" + "="*80)
    print("DDP BENCHMARKING RESULTS")
    print("="*80)
    
    # Model configuration
    print(f"Model Configuration (XL size):")
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
    
    # Model hyperparameters (XL size from benchmarking_script.py)
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
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
    
    # Create model configuration
    config = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta
    }
    
    print("Benchmarking Setup:")
    print(f"  - Model: XL size Transformer ({config['d_model']} dim, {config['num_layers']} layers)")
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
    ddp_results_list = []
    
    for rank in range(args.world_size):
        p = mp.Process(
            target=lambda rank=rank: ddp_results_list.append(
                benchmark_ddp_training(
                    rank=rank,
                    world_size=args.world_size,
                    config=config,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    num_warmup=args.num_warmup,
                    num_steps=args.num_steps
                )
            )
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Average results across all processes
    ddp_results = {}
    for key in ddp_results_list[0].keys():
        values = [results[key] for results in ddp_results_list]
        ddp_results[key] = np.mean(values)
    
    # Print results
    print_results(single_results, ddp_results, config, args.batch_size, args.world_size)


if __name__ == '__main__':
    main() 