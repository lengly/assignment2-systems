#!/usr/bin/env python3
"""
Benchmarking script for Transformer model forward and backward passes.
"""

import argparse
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from typing import Dict, Any
import sys
import os

from cs336_basics.model import BasicsTransformerLM


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


def benchmark_model(
    model: nn.Module,
    batch: torch.Tensor,
    num_warmup: int,
    num_steps: int,
    forward_only: bool = False,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark the model's forward and/or backward passes.
    
    Args:
        model: The model to benchmark
        batch: Input batch tensor
        num_warmup: Number of warm-up steps
        num_steps: Number of steps to time
        forward_only: If True, only time forward pass; if False, time both forward and backward
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with timing results
    """
    model.train()
    
    # Warm-up phase
    print(f"Running {num_warmup} warm-up steps...")
    with nvtx.range("warmup", color="green"):
        for _ in range(num_warmup):
            if forward_only:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = model(batch)
            else:
                # Forward pass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                # Create dummy targets for loss calculation
                targets = torch.randint(0, logits.size(-1), (batch.size(0), batch.size(1)), device=device)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # Backward pass
                loss.backward()
            
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmarking phase
    print(f"Running {num_steps} benchmark steps...")
    forward_times = []
    backward_times = []
    total_times = []
    
    with nvtx.range("benchmarking", color="blue"):
        for step in range(num_steps):
            if forward_only:
                # Forward pass only
                start_time = timeit.default_timer()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        _ = model(batch)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = timeit.default_timer()
                forward_times.append(end_time - start_time)
                total_times.append(end_time - start_time)
            else:
                # Forward pass timing
                start_forward = timeit.default_timer()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch)
                # Create dummy targets for loss calculation
                targets = torch.randint(0, logits.size(-1), (batch.size(0), batch.size(1)), device=device)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_forward = timeit.default_timer()
                forward_times.append(end_forward - start_forward)
                
                # Backward pass timing
                start_backward = timeit.default_timer()
                loss.backward()
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_backward = timeit.default_timer()
                backward_times.append(end_backward - start_backward)
                
                # Total time
                total_times.append(end_backward - start_forward)
            
            if (step + 1) % 10 == 0:
                print(f"Completed {step + 1}/{num_steps} steps")
    
    # Calculate statistics
    forward_times = torch.tensor(forward_times)
    total_times = torch.tensor(total_times)
    
    results = {
        'forward_mean': forward_times.mean().item(),
        'forward_std': forward_times.std().item(),
        'forward_min': forward_times.min().item(),
        'forward_max': forward_times.max().item(),
        'total_mean': total_times.mean().item(),
        'total_std': total_times.std().item(),
        'total_min': total_times.min().item(),
        'total_max': total_times.max().item(),
        'total_time': total_times.sum().item(),
        'throughput': num_steps / total_times.sum().item() if total_times.sum().item() > 0 else 0
    }
    
    if not forward_only:
        backward_times = torch.tensor(backward_times)
        results.update({
            'backward_mean': backward_times.mean().item(),
            'backward_std': backward_times.std().item(),
            'backward_min': backward_times.min().item(),
            'backward_max': backward_times.max().item(),
        })
    
    return results


def print_results(results: Dict[str, float], config: Dict[str, Any], model: nn.Module, forward_only: bool):
    """Print benchmarking results in a formatted way."""
    print("\n" + "="*60)
    print("BENCHMARKING RESULTS")
    print("="*60)
    
    # Model configuration
    print(f"Model Configuration:")
    print(f"  - Vocab size: {config['vocab_size']:,}")
    print(f"  - Context length: {config['context_length']:,}")
    print(f"  - d_model: {config['d_model']:,}")
    print(f"  - Num layers: {config['num_layers']:,}")
    print(f"  - Num heads: {config['num_heads']:,}")
    print(f"  - d_ff: {config['d_ff']:,}")
    print(f"  - Rope theta: {config['rope_theta']}")
    
    # Timing results
    print(f"\nTiming Results ({'Forward only' if forward_only else 'Forward + Backward'}):")
    
    print(f"\nForward Pass:")
    print(f"  - Mean time: {results['forward_mean']*1000:.3f} ms")
    print(f"  - Std time: {results['forward_std']*1000:.3f} ms")
    print(f"  - Min time: {results['forward_min']*1000:.3f} ms")
    print(f"  - Max time: {results['forward_max']*1000:.3f} ms")
    
    if not forward_only:
        print(f"\nBackward Pass:")
        print(f"  - Mean time: {results['backward_mean']*1000:.3f} ms")
        print(f"  - Std time: {results['backward_std']*1000:.3f} ms")
        print(f"  - Min time: {results['backward_min']*1000:.3f} ms")
        print(f"  - Max time: {results['backward_max']*1000:.3f} ms")
    
    print(f"\nTotal Time:")
    print(f"  - Mean time: {results['total_mean']*1000:.3f} ms")
    print(f"  - Std time: {results['total_std']*1000:.3f} ms")
    print(f"  - Min time: {results['total_min']*1000:.3f} ms")
    print(f"  - Max time: {results['total_max']*1000:.3f} ms")
    print(f"  - Total time: {results['total_time']:.3f} s")
    print(f"  - Throughput: {results['throughput']:.2f} steps/s")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer model performance')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=512, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Benchmarking parameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of benchmark steps')
    parser.add_argument('--forward_only', action='store_true', help='Only benchmark forward pass')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
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
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    print("Initializing model...")
    model = create_model(config, args.device)
    
    print("Generating random batch...")
    batch = generate_random_batch(args.batch_size, args.seq_len, args.vocab_size, args.device)
    
    print(f"Starting benchmark on {args.device.upper()}...")
    results = benchmark_model(
        model=model,
        batch=batch,
        num_warmup=args.num_warmup,
        num_steps=args.num_steps,
        forward_only=args.forward_only,
        device=args.device
    )
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    print_results(results, config, model, args.forward_only)


if __name__ == '__main__':
    main()
