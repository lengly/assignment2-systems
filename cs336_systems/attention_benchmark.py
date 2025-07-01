#!/usr/bin/env python3
"""
Attention Benchmarking Script

This script benchmarks attention functions at different scales:
- Fixed batch size of 8
- No multihead attention (removed head dimension)
- Iterates through cartesian product of d_model and sequence lengths
- Measures forward/backward timing and memory usage
- Supports both scaled_dot_product_attention and custom attention functions
"""

import torch
torch.backends.cuda.sdp_kernel.enable_flash = True
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
import psutil
import os
from typing import Dict, List, Tuple, Optional, Callable
import math
from dataclasses import dataclass
import pandas as pd

# Import the attention function from the existing codebase
from attention_functions import (
    ScaledDotProductAttention,
    CustomAttentionImplementation,
    PytorchFlashAttention,
    CompiledScaledDotProductAttention,
)


benchmark_class_dict = {
    "CompiledScaledDotProductAttention": CompiledScaledDotProductAttention,
    "ScaledDotProductAttention": ScaledDotProductAttention,
    "CustomAttentionImplementation": CustomAttentionImplementation,
    "PytorchFlashAttention": PytorchFlashAttention,
}

@dataclass
class BenchmarkConfig:
    """Configuration for attention benchmarking."""
    batch_size: int = 8
    d_model_values: Optional[List[int]] = None
    seq_len_values: Optional[List[int]] = None
    num_warmup: int = 10
    num_forward_passes: int = 100
    num_backward_passes: int = 100
    device: str = 'cuda'
    
    def __post_init__(self):
        if self.d_model_values is None:
            self.d_model_values = [16, 32, 64, 128]
        if self.seq_len_values is None:
            self.seq_len_values = [256, 1024, 4096, 8192, 16384]


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # Fallback to system memory
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


def create_attention_inputs(batch_size: int, seq_len: int, d_model: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V tensors for attention computation."""
    torch.manual_seed(42)  # For reproducible results
    
    # Create tensors with shape (batch_size, seq_len, d_model)
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.bfloat16)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.bfloat16)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True, dtype=torch.bfloat16)
    
    return Q, K, V


def benchmark_attention_function(
    attention_cls: torch.nn.Module,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: BenchmarkConfig,
    cls_name: str
) -> Dict:
    """
    Benchmark a specific attention function.
    
    Args:
        attention_func: The attention function to benchmark
        Q, K, V: Input tensors
        config: Benchmark configuration
        func_name: Name of the function for reporting
    
    Returns:
        Dictionary with benchmark results
    """
    device = Q.device
    casual_mask = torch.triu(torch.ones(Q.shape[1], K.shape[1], device=device), diagonal=1)
    
    # Warmup phase
    model = attention_cls()
    print(f"  Warming up {cls_name}...")
    for _ in range(config.num_warmup):
        # with torch.no_grad():
            # with torch.autocast("cuda", dtype=torch.bfloat16):
        _ = model(Q, K, V, mask=casual_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Clear memory and gradients
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    Q.grad = None
    K.grad = None
    V.grad = None

    # Forward pass benchmarking
    print(f"  Benchmarking {config.num_forward_passes} forward passes...")
    forward_times = []
    
    for _ in range(config.num_forward_passes):
        start_time = time.time()
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(Q, K, V, mask=casual_mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        forward_times.append(end_time - start_time)
    
    # Measure memory before backward pass
    memory_before_backward = get_memory_usage()
    
    # Backward pass benchmarking
    print(f"  Benchmarking {config.num_backward_passes} backward passes...")
    backward_times = []
    
    for _ in range(config.num_backward_passes):
        # Reset gradients
        Q.grad = None
        K.grad = None
        V.grad = None
        
        # Forward pass
        output = model(Q, K, V)
        
        # Create dummy gradient for backward pass
        grad_output = torch.randn_like(output)
        
        # Backward pass timing
        start_time = time.time()
        output.backward(grad_output)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        backward_times.append(end_time - start_time)
    
    # Calculate statistics
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    
    results = {
        'class_name': cls_name,
        'forward_mean_ms': forward_times.mean().item() * 1000,
        'forward_std_ms': forward_times.std().item() * 1000,
        'forward_min_ms': forward_times.min().item() * 1000,
        'forward_max_ms': forward_times.max().item() * 1000,
        'backward_mean_ms': backward_times.mean().item() * 1000,
        'backward_std_ms': backward_times.std().item() * 1000,
        'backward_min_ms': backward_times.min().item() * 1000,
        'backward_max_ms': backward_times.max().item() * 1000,
        'memory_before_backward_mb': memory_before_backward,
        'total_forward_time_s': forward_times.sum().item(),
        'total_backward_time_s': backward_times.sum().item(),
    }
    
    return results


def calculate_memory_usage_theoretical(batch_size: int, seq_len: int, d_model: int) -> Dict[str, float]:
    """
    Calculate theoretical memory usage for attention computation.
    
    Returns:
        Dictionary with memory usage breakdown in MB
    """
    # Assuming float32 (4 bytes per element)
    bytes_per_element = 4
    
    # Input tensors: Q, K, V
    input_memory = 3 * batch_size * seq_len * d_model * bytes_per_element
    
    # Attention scores matrix: (batch_size, seq_len, seq_len)
    attention_scores_memory = batch_size * seq_len * seq_len * bytes_per_element
    
    # Attention weights matrix: (batch_size, seq_len, seq_len)
    attention_weights_memory = batch_size * seq_len * seq_len * bytes_per_element
    
    # Output tensor: (batch_size, seq_len, d_model)
    output_memory = batch_size * seq_len * d_model * bytes_per_element
    
    # Total forward pass memory
    forward_memory = input_memory + attention_scores_memory + attention_weights_memory + output_memory
    
    # Backward pass memory (gradients for Q, K, V + intermediate gradients)
    backward_memory = input_memory + attention_scores_memory + attention_weights_memory
    
    total_memory = forward_memory + backward_memory
    
    return {
        'input_memory_mb': input_memory / 1024 / 1024,
        'attention_scores_memory_mb': attention_scores_memory / 1024 / 1024,
        'attention_weights_memory_mb': attention_weights_memory / 1024 / 1024,
        'output_memory_mb': output_memory / 1024 / 1024,
        'forward_memory_mb': forward_memory / 1024 / 1024,
        'backward_memory_mb': backward_memory / 1024 / 1024,
        'total_memory_mb': total_memory / 1024 / 1024,
    }


def run_attention_benchmarks(config: BenchmarkConfig) -> List[Dict]:
    """
    Run attention benchmarks for all combinations of d_model and seq_len.
    
    Returns:
        List of benchmark results
    """
    results = []
    
    # Check device availability
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.device = 'cpu'
    
    device = torch.device(config.device)
    print(f"Running benchmarks on device: {device}")
    
    # Generate all combinations
    d_model_values = config.d_model_values or [16, 32, 64, 128]
    seq_len_values = config.seq_len_values or [256, 1024, 4096, 8192, 16384]
    combinations = list(itertools.product(d_model_values, seq_len_values))
    total_combinations = len(combinations)
    
    print(f"Total combinations to test: {total_combinations}")
    print(f"d_model values: {d_model_values}")
    print(f"sequence length values: {seq_len_values}")
    
    for i, (d_model, seq_len) in enumerate(combinations):
        print(f"\n[{i+1}/{total_combinations}] Testing d_model={d_model}, seq_len={seq_len}")
        
        try:
            # Create inputs
            Q, K, V = create_attention_inputs(config.batch_size, seq_len, d_model, device)
            
            # Calculate theoretical memory usage
            theoretical_memory = calculate_memory_usage_theoretical(config.batch_size, seq_len, d_model)
            
            for cls_name, cls_instance in benchmark_class_dict.items():
                try:
                    sdp_results = benchmark_attention_function(cls_instance, Q, K, V, config, cls_name)
                    sdp_results.update({
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'batch_size': config.batch_size,
                        'status': 'success',
                        'error': None,
                    })
                    sdp_results.update(theoretical_memory)
                    results.append(sdp_results)
                except Exception as e:
                    print(f"  Error with {cls_name}: {e}")
                    results.append({
                        'class_name': cls_name,
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'batch_size': config.batch_size,
                        'status': 'error',
                        'error': str(e),
                        'forward_mean_ms': None,
                        'backward_mean_ms': None,
                        'memory_before_backward_mb': None,
                    })
            # Clean up
            del Q, K, V
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error creating inputs for d_model={d_model}, seq_len={seq_len}: {e}")
            # Add error entries for both functions
            for cls_name in ["ScaledDotProductAttention", "CustomAttentionImplementation"]:
                results.append({
                    'class_name': cls_name,
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'batch_size': config.batch_size,
                    'status': 'error',
                    'error': str(e),
                    'forward_mean_ms': None,
                    'backward_mean_ms': None,
                    'memory_before_backward_mb': None,
                })
    
    return results


def print_results_table(results: List[Dict]):
    """Print results in a formatted table."""
    print("\n" + "="*120)
    print("ATTENTION BENCHMARK RESULTS")
    print("="*120)
    
    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']
    error_results = [r for r in results if r['status'] == 'error']
    
    if successful_results:
        print("\nSUCCESSFUL BENCHMARKS:")
        print("-" * 120)
        print(f"{'Function':<25} {'d_model':<8} {'seq_len':<8} {'Forward (ms)':<12} {'Backward (ms)':<13} {'Memory (MB)':<12} {'Total Mem (MB)':<15}")
        print("-" * 120)
        
        for result in successful_results:
            print(f"{result['class_name']:<25} {result['d_model']:<8} {result['seq_len']:<8} "
                  f"{result['forward_mean_ms']:<12.3f} {result['backward_mean_ms']:<13.3f} "
                  f"{result['memory_before_backward_mb']:<12.1f} {result['total_memory_mb']:<15.1f}")
    
    if error_results:
        print(f"\nERRORS ({len(error_results)} configurations failed):")
        print("-" * 120)
        print(f"{'Function':<25} {'d_model':<8} {'seq_len':<8} {'Error':<50}")
        print("-" * 120)
        
        for result in error_results:
            error_msg = result['error'][:47] + "..." if len(result['error']) > 50 else result['error']
            print(f"{result['class_name']:<25} {result['d_model']:<8} {result['seq_len']:<8} {error_msg:<50}")


def save_results_to_csv(results: List[Dict], filename: str = "attention_benchmark_results.csv"):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def analyze_memory_usage(results: List[Dict]):
    """Analyze memory usage patterns and provide insights."""
    print("\n" + "="*80)
    print("MEMORY USAGE ANALYSIS")
    print("="*80)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to analyze.")
        return
    
    # Find the smallest configuration that ran out of memory
    error_results = [r for r in results if r['status'] == 'error' and 'out of memory' in r['error'].lower()]
    
    if error_results:
        print("\nOUT OF MEMORY ANALYSIS:")
        print("-" * 80)
        
        # Find the smallest configuration that failed
        min_error = min(error_results, key=lambda x: (x['d_model'], x['seq_len']))
        print(f"Smallest configuration that ran out of memory:")
        print(f"  d_model: {min_error['d_model']}")
        print(f"  seq_len: {min_error['seq_len']}")
        print(f"  batch_size: {min_error['batch_size']}")
        
        # Calculate theoretical memory for this configuration
        theoretical_memory = calculate_memory_usage_theoretical(
            min_error['batch_size'], min_error['seq_len'], min_error['d_model']
        )
        
        print(f"\nTheoretical memory usage for this configuration:")
        print(f"  Input tensors (Q, K, V): {theoretical_memory['input_memory_mb']:.1f} MB")
        print(f"  Attention scores matrix: {theoretical_memory['attention_scores_memory_mb']:.1f} MB")
        print(f"  Attention weights matrix: {theoretical_memory['attention_weights_memory_mb']:.1f} MB")
        print(f"  Output tensor: {theoretical_memory['output_memory_mb']:.1f} MB")
        print(f"  Forward pass total: {theoretical_memory['forward_memory_mb']:.1f} MB")
        print(f"  Backward pass memory: {theoretical_memory['backward_memory_mb']:.1f} MB")
        print(f"  Total memory required: {theoretical_memory['total_memory_mb']:.1f} MB")
        
        # Check available GPU memory
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            print(f"\nGPU Memory Information:")
            print(f"  Total GPU memory: {total_gpu_memory:.1f} MB")
            print(f"  Memory utilization: {theoretical_memory['total_memory_mb'] / total_gpu_memory * 100:.1f}%")
    
    # Analyze memory scaling with sequence length
    print("\nMEMORY SCALING ANALYSIS:")
    print("-" * 80)
    
    # Group by d_model and analyze memory scaling with seq_len
    for d_model in sorted(set(r['d_model'] for r in successful_results)):
        d_model_results = [r for r in successful_results if r['d_model'] == d_model]
        if len(d_model_results) > 1:
            print(f"\nd_model = {d_model}:")
            print(f"{'seq_len':<8} {'Memory (MB)':<12} {'Scaling Factor':<15}")
            print("-" * 35)
            
            prev_memory = None
            for result in sorted(d_model_results, key=lambda x: x['seq_len']):
                memory = result['memory_before_backward_mb']
                scaling_factor = ""
                if prev_memory is not None:
                    scaling_factor = f"{memory / prev_memory:.2f}x"
                print(f"{result['seq_len']:<8} {memory:<12.1f} {scaling_factor:<15}")
                prev_memory = memory


def main():
    """Main function to run attention benchmarks."""
    print("Attention Benchmarking Script")
    print("=" * 50)
    
    # Create configuration
    config = BenchmarkConfig()
    
    # Run benchmarks
    results = run_attention_benchmarks(config)
    
    # Print results
    print_results_table(results)
    
    # Analyze memory usage
    analyze_memory_usage(results)
    
    # Save results
    save_results_to_csv(results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)
    
    # Summary
    successful_count = len([r for r in results if r['status'] == 'success'])
    error_count = len([r for r in results if r['status'] == 'error'])
    total_count = len(results)
    
    print(f"Total configurations tested: {total_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {error_count}")
    print(f"Success rate: {successful_count/total_count*100:.1f}%")

def check_flash_attention_availability():
    """
    Check if Flash Attention is available and working.
    """
    print("Checking Flash Attention availability...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - Flash Attention requires CUDA")
        return False
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if Flash Attention kernels are available
    try:
        # Create test tensors
        batch_size, seq_len, d_model = 2, 128, 64
        Q = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)
        
        # Try to use Flash Attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        print("✅ Flash Attention is available and working")
        return True
        
    except Exception as e:
        print(f"❌ Flash Attention not available: {e}")
        return False

if __name__ == "__main__":
    main() 