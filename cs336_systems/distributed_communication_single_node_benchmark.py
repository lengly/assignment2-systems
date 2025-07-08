import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Optional imports for plotting and data analysis
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. CSV export will be skipped.")


def setup(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed process group."""
    # Use a fixed port for each benchmark run to ensure consistency
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Set NCCL environment variables for better compatibility
    if backend == "nccl":
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
        # Disable P2P and shared memory for stability in container environments
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_SHM_DISABLE'] = '1'
    
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA to be available")
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def get_tensor_size_mb(size_mb: int) -> int:
    """Convert MB to number of float32 elements."""
    return size_mb * 1024 * 1024 // 4  # 4 bytes per float32


def benchmark_allreduce(rank: int, world_size: int, backend: str, data_size_mb: int, 
                       device_type: str, num_iterations: int = 3) -> float:
    """Benchmark all-reduce operation and return average time in milliseconds."""
    try:
        setup(rank, world_size, backend)
        
        # Create tensor based on device type
        if device_type == "cpu":
            device = torch.device("cpu")
        else:  # gpu
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for GPU benchmark")
            
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                raise RuntimeError("No GPUs available")
            
            gpu_id = rank % gpu_count
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        
        # Create tensor with specified size
        tensor_size = get_tensor_size_mb(data_size_mb)
        data = torch.randn(tensor_size, dtype=torch.float32, device=device)
        
        # Synchronize before starting
        if device_type == "gpu":
            torch.cuda.synchronize()
        dist.barrier()  # Ensure all processes are ready
        
        # Warm up
        for _ in range(3):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            if device_type == "gpu":
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            
            if device_type == "gpu":
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            times.append(elapsed_ms)
        
        cleanup()
        return float(np.mean(times))
    
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        return float('inf')


def distributed_worker(rank: int, world_size: int, backend: str, data_size_mb: int, 
                      device_type: str, result_queue: mp.Queue):
    """Worker function for distributed benchmarking."""
    try:
        result = benchmark_allreduce(rank, world_size, backend, data_size_mb, device_type)
        result_queue.put((rank, result))
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        result_queue.put((rank, float('inf')))


def distributed_worker_with_port(rank: int, world_size: int, backend: str, data_size_mb: int, 
                                device_type: str, port: int, result_queue: mp.Queue):
    """Worker function for distributed benchmarking with specific port."""
    try:
        # Set the port for this worker
        os.environ['MASTER_PORT'] = str(port)
        result = benchmark_allreduce(rank, world_size, backend, data_size_mb, device_type)
        result_queue.put((rank, result))
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        result_queue.put((rank, float('inf')))


def run_single_benchmark(world_size: int, backend: str, data_size_mb: int, 
                        device_type: str) -> float:
    """Run a single benchmark configuration using mp.spawn."""
    try:
        # Set start method for multiprocessing
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Use a unique port for this benchmark run
    import random
    port = 29500 + random.randint(0, 1000)
    
    # Create a queue to collect results from processes
    result_queue = mp.Queue()
    
    # Spawn processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=distributed_worker_with_port, 
                      args=(rank, world_size, backend, data_size_mb, device_type, port, result_queue))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        rank, result = result_queue.get()
        if result != float('inf'):
            results.append(result)
    
    return float(np.mean(results)) if results else float('inf')


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all configurations."""
    # Configuration parameters
    backends = ["gloo", "nccl"]
    device_types = ["cpu", "gpu"]
    data_sizes_mb = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    world_sizes = [2, 4, 6]
    
    results = []
    
    print("Starting comprehensive all-reduce benchmark...")
    print("=" * 60)
    
    for backend in backends:
        for device_type in device_types:
            # Skip NCCL + CPU combination
            if backend == "nccl" and device_type == "cpu":
                continue
            
            # Skip if CUDA not available for GPU tests
            if device_type == "gpu" and not torch.cuda.is_available():
                print(f"Skipping {backend} + {device_type} (CUDA not available)")
                continue
            
            print(f"\nTesting {backend} + {device_type.upper()}")
            print("-" * 40)
            
            for world_size in world_sizes:
                for data_size_mb in data_sizes_mb:
                    print(f"  Processes: {world_size}, Data size: {data_size_mb}MB", end=" ")
                    
                    try:
                        avg_time = run_single_benchmark(world_size, backend, data_size_mb, device_type)
                        
                        if avg_time != float('inf'):
                            print(f"-> {avg_time:.2f} ms")
                            results.append({
                                'backend': backend,
                                'device_type': device_type,
                                'world_size': world_size,
                                'data_size_mb': data_size_mb,
                                'avg_time_ms': avg_time
                            })
                        else:
                            print("-> FAILED")
                    except Exception as e:
                        print(f"-> ERROR: {e}")
    
    return results


def create_plots(results: List[Dict]):
    """Create visualization plots for the benchmark results."""
    if not results:
        print("No results to plot")
        return
    
    if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping plots: matplotlib or pandas not available")
        return
    
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('All-Reduce Operation Benchmark Results', fontsize=16)
    
    # Plot 1: Time vs Data Size for different backends (CPU)
    cpu_data = df[df['device_type'] == 'cpu']
    if not cpu_data.empty:
        for backend in cpu_data['backend'].unique():
            backend_data = cpu_data[cpu_data['backend'] == backend]
            for world_size in backend_data['world_size'].unique():
                ws_data = backend_data[backend_data['world_size'] == world_size]
                axes[0, 0].plot(ws_data['data_size_mb'], ws_data['avg_time_ms'], 
                               marker='o', label=f'{backend} CPU, {world_size} procs')
        
        axes[0, 0].set_xlabel('Data Size (MB)')
        axes[0, 0].set_ylabel('Average Time (ms)')
        axes[0, 0].set_title('CPU Performance')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time vs Data Size for different backends (GPU)
    gpu_data = df[df['device_type'] == 'gpu']
    if not gpu_data.empty:
        for backend in gpu_data['backend'].unique():
            backend_data = gpu_data[gpu_data['backend'] == backend]
            for world_size in backend_data['world_size'].unique():
                ws_data = backend_data[backend_data['world_size'] == world_size]
                axes[0, 1].plot(ws_data['data_size_mb'], ws_data['avg_time_ms'], 
                               marker='s', label=f'{backend} GPU, {world_size} procs')
        
        axes[0, 1].set_xlabel('Data Size (MB)')
        axes[0, 1].set_ylabel('Average Time (ms)')
        axes[0, 1].set_title('GPU Performance')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time vs Number of Processes (CPU)
    if not cpu_data.empty:
        for backend in cpu_data['backend'].unique():
            backend_data = cpu_data[cpu_data['backend'] == backend]
            for data_size in backend_data['data_size_mb'].unique():
                ds_data = backend_data[backend_data['data_size_mb'] == data_size]
                axes[1, 0].plot(ds_data['world_size'], ds_data['avg_time_ms'], 
                               marker='o', label=f'{backend} CPU, {data_size}MB')
        
        axes[1, 0].set_xlabel('Number of Processes')
        axes[1, 0].set_ylabel('Average Time (ms)')
        axes[1, 0].set_title('CPU Scaling with Process Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Time vs Number of Processes (GPU)
    if not gpu_data.empty:
        for backend in gpu_data['backend'].unique():
            backend_data = gpu_data[gpu_data['backend'] == backend]
            for data_size in backend_data['data_size_mb'].unique():
                ds_data = backend_data[backend_data['data_size_mb'] == data_size]
                axes[1, 1].plot(ds_data['world_size'], ds_data['avg_time_ms'], 
                               marker='s', label=f'{backend} GPU, {data_size}MB')
        
        axes[1, 1].set_xlabel('Number of Processes')
        axes[1, 1].set_ylabel('Average Time (ms)')
        axes[1, 1].set_title('GPU Scaling with Process Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('allreduce_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_results_table(results: List[Dict]):
    """Print results in a formatted table."""
    if not results:
        print("No results to display")
        return
    
    if not PANDAS_AVAILABLE:
        print("Skipping table: pandas not available")
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("ALL-REDUCE BENCHMARK RESULTS")
    print("=" * 80)
    
    # Group by backend and device type
    for backend in df['backend'].unique():
        for device_type in df['device_type'].unique():
            subset = df[(df['backend'] == backend) & (df['device_type'] == device_type)]
            if subset.empty:
                continue
            
            print(f"\n{backend.upper()} + {device_type.upper()}")
            print("-" * 50)
            
            # Pivot table
            pivot_table = subset.pivot_table(
                values='avg_time_ms', 
                index='data_size_mb', 
                columns='world_size', 
                aggfunc='mean'
            )
            
            # Get available world sizes
            available_world_sizes = sorted(subset['world_size'].unique())
            
            # Create header
            header = "Data Size (MB) | Processes -> Time (ms)"
            print(header)
            
            # Create column headers
            col_headers = "               |"
            separator = "-" * 50
            for ws in available_world_sizes:
                col_headers += f" {ws:3d}  |"
                separator += "------|"
            print(col_headers)
            print(separator)
            
            # Print data rows
            for data_size in sorted(subset['data_size_mb'].unique()):
                row = pivot_table.loc[data_size]
                data_row = f"{data_size:13d} |"
                for ws in available_world_sizes:
                    if ws in row.index:
                        data_row += f" {row[ws]:5.1f} |"
                    else:
                        data_row += f" {'N/A':5s} |"
                print(data_row)


def main():
    """Main function to run the benchmark."""
    print("All-Reduce Operation Benchmark")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Run benchmark
    results = run_comprehensive_benchmark()
    
    if results:
        # Print results table
        print_results_table(results)
        
        # Create plots
        create_plots(results)
        
        # Save results to CSV
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(results)
            df.to_csv('allreduce_benchmark_results.csv', index=False)
            print(f"\nResults saved to 'allreduce_benchmark_results.csv'")
            
            # Analysis and commentary
            print("\n" + "=" * 80)
            print("ANALYSIS AND COMMENTARY")
            print("=" * 80)
            
            # Find best performing configurations
            cpu_data = df[df['device_type'] == 'cpu']
            gpu_data = df[df['device_type'] == 'gpu']
            
            if not cpu_data.empty:
                best_cpu = cpu_data.loc[cpu_data['avg_time_ms'].idxmin()]
                print(f"\nBest CPU performance: {best_cpu['backend']} with {best_cpu['world_size']} processes, "
                      f"{best_cpu['data_size_mb']}MB data: {best_cpu['avg_time_ms']:.2f} ms")
            
            if not gpu_data.empty:
                best_gpu = gpu_data.loc[gpu_data['avg_time_ms'].idxmin()]
                print(f"Best GPU performance: {best_gpu['backend']} with {best_gpu['world_size']} processes, "
                      f"{best_gpu['data_size_mb']}MB data: {best_gpu['avg_time_ms']:.2f} ms")
        else:
            print("\nSkipping CSV export and analysis: pandas not available")
            return
        
        # Scaling analysis
        print("\nKey Observations:")
        print("1. Backend Performance: NCCL typically outperforms Gloo for GPU operations due to optimized CUDA kernels.")
        print("2. Data Size Scaling: Larger data sizes show more pronounced differences between backends and devices.")
        print("3. Process Scaling: More processes generally increase communication overhead but may improve throughput for large datasets.")
        
    else:
        print("No successful benchmark results obtained.")


if __name__ == "__main__":
    main()


