import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import os
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
from copy import deepcopy

# Set multiprocessing start method to spawn for CUDA compatibility
mp.set_start_method('spawn', force=True)


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
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


class ToyModel(nn.Module):
    """Simple toy model for testing DDP."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=True)
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_process_group(rank, world_size):
    """Setup distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set device for GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = rank % device_count
        # Initialize CUDA before setting device
        torch.cuda.init()
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


def train_single_process(model, data, labels, optimizer, loss_fn, num_epochs=5):
    """Train model in single process mode."""
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


def train_ddp_process(rank, world_size, model, all_data, all_labels, optimizer, loss_fn, num_epochs=5):
    """Train model using DDP."""
    device = setup_process_group(rank, world_size)
    
    # Create DDP model and move to device
    ddp_model = NaiveDDP(model).to(device)
    
    # Split data across processes and move to device
    batch_size = all_data.size(0) // world_size
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    local_data = all_data[start_idx:end_idx].to(device)
    local_labels = all_labels[start_idx:end_idx].to(device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = ddp_model(local_data)
        loss = loss_fn(outputs, local_labels)
        loss.backward()
        
        # All-reduce gradients
        ddp_model.all_reduce_gradients()
        
        optimizer.step()
    
    cleanup_process_group()
    return ddp_model.module


def verify_ddp_correctness():
    """Verify that DDP training produces same results as single process training."""
    world_size = 2
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate random data
    torch.manual_seed(42)
    all_data = torch.randn(20, 10).to(device)
    all_labels = torch.randn(20, 5).to(device)
    
    # Create models
    single_model = ToyModel().to(device)
    ddp_model = deepcopy(single_model)
    # Create optimizers
    single_optimizer = optim.SGD(single_model.parameters(), lr=0.1)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Train single process model
    single_trained = train_single_process(single_model, all_data, all_labels, single_optimizer, loss_fn)
    
    # Train DDP model
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_ddp_process, args=(rank, world_size, ddp_model, all_data, all_labels, ddp_optimizer, loss_fn))
        p.start()
        processes.append(p)
    print('waiting for processes to finish')
    for p in processes:
        p.join()
    
    # Move models to CPU for comparison
    single_trained = single_model.cpu()
    ddp_model = ddp_model.cpu()
    
    # Verify weights match
    for i, (single_param, ddp_param) in enumerate(zip(single_trained.parameters(), ddp_model.parameters())):
        if single_param.requires_grad:
            diff = torch.abs(single_param - ddp_param).max().item()
            print(f"Parameter {i} max difference: {diff}")
            assert torch.allclose(single_param, ddp_param, atol=1e-5), f"Weights don't match for parameter {i}!"
    
    print("✅ DDP implementation verified! Weights match single process training.")


if __name__ == "__main__":
    verify_ddp_correctness()
