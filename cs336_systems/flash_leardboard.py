import torch
import triton
from flashattention import CustomFlashAttentTriton

def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    seq_len = 16384
    q, k, v = torch.randn(
        3, n_heads, seq_len, d_head, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(CustomFlashAttentTriton.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()
    
    print("Timing flash forward backward...")
    results = triton.testing.do_bench(flash_forward_backward, warmup=100, rep=100)
    print(results)

if __name__ == "__main__":
    test_timing_flash_forward_backward()

