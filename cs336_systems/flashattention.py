import torch
from torch import Tensor
import math
from einops import einsum, rearrange
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    output_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # Initialize accumulators for Flash Attention
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    for key_tile_index in range(0, N_KEYS, K_TILE_SIZE):
        k_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute attention scores
        s = tl.dot(q_block, k_block.T) * scale

        # Apply causal mask if needed
        if is_causal:
            # Calculate the actual positions for this tile
            q_start = query_tile_index * Q_TILE_SIZE
            k_start = key_tile_index
            
            # Create position indices for queries and keys in this tile
            q_pos = q_start + tl.arange(0, Q_TILE_SIZE)
            k_pos = k_start + tl.arange(0, K_TILE_SIZE)
            
            # Create causal mask: q_pos[:, None] >= k_pos[None, :]
            # This ensures each query can only attend to keys at or before its position
            q_pos_expanded = tl.expand_dims(q_pos, 1)  # (Q_TILE_SIZE, 1)
            k_pos_expanded = tl.expand_dims(k_pos, 0)  # (1, K_TILE_SIZE)
            causal_mask = q_pos_expanded >= k_pos_expanded  # (Q_TILE_SIZE, K_TILE_SIZE)
            
            # Apply mask: set masked positions to -inf
            s = tl.where(causal_mask, s, -1e6)


        # Flash Attention algorithm
        s_max = tl.max(s, axis=-1)
        m_next = tl.maximum(m, s_max)
        
        # Compute softmax weights
        m_next_expanded = tl.expand_dims(m_next, 1)
        p = tl.exp(s - m_next_expanded)
        
        # Update running statistics
        l = tl.exp(m - m_next) * l + tl.sum(p, axis=-1)
        m_expanded = tl.expand_dims(m, 1)
        o = tl.exp(m_expanded - m_next_expanded) * o
        o = tl.dot(p.to(v_block.dtype), v_block, acc=o)
        
        m = m_next

        # Update block pointers for next iteration
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Final normalization and L computation
    l_expanded = tl.expand_dims(l, 1)
    tl.store(output_block_ptr, (o / l_expanded).to(output_block_ptr.type.element_ty))
    # L is the log-sum-exp of all attention scores for each query
    tl.store(L_block_ptr, m + tl.log(l))


class CustomFlashAttentTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        assert K.shape[-1] == V.shape[-1] == d

        scale = 1 / (d ** 0.5)

        Bq = 16
        Bk = 16
        assert d % Bq == 0 and d % Bk == 0
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        output = torch.empty((B, Nq, d), device=Q.device, dtype=Q.dtype)
        L = torch.zeros((B, Nq), device=Q.device, dtype=Q.dtype)

        flash_attn_fwd_kernel[Tq, B](
            Q, K, V,
            output, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        ctx.save_for_backward(L)
        return output


class CustomFlashAttentPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, Nq, d = Q.shape
        Nk = K.shape[-2]
        assert K.shape[-1] == V.shape[-1] == d
        
        # Add scale factor
        scale = 1 / (d ** 0.5)
        
        Bq = 16
        Bk = 16
        assert d % Bq == 0 and d % Bk == 0
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        output = torch.empty_like(Q)
        L = torch.zeros((B, Nq))

        for i in range(Tq):
            q_start = i * Bq
            q_end = min(q_start + Bq, Nq)
            q = Q[..., q_start:q_end, :]
            o = torch.zeros((B, q_end - q_start, d))
            l = torch.zeros((B, q_end - q_start,))
            m = torch.full((B, q_end - q_start,), float("-inf"))

            for j in range(Tk):
                k_start = j * Bk
                k_end = min(k_start + Bk, Nk)
                k = K[..., k_start:k_end, :]
                v = V[..., k_start:k_end, :]
                
                # Apply scale factor
                s = einsum(q, k, "B Bq d, B Bk d -> B Bq Bk") * scale
                
                # Handle causal masking if needed
                if is_causal:
                    causal_mask = torch.arange(q_start, q_end, device=s.device)[:, None] >= torch.arange(k_start, k_end, device=s.device)[None, :]
                    s = torch.where(causal_mask.unsqueeze(0), s, float("-inf"))
                
                m_next = torch.max(m, torch.max(s, dim=-1).values)
                
                # More numerically stable computation
                p = torch.exp(s - m_next.unsqueeze(-1))
                l_next = torch.exp(m - m_next) * l + p.sum(dim=-1)
                o_next = torch.exp(m - m_next).unsqueeze(-1) * o + einsum(p, v, "B Bq Bk, B Bk d -> B Bq d")
                
                m = m_next
                l = l_next
                o = o_next
            
            # Correct final output calculation
            output[..., q_start:q_end, :] = o / l.unsqueeze(-1)
            L[..., q_start:q_end] = m + torch.log(l)
        
        ctx.save_for_backward(L)
        return output
        

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

