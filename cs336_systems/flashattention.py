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
    tl.store(L_block_ptr, (m + tl.log(l)).to(L_block_ptr.type.element_ty))


# (A * B).sum(axis=-1)
@triton.jit
def flash_attn_bwd_preprocess_kernel(
    A_ptr, B_ptr,
    output_ptr,
    stride_ab, stride_ar, stride_ad,
    stride_bb, stride_br, stride_bd,
    stride_ob, stride_or,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    A_block_ptr = tl.make_block_ptr(
        A_ptr + batch_index * stride_ab,
        shape=(ROWS, D),
        strides=(stride_ar, stride_ad),
        offsets=(row_tile_index * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    B_block_ptr = tl.make_block_ptr(
        B_ptr + batch_index * stride_bb,
        shape=(ROWS, D),
        strides=(stride_br, stride_bd),
        offsets=(row_tile_index * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr + batch_index * stride_ob,
        shape=(ROWS,),
        strides=(stride_or,),
        offsets=(row_tile_index * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
        output += tl.sum(a * b, axis=-1)
        A_block_ptr = A_block_ptr.advance((0, D_TILE_SIZE))
        B_block_ptr = B_block_ptr.advance((0, D_TILE_SIZE))
        
    tl.store(output_block_ptr, output.to(output_block_ptr.type.element_ty))


@triton.jit
def flash_attn_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr,
    grad_output_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dQb, stride_dQq, stride_dQd,
    stride_dKb, stride_dKk, stride_dKd,
    stride_dVb, stride_dVk, stride_dVd,
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

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr + batch_index * stride_ob,
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
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dQb,
        shape=(N_QUERIES, D),
        strides=(stride_dQq, stride_dQd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do_block = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    D_block_expanded = tl.expand_dims(tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero"), 1)
    
    dq_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE
    q_pos = q_start + tl.arange(0, Q_TILE_SIZE)
    q_pos_expanded = tl.expand_dims(q_pos, 1)  # (Q_TILE_SIZE, 1)
    for key_tile_index in range(0, N_KEYS, K_TILE_SIZE):
        k_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        s = tl.dot(q_block, k_block.T) * scale    # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            k_start = key_tile_index
            k_pos = k_start + tl.arange(0, K_TILE_SIZE)
            k_pos_expanded = tl.expand_dims(k_pos, 0)  # (1, K_TILE_SIZE)
            causal_mask = q_pos_expanded >= k_pos_expanded  # (Q_TILE_SIZE, K_TILE_SIZE)
            s = tl.where(causal_mask, s, -1e6)

        L_block_expanded = tl.expand_dims(L_block, 1)   # (Q_TILE_SIZE, 1)
        p = tl.exp(s - L_block_expanded)   # (Q_TILE_SIZE, K_TILE_SIZE)
        dp_block = tl.dot(do_block, v_block.T)   # (Q_TILE_SIZE, K_TILE_SIZE)
        ds_block = p * (dp_block - D_block_expanded) * scale
        dq_block = tl.dot(ds_block.to(k_block.dtype), k_block, acc=dq_block)

        # update pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(dQ_block_ptr, dq_block.to(dQ_block_ptr.type.element_ty))


@triton.jit
def flash_attn_bwd_dKV_kernel(
    Q_ptr, K_ptr, V_ptr,
    grad_output_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dQb, stride_dQq, stride_dQd,
    stride_dKb, stride_dKk, stride_dKd,
    stride_dVb, stride_dVk, stride_dVd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dKb,
        shape=(N_KEYS, D),
        strides=(stride_dKk, stride_dKd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dVb,
        shape=(N_KEYS, D),
        strides=(stride_dVk, stride_dVd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    k_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    dk_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dv_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    k_start = key_tile_index * K_TILE_SIZE
    k_pos = k_start + tl.arange(0, K_TILE_SIZE)
    k_pos_expanded = tl.expand_dims(k_pos, 0)  # (1, K_TILE_SIZE)
    for query_tile_index in range(0, N_QUERIES, Q_TILE_SIZE):
        q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do_block = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_block_expanded = tl.expand_dims(tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero"), 1)
        
        s = tl.dot(q_block, k_block.T) * scale    # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            q_start = query_tile_index
            q_pos = q_start + tl.arange(0, Q_TILE_SIZE)
            q_pos_expanded = tl.expand_dims(q_pos, 1)  # (Q_TILE_SIZE, 1)
            causal_mask = q_pos_expanded >= k_pos_expanded  # (Q_TILE_SIZE, K_TILE_SIZE)
            s = tl.where(causal_mask, s, -1e6)

        L_block_expanded = tl.expand_dims(L_block, 1)   # (Q_TILE_SIZE, 1)
        p = tl.exp(s - L_block_expanded)   # (Q_TILE_SIZE, K_TILE_SIZE)
        dp_block = tl.dot(do_block, v_block.T)   # (Q_TILE_SIZE, K_TILE_SIZE)
        ds_block = p * (dp_block - D_block_expanded) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        dv_block = tl.dot(p.T.to(do_block.dtype), do_block, acc=dv_block)  # (K_TILE_SIZE, D)
        dk_block = tl.dot(ds_block.T.to(q_block.dtype), q_block, acc=dk_block)  # (K_TILE_SIZE, D)

        # update pointers
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        grad_output_block_ptr = grad_output_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dk_block.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dv_block.to(dV_block_ptr.type.element_ty))


class CustomFlashAttentTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if Q.ndim == 4:
            ctx.is_4d = True
            B, nh, Nq, d = Q.shape
            Q = rearrange(Q, "B nh Nq d -> (B nh) Nq d")
            K = rearrange(K, "B nh Nk d -> (B nh) Nk d")
            V = rearrange(V, "B nh Nk d -> (B nh) Nk d")
        else:
            ctx.is_4d = False
            B, Nq, d = Q.shape

        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        
        # B, Nq, d = Q.shape
        Nk = K.shape[-2]
        assert K.shape[-1] == V.shape[-1] == d

        scale = 1 / (d ** 0.5)

        Bq = 32
        Bk = 32
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

        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        if ctx.is_4d:
            return rearrange(output, "(B nh) Nq d -> B nh Nq d", B=B)
        else:
            return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if ctx.is_4d:
            B, nh = grad_output.shape[:2]
            grad_output = rearrange(grad_output, "B nh Nq d -> (B nh) Nq d")
        else:
            B, Nq, d = grad_output.shape

        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        assert L.is_cuda and Q.is_cuda and K.is_cuda and V.is_cuda and O.is_cuda
        assert L.is_contiguous() and Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and O.is_contiguous()
        
        Nq, d = Q.shape[-2:]
        Nk = K.shape[-2]
        assert K.shape[-1] == V.shape[-1] == d
        
        scale = 1 / (d ** 0.5)

        Bq = 32
        Bk = 32
        assert d % Bq == 0 and d % Bk == 0
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        # compute D = (O * grad_output).sum(dim=-1)  # (B, Nq)
        D = torch.zeros((B, Nq), device=Q.device, dtype=Q.dtype)
        flash_attn_bwd_preprocess_kernel[Tq, B](
            O, grad_output,
            D,
            O.stride(0), O.stride(1), O.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            D.stride(0), D.stride(1),
            Nq, d,
            ROWS_TILE_SIZE=Bq,
            D_TILE_SIZE=Bq,
        )

        dQ = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)
        dK = torch.zeros_like(K, device=K.device, dtype=K.dtype)
        dV = torch.zeros_like(V, device=V.device, dtype=V.dtype)
        
        flash_attn_bwd_dQ_kernel[Tq, B](
            Q, K, V,
            grad_output, L, D,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        flash_attn_bwd_dKV_kernel[Tk, B](
            Q, K, V,
            grad_output, L, D,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        if ctx.is_4d:
            dQ = rearrange(dQ, "(B nh) Nq d -> B nh Nq d", B=B)
            dK = rearrange(dK, "(B nh) Nk d -> B nh Nk d", B=B)
            dV = rearrange(dV, "(B nh) Nk d -> B nh Nk d", B=B)
        return dQ, dK, dV, None


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
        L = torch.zeros((B, Nq), device=Q.device, dtype=Q.dtype)

        for i in range(Tq):
            q_start = i * Bq
            q_end = min(q_start + Bq, Nq)
            q = Q[..., q_start:q_end, :]
            o = torch.zeros((B, q_end - q_start, d), device=Q.device, dtype=Q.dtype)
            l = torch.zeros((B, q_end - q_start,), device=Q.device, dtype=Q.dtype)
            m = torch.full((B, q_end - q_start,), float("-inf"), device=Q.device, dtype=Q.dtype)

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
                    s = torch.where(causal_mask, s, float("-inf"))
                
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
        
        ctx.save_for_backward(L, Q, K, V, output)
        ctx.is_causal = is_causal
        return output
        

    @staticmethod
    def backward_origin(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        # shape of L is (B, Nq)
        # shape of Q is (B, Nq, d)
        # shape of K is (B, Nk, d)
        # shape of V is (B, Nk, d)
        # shape of O is (B, Nq, d)
        # shape of grad_output is (B, Nq, d)

        assert K.shape[-1] == V.shape[-1]
        scale = 1 / (Q.shape[-1] ** 0.5)
        S = Q @ K.transpose(-2, -1) * scale  # (B, Nq, Nk)
        if is_causal:
            causal_mask = torch.triu(torch.ones(Q.shape[1], K.shape[1], device=Q.device), diagonal=1)
            S = S.masked_fill(causal_mask == 1, float("-inf"))
        P = torch.exp(S - L.unsqueeze(-1))  # (B, Nq, Nk)
        dV = einsum(P, grad_output, "B Nq Nk, B Nq d -> B Nk d")
        dP = einsum(grad_output, V, "B Nq d, B Nk d -> B Nq Nk")
        D = (O * grad_output).sum(dim=-1, keepdim=True)  # (B, Nq, 1)
        dS = P * (dP - D)  # (B, Nq, Nk)
        dQ = einsum(dS, K, "B Nq Nk, B Nk d -> B Nq d") * scale
        dK = einsum(dS, Q, "B Nq Nk, B Nq d -> B Nk d") * scale
        return dQ, dK, dV, None
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, Nq, d = Q.shape
        Nk = K.shape[-2]

        Bq = 16
        Bk = 16
        assert d % Bq == 0 and d % Bk == 0
        Tq = (Nq + Bq - 1) // Bq
        Tk = (Nk + Bk - 1) // Bk

        scale = 1 / (d ** 0.5)
        D = (O * grad_output).sum(dim=-1)  # (B, Nq)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i in range(Tk):
            k_start = i * Bk
            k_end = min(k_start + Bk, Nk)
            k_tile = K[..., k_start:k_end, :]
            v_tile = V[..., k_start:k_end, :]
            dk_tile = torch.zeros_like(k_tile)
            dv_tile = torch.zeros_like(v_tile)

            for j in range(Tq):
                q_start = j * Bq
                q_end = min(q_start + Bq, Nq)
                q_tile = Q[..., q_start:q_end, :]
                # o_tile = O[..., q_start:q_end, :]
                do_tile = grad_output[..., q_start:q_end, :]
                dq_tile = dQ[..., q_start:q_end, :]

                s = einsum(q_tile, k_tile, "B Bq d, B Bk d -> B Bq Bk") * scale
                if is_causal:
                    causal_mask = torch.arange(q_start, q_end, device=s.device)[:, None] >= torch.arange(k_start, k_end, device=s.device)[None, :]
                    s = torch.where(causal_mask, s, float("-inf"))
                p = torch.exp(s - L[..., q_start:q_end, None])  # (B, Bq, Bk)
                dv_tile += einsum(p, do_tile, "B Bq Bk, B Bq d -> B Bk d")
                dp_tile = einsum(do_tile, v_tile, "B Bq d, B Bk d -> B Bq Bk")
                d_tile = D[..., q_start:q_end, None]  # (B, Bq, 1)
                ds_tile = p * (dp_tile - d_tile) * scale  # (B, Bq, Bk)
                dq_tile += einsum(ds_tile, k_tile, "B Bq Bk, B Bk d -> B Bq d")
                dk_tile += einsum(ds_tile, q_tile, "B Bq Bk, B Bq d -> B Bk d")
            
            dK[..., k_start:k_end, :] = dk_tile
            dV[..., k_start:k_end, :] = dv_tile
        
        return dQ, dK, dV, None
