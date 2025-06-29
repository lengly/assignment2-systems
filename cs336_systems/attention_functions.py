import torch
from torch import Tensor
import math
from jaxtyping import Float, Bool, Int
from einops import einsum
import torch.nn.functional as F

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 1, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    
    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def custom_attention_implementation(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Custom attention implementation for comparison.
    This is a basic implementation that can be extended with more sophisticated attention mechanisms.
    """
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask == 1, scores, float("-inf"))
        
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, V)
    
    return output


def pytorch_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
