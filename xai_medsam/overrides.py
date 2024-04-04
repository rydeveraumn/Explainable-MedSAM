# stdlib
import math

# third party
import torch
from torch import Tensor


def ViTAttention_forward_override(self, x):
    """
    Override for the forward calculation of the ViT Attention mechanism.

    The input x has shape (B, N, C)
    """
    # Get the shapes
    B, N, _ = x.shape

    # Normalization
    x = self.norm(x)

    # Query-key-value
    qkv = self.qkv(x)

    # (B, N, num_heads, d)
    q, k, v = qkv.view(B, N, self.num_heads, -1).split(
        [self.key_dim, self.key_dim, self.d], dim=3
    )
    # (B, num_heads, N, d)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) * self.scale + (
        self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
    )
    attn = attn.softmax(dim=-1)
    self.attention_map = attn.detach()
    x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
    x = self.proj(x)

    return x


def SamAttention_forward_override(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    Override function for the SamAttention
    """
    # Input projections
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    # Attention
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)
    self.attention_map = attn.detach()

    # Get output
    out = attn @ v
    out = self._recombine_heads(out)
    out = self.out_proj(out)

    return out
