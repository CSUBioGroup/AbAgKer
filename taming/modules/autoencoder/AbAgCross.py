import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Standard multi-head cross-attention from query tokens to key/value tokens."""

    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in, kv_in, kv_pad_mask=None):
        """
        Args:
            q_in (Tensor): shape [B, Tq, C], query sequence.
            kv_in (Tensor): shape [B, Tk, C], key/value sequence.
            kv_pad_mask (Tensor, optional): shape [B, Tk], 1 for valid and 0 for padding.

        Returns:
            Tuple[Tensor, Tensor]: context [B, Tq, C] and attention map [B, H, Tq, Tk].
        """
        B, Tq, C = q_in.shape
        Tk = kv_in.size(1)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q_in).view(B, Tq, H, D).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Tk, H, D).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Tk, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if kv_pad_mask is not None:
            scores = scores.masked_fill(kv_pad_mask[:, None, None, :] == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.proj_drop(self.out(ctx)), attn


class FFN(nn.Module):
    """Two-layer feed-forward network used after cross-attention updates."""

    def __init__(self, dim, hidden, drop=0.1, act_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act_drop = nn.Dropout(act_drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.act_drop(x)
        x = self.fc2(x)
        return self.drop(x)


class CoAttentionBlock(nn.Module):
    """Bidirectional antibody-antigen co-attention block with pre-norm residuals."""

    def __init__(self, dim, num_heads, ffn_dim, drop=0.1, attn_drop=0.1, act_drop=0.1):
        super().__init__()
        self.ln_a1 = nn.LayerNorm(dim)
        self.ln_a2 = nn.LayerNorm(dim)
        self.ln_a3 = nn.LayerNorm(dim)
        self.ca_a_from_g = CrossAttention(dim, num_heads, attn_drop, drop)
        self.ca_g_from_a = CrossAttention(dim, num_heads, attn_drop, drop)

        self.ln_a21 = nn.LayerNorm(dim)
        self.ln_a22 = nn.LayerNorm(dim)
        self.ffn_a = FFN(dim, ffn_dim, drop, act_drop)
        self.ffn_g = FFN(dim, ffn_dim, drop, act_drop)

        self.ln_ab_out = nn.LayerNorm(dim)
        self.ln_ag_out = nn.LayerNorm(dim)

    def forward(self, x_ab, x_ag, ag_all=None, pad_b=None, pad_g=None):
        """
        Args:
            x_ab (Tensor): shape [B, Ta, C], antibody token features.
            x_ag (Tensor): shape [B, Tg, C], antigen token features.
            ag_all (Tensor, optional): shape [B, Tga, C], alternative antigen context.
            pad_b (Tensor, optional): shape [B, Ta], antibody validity mask.
            pad_g (Tensor, optional): shape [B, Tg] or [B, Tga], antigen validity mask.

        Returns:
            Tuple[Tensor, Tensor]: updated antibody and antigen features.
        """
        xb_norm = self.ln_a1(x_ab)
        xg_norm = self.ln_a2(x_ag)
        xg_all_norm = self.ln_a3(ag_all) if ag_all is not None else xg_norm

        delta_ab, _ = self.ca_a_from_g(xb_norm, xg_all_norm, kv_pad_mask=pad_g)
        delta_ag, _ = self.ca_g_from_a(xg_norm, xb_norm, kv_pad_mask=pad_b)

        x_ab_new = x_ab + delta_ab
        x_ag_new = x_ag + delta_ag

        x_ab_final = x_ab_new + self.ffn_a(self.ln_a21(x_ab_new))
        x_ag_final = x_ag_new + self.ffn_g(self.ln_a22(x_ag_new))
        return x_ab_final, x_ag_final
