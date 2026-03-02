import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network used in transformer blocks."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.act_drop = nn.Dropout(activation_dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.act_drop(x)
        x = self.linear2(x)
        return self.out_drop(x)


class CDRsAttention(nn.Module):
    """Multi-head self-attention with optional CDR-restricted heads.

    A prefix of heads can be forced to attend only to CDR keys. If all keys are
    masked in a sample, the corresponding CDR mask is disabled for stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cdr_heads: int = 0,
        attention_dropout: float = 0.1,
        dropout: float = 0.1,
        return_attention_weights: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.cdr_heads = cdr_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.return_attention_weights = return_attention_weights

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attention_dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None, cdrs_score=None):
        """
        Compute CDR-aware self-attention.

        Args:
            x (Tensor): shape [B, T, C], token features.
            mask (Tensor, optional): shape [B, T], 1 for valid tokens and 0 for padding.
            cdrs_score (Tensor, optional): shape [B, T], 1 for CDR tokens and 0 otherwise.

        Returns:
            Tensor | Tuple[Tensor, Dict[str, Tensor]]: output shape [B, T, C], and
            optional attention diagnostics.
        """
        B, T, C = x.size()
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        if mask is not None:
            padding_mask = mask[:, None, None, :] == 0
            scores = scores.masked_fill(padding_mask, float("-inf"))

        if cdrs_score is not None and self.cdr_heads > 0:
            cdr_key_mask = cdrs_score[:, None, None, :] == 0
            is_all_masked = cdr_key_mask.all(dim=-1, keepdim=True)
            cdr_key_mask = cdr_key_mask.masked_fill(is_all_masked, False)
            scores[:, : self.cdr_heads, :, :] = scores[:, : self.cdr_heads, :, :].masked_fill(
                cdr_key_mask.expand(-1, self.cdr_heads, T, -1),
                float("-inf"),
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_drop(self.out_proj(ctx))

        if self.return_attention_weights:
            return output, {
                "attention_weights": attn,
                "cdr_mask_info": cdrs_score,
                "padding_mask": mask,
            }

        return output


class AbEncoderLayer(nn.Module):
    """Transformer encoder layer where half of heads are CDR-constrained."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = CDRsAttention(
            dim,
            num_heads,
            cdr_heads=num_heads // 2,
            attention_dropout=attention_dropout,
            dropout=dropout,
            return_attention_weights=True,
        )
        self.ffn = FeedForward(
            dim,
            ff_dim,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, cdrs_score=None):
        att_output, attention_info = self.self_attn(self.norm1(x), mask=mask, cdrs_score=cdrs_score)
        x = x + att_output
        output = x + self.ffn(self.norm2(x))
        return output, attention_info


class AbEncoderLayer_noCDRhead(nn.Module):
    """Transformer encoder layer without CDR-specific attention heads."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = CDRsAttention(
            dim,
            num_heads,
            cdr_heads=0,
            attention_dropout=attention_dropout,
            dropout=dropout,
            return_attention_weights=True,
        )
        self.ffn = FeedForward(
            dim,
            ff_dim,
            dropout=dropout,
            activation_dropout=activation_dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, cdrs_score=None):
        att_output, attention_info = self.self_attn(self.norm1(x), mask=mask, cdrs_score=cdrs_score)
        x = x + att_output
        output = x + self.ffn(self.norm2(x))
        return output, attention_info


class AbPooler(nn.Module):
    """Pool antibody token features using attention-derived token importance."""

    def __init__(self, pooling_method: str = "mean", topk: int = 64):
        super().__init__()
        self.pooling_method = pooling_method
        self.topk = topk

    def forward(self, x: torch.Tensor, attention_info):
        """
        Build token importance from self-attention and pool sequence features.

        Args:
            x (Tensor): shape [B, T, C], encoded antibody tokens.
            attention_info (Dict[str, Tensor]): attention diagnostics from encoder.

        Returns:
            Tensor: pooled representation, shape [B, K, C] or [B, 1, C].
        """
        attn_weights = attention_info["attention_weights"]
        valid_mask = attention_info["padding_mask"]

        H = attn_weights.shape[1]
        self_attention = torch.diagonal(attn_weights, dim1=-2, dim2=-1).sum(dim=1) / H
        cross_attention = attn_weights.sum(dim=1).sum(dim=1) / H
        attention_importance = self_attention + cross_attention

        if valid_mask is not None:
            attention_importance = attention_importance.masked_fill(valid_mask == 0, float("-inf"))

        return self.pool_tokens(x, attention_importance, valid_mask)

    def pool_tokens(self, x: torch.Tensor, importance_scores: torch.Tensor, valid_mask):
        """
        Pool token features from per-token importance scores.

        Args:
            x (Tensor): shape [B, T, C], token features.
            importance_scores (Tensor): shape [B, T], token saliency scores.
            valid_mask (Tensor, optional): shape [B, T], 1 for valid tokens.

        Returns:
            Tensor: pooled tensor depending on the configured strategy.
        """
        if self.pooling_method == "topk":
            k = min(self.topk, x.shape[1])
            _, topk_indices = torch.topk(importance_scores, k, dim=1)
            batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, k)
            return x[batch_indices, topk_indices]

        if self.pooling_method == "weighted":
            valid_scores = torch.where(
                torch.isfinite(importance_scores),
                importance_scores,
                torch.tensor(0.0, device=importance_scores.device),
            )
            valid_scores = F.relu(valid_scores)
            weights = F.softmax(valid_scores, dim=1)
            if valid_mask is not None:
                weights = weights * valid_mask.float()
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            pooled_x = torch.sum(x * weights.unsqueeze(-1), dim=1)
            return pooled_x.unsqueeze(1)

        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).float()
            pooled_x = torch.sum(x * mask_expanded, dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            pooled_x = x.mean(dim=1)
        return pooled_x.unsqueeze(1)
