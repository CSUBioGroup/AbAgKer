from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgMixPooler(nn.Module):
    """Fuse convolutional and SSF token scores for fixed-length antigen pooling."""

    def __init__(self, target_len: int, window_T: int = 7, pooling: str = "segment", bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ("segment", "topk")
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        self.conv_attn: nn.Conv2d | None = None
        self.ssf_weight = nn.Parameter(torch.randn(7))
        self.ssf_bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.gate_logit = nn.Parameter(torch.zeros(1))
        self.bias = bias

    def _build_conv_if_needed(self, E: int, device, dtype):
        """Instantiate token-axis convolution after observing embedding width E."""
        if self.conv_attn is None:
            pad_T = self.window_T // 2
            self.conv_attn = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(self.window_T, E),
                stride=(1, 1),
                padding=(pad_T, 0),
                bias=self.bias,
            ).to(device=device, dtype=dtype)

    def _conv_weights(self, l_full_embs: torch.Tensor) -> torch.Tensor:
        """Compute convolutional token scores from antigen embeddings [B, T, E]."""
        B, T, E = l_full_embs.shape
        self._build_conv_if_needed(E, l_full_embs.device, l_full_embs.dtype)
        x = l_full_embs.unsqueeze(1)
        y = self.conv_attn(x)
        return y.squeeze(1)

    def _ssf_weights(self, ssf_x: torch.Tensor) -> torch.Tensor:
        """Compute scalar token scores from SSF features [B, T, 7]."""
        weights = torch.einsum("bti,i->bt", ssf_x, self.ssf_weight)
        if self.ssf_bias is not None:
            weights = weights + self.ssf_bias
        return weights.unsqueeze(-1)

    def _fuse_and_normalize(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Fuse two score streams and normalize over tokens with softmax."""
        alpha = torch.sigmoid(self.gate_logit)
        w = alpha * w1 + (1 - alpha) * w2
        a = torch.tanh(w).squeeze(-1)
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float("-inf"))
        return F.softmax(a, dim=-1).unsqueeze(-1)

    def _segment_pool(
        self,
        x: torch.Tensor,
        attn: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Pool [B, T, E] into [B, L, E] by weighted averaging within uniform segments."""
        B, T, E = x.shape
        L = self.target_len

        idx = torch.arange(T, device=x.device)
        seg = torch.clamp((idx * L) // max(T, 1), 0, L - 1)
        seg_onehot = F.one_hot(seg, num_classes=L).to(dtype=x.dtype)
        seg_onehot = seg_onehot.unsqueeze(0).expand(B, -1, -1)

        if padding_mask is not None:
            attn = attn * padding_mask.unsqueeze(-1).float()

        seg_w = torch.bmm(attn.transpose(1, 2), seg_onehot).transpose(1, 2)
        weighted_x = x * attn
        seg_feat = torch.einsum("bte, btl -> ble", weighted_x, seg_onehot)
        return seg_feat / seg_w.clamp_min(1e-8)

    def _topk_pool(
        self,
        x: torch.Tensor,
        attn: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Select top-K tokens by attention score and preserve their sequence order."""
        B, T, E = x.shape
        K = min(self.target_len, T)
        a = attn.squeeze(-1)
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float("-inf"))

        _, idx = torch.topk(a, k=K, dim=-1, largest=True, sorted=True)
        idx_sorted, _ = torch.sort(idx, dim=-1)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, K)
        return x[batch_idx, idx_sorted, :]

    def forward(
        self,
        l_full_embs: torch.Tensor,
        ssf_x: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Compute pooled antigen features.

        Args:
            l_full_embs (Tensor): shape [B, T, E], antigen token embeddings.
            ssf_x (Tensor): shape [B, T, 7], auxiliary sequence-level features.
            padding_mask (BoolTensor, optional): shape [B, T], True for valid tokens.

        Returns:
            Tuple[Tensor, Tensor]: pooled features [B, L_out, E] and token weights [B, T, 1].
        """
        assert l_full_embs.dim() == 3 and ssf_x.dim() == 3, "Expected l_full_embs [B,T,E] and ssf_x [B,T,7]."
        B, T, _ = l_full_embs.shape
        assert ssf_x.shape[:2] == (B, T) and ssf_x.shape[2] == 7, "ssf_x must have shape [B,T,7]."

        w_conv = self._conv_weights(l_full_embs)
        w_ssf = self._ssf_weights(ssf_x)
        attn = self._fuse_and_normalize(w_conv, w_ssf, padding_mask)

        if self.pooling == "segment":
            pooled = self._segment_pool(l_full_embs, attn, padding_mask)
        else:
            pooled = self._topk_pool(l_full_embs, attn, padding_mask)

        return pooled, attn


class AgMixPooler_noSSF(nn.Module):
    """Token reduction variant that uses learned projection without SSF features."""

    def __init__(self, target_len: int, window_T: int = 7, pooling: str = "segment", bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ("segment", "topk")
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        self.token_reduce_layer = nn.Sequential(
            nn.Linear(896, 256),
            nn.GELU(),
            nn.Linear(256, target_len),
            nn.GELU(),
        )
        self.ln1 = nn.LayerNorm(512)

    def forward(self, l_full_embs: torch.Tensor, ssf_x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None):
        pooled = self.token_reduce_layer(l_full_embs.transpose(1, 2)).transpose(1, 2)
        return self.ln1(pooled), None


class AgMixPooler_1206(nn.Module):
    """Top-k pooling variant with an embedding bottleneck before score generation."""

    def __init__(self, target_len: int, window_T: int = 7, pooling: str = "topk", bias: bool = True):
        super().__init__()
        assert target_len > 0
        assert pooling in ("segment", "topk")
        self.target_len = target_len
        self.window_T = window_T
        self.pooling = pooling

        self.dim = 512
        self.ab_token_reduce_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.dim // 8),
        )
        self.conv_attn = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.window_T, self.dim // 8),
            stride=(1, 1),
            padding=(self.window_T // 2, 0),
            bias=True,
        )

        self.ssf_weight = nn.Parameter(torch.randn(7))
        self.ssf_bias = nn.Parameter(torch.zeros(1)) if bias else None
        self.gate_logit = nn.Parameter(torch.zeros(1))
        self.bias = bias

    def _conv_weights(self, l_full_embs: torch.Tensor) -> torch.Tensor:
        x = l_full_embs.unsqueeze(1)
        y = self.conv_attn(x)
        return y.squeeze(1)

    def _ssf_weights(self, ssf_x: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum("bti,i->bt", ssf_x, self.ssf_weight)
        if self.ssf_bias is not None:
            weights = weights + self.ssf_bias
        return weights.unsqueeze(-1)

    def _fuse_and_normalize(self, w1: torch.Tensor, w2: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate_logit)
        a = torch.tanh(alpha * w1 + (1 - alpha) * w2).squeeze(-1)
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float("-inf"))
        return F.softmax(a, dim=-1).unsqueeze(-1)

    def _topk_pool(self, x: torch.Tensor, attn: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        B, T, E = x.shape
        K = min(self.target_len, T)
        a = attn.squeeze(-1)
        if padding_mask is not None:
            a = a.masked_fill(~padding_mask, float("-inf"))
        _, idx = torch.topk(a, k=K, dim=-1, largest=True, sorted=True)
        idx_sorted, _ = torch.sort(idx, dim=-1)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, K)
        return x[batch_idx, idx_sorted, :]

    def forward(self, l_full_embs: torch.Tensor, ssf_x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None):
        l_full_pooled = self.ab_token_reduce_layer(l_full_embs)
        w_conv = self._conv_weights(l_full_pooled)
        w_ssf = self._ssf_weights(ssf_x)
        attn = self._fuse_and_normalize(w_conv, w_ssf, padding_mask)
        pooled = self._topk_pool(l_full_embs, attn, padding_mask)
        return pooled, attn
