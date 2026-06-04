import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm1d(nn.Module):
    """
    RMSNorm for 1D conv features: (B, C, L)
    """
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.norm = nn.RMSNorm(channels, eps=eps)

    def forward(self, x):
        # x: (B, C, L) -> RMSNorm expects last dim
        x = x.permute(0, 2, 1)        # (B, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)        # back to (B, C, L)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,)
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        """
        x: (B, C, L) -> we'll transpose to (B, L, C) for attention
        """
        B, C, L = x.shape
        x_ = x.permute(0, 2, 1)  # (B, L, C)
        x_norm = self.norm(x_)

        qkv = self.qkv(x_norm)  # (B, L, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # split heads
        q = q.view(B, L, self.num_heads, C // self.num_heads).transpose(1,2)  # (B, heads, L, dim_head)
        k = k.view(B, L, self.num_heads, C // self.num_heads).transpose(1,2)
        v = v.view(B, L, self.num_heads, C // self.num_heads).transpose(1,2)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(C // self.num_heads)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # (B, heads, L, dim_head)

        out = out.transpose(1,2).contiguous().view(B, L, C)
        out = self.proj(out)
        return (x_ + out).permute(0,2,1)  # back to (B, C, L)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, in_channels)
        self.cond_proj = nn.Conv1d(cond_channels, in_channels, 1)

        self.norm1 = RMSNorm1d(in_channels)
        self.norm2 = RMSNorm1d(in_channels)
        if use_attention:
            self.attn = SelfAttention1D(in_channels)

    def forward(self, x, cond, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(t_emb)[:, :, None]
        h = h + t
        h = h + self.cond_proj(cond)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.use_attention:
            h = self.attn(h)

        return x + h
    
class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels,
        cond_channels,
        base_channels=64,
        time_emb_dim=128,
        num_res_blocks=4,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                base_channels,
                cond_channels,
                time_emb_dim
            )
            for _ in range(num_res_blocks)
        ])

        self.final_conv = nn.Sequential(
            RMSNorm1d(base_channels),
            #nn.SiLU(),
            nn.Conv1d(base_channels, in_channels, 1, padding=0)
        )

    def forward(self, input_data):
        """
        input_data = (noise, conditional, mask, diffusion_steps)
        """
        noise, conditional, mask, diffusion_steps = input_data

        conditional = torch.cat([conditional, mask.float()], dim=1)

        # time embedding
        t_emb = self.time_embedding(diffusion_steps)

        # UNet
        x = noise

        x = self.init_conv(x)

        for block in self.res_blocks:
            x = block(x, conditional, t_emb)

        y = self.final_conv(x)

        return y