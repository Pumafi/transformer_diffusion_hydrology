import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return self.scale * x / (norm + self.eps) + self.bias


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class Augmented_Arch_Model(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.nb_covariates = config["nb_covariates"]
        self.covariate_projection = Conv1d_with_init(self.nb_covariates, self.channels, 1)

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, covariates, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.selu(x)
        x = x.reshape(B, self.channels, K, L)

        covariates = covariates.reshape(B, self.nb_covariates, K * L)
        covariate_proj = self.covariate_projection(covariates)
        covariate_proj = F.selu(covariate_proj)
        covariate_proj = covariate_proj.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, covariate_proj, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.selu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x
    
class ScalingConv(nn.Module):
    def __init__(self, channels, filters):
        super().__init__()

        # Initial projection
        self.initial_conv = nn.Conv2d(channels, channels, (1, filters), padding="same")

        # Downsampling path
        self.downsample_pool_1 = nn.MaxPool2d((1, 2))
        self.downsample_conv_1 = nn.Conv2d(channels, channels, (1, filters), padding="same")

        self.downsample_pool_2 = nn.MaxPool2d((1, 2))
        self.downsample_conv_2 = nn.Conv2d(channels, channels*2, (1, filters), padding="same")

        # Upsampling path
        self.upsample_transpose_1 = nn.ConvTranspose2d(
            channels*2, channels, (1, filters), stride=(1, 2),
            padding=(0, filters // 2), output_padding=(0, 1)
        )
        self.upsample_transpose_2 = nn.ConvTranspose2d(
            channels, channels, (1, filters), stride=(1, 2),
            padding=(0, filters // 2), output_padding=(0, 1)
        )

        # Optional: Conv layers after concatenation to reduce artifacts
        self.conv_after_skip1 = nn.Conv2d(channels * 2, channels, (1, 3), padding="same")
        self.conv_after_skip2 = nn.Conv2d(channels * 2, channels, (1, 3), padding="same")

    def forward(self, x):
        # Input shape: [batch, channels, 1, time]
        x0 = F.selu(self.initial_conv(x))      # Initial conv

        # Encoder
        x1 = self.downsample_pool_1(x0)
        x1 = F.selu(self.downsample_conv_1(x1))

        x2 = self.downsample_pool_2(x1)
        x2 = F.selu(self.downsample_conv_2(x2))

        # Decoder with skip connections
        u1 = F.selu(self.upsample_transpose_1(x2))

        # Skip connection with x1
        if u1.shape[-1] != x1.shape[-1]:  # Handle possible size mismatch
            diff = x1.shape[-1] - u1.shape[-1]
            u1 = F.pad(u1, (0, diff))
        u1 = torch.cat([u1, x1], dim=1)   # Concat skip
        u1 = F.selu(self.conv_after_skip1(u1))

        u2 = F.selu(self.upsample_transpose_2(u1))

        # Skip connection with x0
        if u2.shape[-1] != x0.shape[-1]:  # Handle possible size mismatch
            diff = x0.shape[-1] - u2.shape[-1]
            u2 = F.pad(u2, (0, diff))
        u2 = torch.cat([u2, x0], dim=1)   # Concat skip
        u2 = F.selu(self.conv_after_skip2(u2))

        return u2



class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.chanels = channels
        self.input_conv = nn.Conv2d(channels, channels, (1, 3), padding="same")
        self.input_dropout = nn.Dropout(0.2)

        self.covariate_conv = nn.Conv2d(channels, channels, (1, 3), padding="same")

        self.covariates_mix = Conv1d_with_init(2*channels, channels, 1)

        self.initial_conv = nn.Conv2d(channels, channels, (1, 3), padding="same")
        self.followup_conv = nn.Conv2d(channels, channels, (1, 3), padding="same")
        self.input_norm = RMSNorm(channels)

        self.scaling_conv = ScalingConv(channels, 5)
        self.scaling_dropout = nn.Dropout(0.2)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.cond_dropout = nn.Dropout(0.1)

        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, covariates, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape

        y = self.input_conv(x)
        #y = self.input_dropout(y)
        #y = x

        covariates = self.covariate_conv(covariates)

        y = torch.concat([y, covariates], dim=1)
        y = y.reshape(B, 2*channel, K * L)
        y = self.covariates_mix(y)
        y = F.selu(y)
        y = y.reshape(B, channel, K, L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_proj = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        cond_proj = self.cond_dropout(cond_proj)
        cond_proj = cond_proj.reshape(B, 2*self.chanels, K, L)
        gamma, beta = torch.chunk(cond_proj, 2, dim=1)
        y = (1 + gamma) * y + beta

        y = self.scaling_conv(y)
        y = y.transpose(1, 3)
        y = self.input_norm(y)
        y = y.transpose(1, 3)
        y = self.scaling_dropout(y)
        y = F.selu(y)
        
        y = y.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = y + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)


        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        residual = self.initial_conv(residual)
        residual = F.selu(residual)

        skip = self.followup_conv(skip)

        return (x + residual) / math.sqrt(2.0), skip
