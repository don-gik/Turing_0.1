import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, einsum, nn
from torchsummary import summary


class ResidualAdd(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor, **kwargs):
        res = x
        x = self.layer(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Sequential):
    def __init__(self, embed_size: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__(
            nn.Linear(embed_size, expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embed_size, embed_size),
        )


class PatchEmbedding(nn.Module):
    """
    Patch Embedding block for ViT encoder.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_H: int,
        img_W: int,
    ) -> None:
        super().__init__()

        embed_dim = in_channels * (patch_size**2)
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c (h) (w) -> b (h w) c"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position = nn.Parameter(torch.randn(img_H * img_W // (patch_size**2) + 1, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position

        return x


class SelfAttention(nn.Module):
    """
    Simple self attention module.
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.heads = heads
        self.inv_sqrt_dim: float = (embed_dim) ** (-0.5)

        self.weight_k = nn.Linear(embed_dim, embed_dim)
        self.weight_q = nn.Linear(embed_dim, embed_dim)
        self.weight_v = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(embed_dim, embed_dim)

        self.last_attention: Tensor | None = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        q: Tensor = rearrange(self.weight_q(x), "b n (h d) -> b h n d", h=self.heads)
        k: Tensor = rearrange(self.weight_q(x), "b n (h d) -> b h n d", h=self.heads)
        v: Tensor = rearrange(self.weight_v(x), "b n (h d) -> b h n d", h=self.heads)

        energy = einsum("bhqd, bhkd -> bhqk", q, k)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(mask, fill_value)

        attention = F.softmax(energy * self.inv_sqrt_dim, dim=-1)
        attention = self.attn_dropout(attention)

        self.last_attention = attention.detach()

        out = einsum("bhqk, bhkv -> bhqv", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.projection(out)


class TransformerBlock(nn.Sequential):
    def __init__(
        self, embed_size: int, dropout: float = 0.1, expansion: int = 4, forward_dropout: float = 0.1, **kwargs
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(embed_size), SelfAttention(embed_size, **kwargs), nn.Dropout(dropout))
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(embed_size), FeedForward(embed_size, expansion, forward_dropout), nn.Dropout(dropout)
                )
            ),
        )


class Encoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class Model(nn.Module):
    """
    Basic model using ViT as a base model
    It enables the full network to catch details,
    find out relationships between variables in ENSO images.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        img_H: int,
        img_W: int,
        depth: int,
        n_classes: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, img_H=img_H, img_W=img_W
        )
        self.encoder = Encoder(depth, embed_size=in_channels * (patch_size) ** 2, **kwargs)
        self.decoder = ClassificationHead(
            emb_size=in_channels * (patch_size) ** 2, n_classes=n_classes
        )

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


if __name__ == "__main__":
    model: nn.Module = Model(in_channels=9, patch_size=4, time=8, img_H=40, img_W=200, depth=6, n_classes=3)

    summary(model, (9, 8, 40, 200), device="cpu")
