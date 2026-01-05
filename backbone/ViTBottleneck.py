import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from backbone import MammothBackbone


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(MammothBackbone):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0
    ):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches+1, embed_dim)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return class token representation
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_tiny(num_classes: int = 10, img_size: int = 32, patch_size: int = 4):
    """
    Tiny ViT: suitable for CIFAR-10/100
    ~5M parameters for CIFAR-10
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    )


def vit_small(num_classes: int = 10, img_size: int = 32, patch_size: int = 4):
    """
    Small ViT: more capacity than tiny
    ~22M parameters for CIFAR-10
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    )


def vit_base(num_classes: int = 10, img_size: int = 224, patch_size: int = 16):
    """
    Base ViT: standard configuration for ImageNet
    ~86M parameters for ImageNet
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    )


def vit_large(num_classes: int = 10, img_size: int = 224, patch_size: int = 16):
    """
    Large ViT: higher capacity for complex datasets
    ~304M parameters for ImageNet
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0
    )


def test_vit(model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> None:
    batch_size = 32
    inputs = torch.randn(batch_size, *input_shape)

    model_map = {
        "vit_tiny": lambda: vit_tiny(num_classes, img_size=input_shape[1], 
                                      patch_size=4 if input_shape[1] == 32 else 16),
        "vit_small": lambda: vit_small(num_classes, img_size=input_shape[1],
                                        patch_size=4 if input_shape[1] == 32 else 16),
        "vit_base": lambda: vit_base(num_classes, img_size=input_shape[1], patch_size=16),
        "vit_large": lambda: vit_large(num_classes, img_size=input_shape[1], patch_size=16)
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(model_map.keys())}.")

    model = model_map[model_name]()
    model.eval()

    with torch.no_grad():
        outputs = model(inputs)

    # Verify output shape
    assert outputs.shape == (batch_size, num_classes), \
        f"Output shape mismatch: expected {(batch_size, num_classes)}, got {outputs.shape}"

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} test passed! Parameters: {num_params:,}")


if __name__ == "__main__":
    # Test ViT models with CIFAR-10 configuration (3 channels, 32x32 input, 10 classes)
    print("Testing ViT implementations for CIFAR-10:")
    test_vit(model_name="vit_tiny", input_shape=(3, 32, 32), num_classes=10)
    test_vit(model_name="vit_small", input_shape=(3, 32, 32), num_classes=10)

    # Test with ImageNet-like configuration
    print("\nTesting ViT implementations for ImageNet-like inputs:")
    test_vit(model_name="vit_base", input_shape=(3, 224, 224), num_classes=1000)
    test_vit(model_name="vit_large", input_shape=(3, 224, 224), num_classes=1000)

