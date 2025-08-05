import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def affine(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super(DiTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).chunk(
            6, dim=1
        )
        x = x + self.attn(affine(self.norm1(x), gamma1, beta1)) * alpha1.unsqueeze(1)
        x = x + self.mlp(affine(self.norm2(x), gamma2, beta2)) * alpha2.unsqueeze(1)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        gamma, beta = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.linear(affine(self.norm_final(x), gamma, beta))
        return x


class PatchEmbd(nn.Module):
    def __init__(self, in_channels, image_size, dim, patch_size):
        super(PatchEmbd, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (image_size // self.patch_size) ** 2
        self.conv_project = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_tuple,
            stride=patch_tuple,
            bias=False,
            padding_mode="zeros",
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_project(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


def pos_embd1d(pos, dim, depth=10000):
    omega = 1 / depth ** (torch.arange(dim // 2) / dim * 2)
    pos = pos.reshape(-1, 1)
    pos = pos * omega.reshape(1, -1)
    emb_sin = torch.sin(pos)
    emb_cos = torch.cos(pos)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def time_emb(t, dim):
    t = t * 1000
    freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
    sin_emb = torch.sin(t[:, None] / freqs)
    cos_emb = torch.cos(t[:, None] / freqs)
    return torch.cat([sin_emb, cos_emb], dim=-1)


class DiTModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=28,
        hidden_size=32,
        patch_size=2,
        out_channels=1,
        num_heads=4,
        mlp_ratio=4,
        num_layers=4,
    ):
        super(DiTModel, self).__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_embd = PatchEmbd(
            in_channels=in_channels,
            image_size=image_size,
            dim=hidden_size,
            patch_size=patch_size,
        )
        self.dits = [
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)
        ]
        [self.register_module(f"DiTBlock{i}", dit) for i, dit in enumerate(self.dits)]
        self.out_layer = FinalLayer(hidden_size, patch_size, out_channels)

    def _unpatchify(self, x, patch_size=(2, 2), height=32, width=32):
        bs, num_patches, patch_dim = x.shape
        H, W = patch_size
        in_channels = patch_dim // (H * W)
        num_patches_h, num_patches_w = height // H, width // W
        x = x.view(bs, in_channels, num_patches_h, num_patches_w, H, W)
        x = x.transpose(3, 4).contiguous()
        reconstructed = x.view(
            bs,
            in_channels,
            height,
            width,
        )
        return reconstructed

    def forward(self, x, c=None):
        x = self.patch_embd(x)
        pos_embed = pos_embd1d(
            torch.arange(self.patch_embd.num_patches), self.hidden_size
        )
        x += pos_embed.to(x.device).unsqueeze(0)
        if c is None:
            c = torch.zeros(1, self.hidden_size, device=x.device)
        else:
            c = time_emb(c, self.hidden_size)
        for dit in self.dits:
            x = dit(x, c)
        x = self.out_layer(x, c)
        x = self._unpatchify(
            x, [self.patch_embd.patch_size] * 2, self.image_size, self.image_size
        )
        return x


if __name__ == "__main__":
    device = "cuda"
    model = DiTModel(1, 28, 64, 2, 1, 4, 4, 4)
    model.to(device)
    print(model(torch.zeros(16,1,28,28,device=device)).shape)
    print(model(torch.zeros(16,1,28,28,device=device),torch.ones(16,device=device)).shape)