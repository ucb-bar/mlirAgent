import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- FIXED: Added missing import
from torch.nn.utils import remove_spectral_norm, spectral_norm
from torchao.quantization import Int8WeightOnlyConfig, quantize_

try:
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.quantization.quantize_.common import KernelPreference
    HAS_MX = True
except ImportError:
    HAS_MX = False
    print("⚠️ TorchAO MX prototypes not found.")

# --- MODEL (Same Definition) ---
class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        self.layerNorm = nn.LayerNorm(out_channels)
    def forward(self, patches):
        x = self.cn1(patches)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layerNorm(x)
        return x, H, W

class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        self.heads = num_heads
        self.cn1 = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x1 = x.permute(0, 2, 1).reshape(B, C, H, W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        x1 = self.ln1(x1)
        keyVal = self.keyValueExtractor(x1)
        k, v = keyVal.chunk(2, dim=-1)
        q = self.query(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = self.smax(scores)
        out = torch.matmul(attn, v)
        return self.finalLayer(out)

class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels * expansion_factor
        self.mlp1 = nn.Linear(channels, expanded_channels)
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding='same', groups=channels)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(expanded_channels, channels)
    def forward(self, x, H, W):
        x = self.mlp1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1, 2))
        x = self.mlp2(x)
        return x

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self.out_channels = out_channels
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])
    def forward(self, x):
        x, H, W = self.patchMerge(x)
        for i in range(len(self._attn)):
            x = x + self._attn[i](x, H, W)
            x = x + self._ffn[i](x, H, W)
            x = self._lNorm[i](x)
        x = x.reshape(x.shape[0], H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        self.decoder = nn.Linear(4608, 512)
        self.nn_fc1 = spectral_norm(nn.Linear(517, 256))
        self.nn_fc2 = spectral_norm(nn.Linear(256, 3))
    def forward(self, x, state_vector):
        for block in self.encoder_blocks:
            x = block(x)
        x = x.flatten(1)
        x = self.decoder(x)
        x = torch.cat([x, state_vector], dim=1)
        x = F.relu(self.nn_fc1(x))
        x = self.nn_fc2(x)
        return x

def main():
    print("[1] Initializing Model...")
    model = ViT().to(torch.bfloat16)
    
    # Remove hooks
    for m in [model.nn_fc1, model.nn_fc2]:
        remove_spectral_norm(m)

    print("[2] Quantizing (MX-FP8 + INT8)...")
    if HAS_MX:
        mx_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.EMULATED 
        )
        int8_config = Int8WeightOnlyConfig()

        for name, mod in model.named_modules():
            if isinstance(mod, EfficientSelfAttention):
                quantize_(mod.keyValueExtractor, mx_config)
                quantize_(mod.query, mx_config)
                quantize_(mod.finalLayer, mx_config)
            elif isinstance(mod, MixFFN):
                quantize_(mod.mlp1, int8_config)
                quantize_(mod.mlp2, int8_config)
            elif name in ["decoder", "nn_fc1", "nn_fc2"]:
                quantize_(mod, int8_config)

    input_img = torch.randn(1, 1, 48, 96, dtype=torch.bfloat16)
    input_state = torch.randn(1, 5, dtype=torch.bfloat16)

    print("[3] Running Eager Mode (Validation)...")
    with torch.no_grad():
        out_eager = model(input_img, input_state)
    print(f"    Eager Output Shape: {out_eager.shape}")

    print("[4] Running torch.compile (Backend='eager')...")
    # We use backend='eager' to test graph capture without requiring Inductor 
    # to support the specific OCP CPU kernels (which caused the crash before).
    opt_model = torch.compile(model, backend="eager", fullgraph=True)
    
    with torch.no_grad():
        out_compiled = opt_model(input_img, input_state)
    print(f"    Compiled Output Shape: {out_compiled.shape}")
    print("✅ torch.compile (Graph Capture) Successful!")

if __name__ == "__main__":
    main()