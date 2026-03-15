import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_spectral_norm, spectral_norm


# --- 1. RUNTIME PATCH FOR UPSTREAM IMPORTER ---
# This fixes the "AssertionError: Can not create literal tensor"
# by teaching the installed iree-compiler how to handle OCP types.
def patch_upstream_importer():
    print("[System] Patching upstream iree.compiler.extras.fx_importer...")
    try:
        from iree.compiler.extras import fx_importer
        
        # A. Patch MLIR Type Strings (Fixes "invalid dtype" in MLIR parser)
        # We map exotic floats to "ui8" (bytes) so the compiler accepts the bitstream.
        if hasattr(torch, "float8_e8m0fnu"):
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[torch.float8_e8m0fnu] = "ui8"
        if hasattr(torch, "float4_e2m1fn_x2"):
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[torch.float4_e2m1fn_x2] = "ui8"
            
        # B. Patch NumPy Mappings (Fixes "Can not create literal tensor")
        # We map exotic floats to compatible numpy integers to preserve bits.
        # This dictionary is what fx_importer checks before crashing.
        if hasattr(fx_importer, "TORCH_DTYPE_TO_NUMPY_DTYPE"):
            mapping = fx_importer.TORCH_DTYPE_TO_NUMPY_DTYPE
            
            # Fix BFloat16 (often missing in older versions)
            mapping[torch.bfloat16] = np.int16
            
            # Fix OCP FP8 Scales
            if hasattr(torch, "float8_e8m0fnu"):
                mapping[torch.float8_e8m0fnu] = np.uint8
                
            # Fix OCP FP4 Weights
            if hasattr(torch, "float4_e2m1fn_x2"):
                mapping[torch.float4_e2m1fn_x2] = np.uint8
                
        print("✅ Upstream importer patched successfully.")
        
    except ImportError:
        print("⚠️ Could not import fx_importer. Is iree-compiler installed?")
    except Exception as e:
        print(f"⚠️ Patch failed: {e}")

# Apply patch immediately
patch_upstream_importer()

# --- 2. STANDARD IMPORTS ---
import iree.turbine.aot as aot

try:
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    from torchao.quantization.quantize_.common import KernelPreference
    HAS_MX = True
except ImportError:
    HAS_MX = False
    print("⚠️ TorchAO not found. Exporting standard BF16.")

# --- 3. MODEL DEFINITION ---
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

# --- 4. MAIN EXPORT ROUTINE ---
def main():
    print("[Setup] Initializing ViT...")
    model = ViT()
    model = model.to(torch.bfloat16)

    # Cleanup Hooks
    for m in [model.nn_fc1, model.nn_fc2]:
        remove_spectral_norm(m)

    if HAS_MX:
        print("[Quantizer] Applying OCP MXFP8 Configuration...")
        mx_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.EMULATED 
        )
        int8_config = Int8WeightOnlyConfig()

        for name, mod in model.named_modules():
            if isinstance(mod, EfficientSelfAttention):
                print(f"   -> MX-FP8: {name}")
                quantize_(mod.keyValueExtractor, mx_config)
                quantize_(mod.query, mx_config)
                quantize_(mod.finalLayer, mx_config)
            elif isinstance(mod, MixFFN):
                print(f"   -> INT8:   {name}")
                quantize_(mod.mlp1, int8_config)
                quantize_(mod.mlp2, int8_config)
            elif name in ["decoder", "nn_fc1", "nn_fc2"]:
                quantize_(mod, int8_config)

    model.eval()
    
    # Dummy Inputs (BF16)
    input_img = torch.randn(1, 1, 48, 96, dtype=torch.bfloat16)
    input_state = torch.randn(1, 5, dtype=torch.bfloat16)
    
    print("\n[Export] Compiling to MLIR...")
    try:
        # AOT Export with relaxed strictness
        exported = aot.export(model, args=(input_img, input_state), strict_export=False)
        
        filename = "vit_quantized_mxfp8.mlir"
        exported.save_mlir(filename)
        print(f"✅ Success! MLIR saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Export Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()