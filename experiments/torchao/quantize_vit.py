import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, remove_spectral_norm
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# --- 1. SETUP MX CONFIGURATION ---
try:
    import torchao.prototype.mx_formats
    from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
    from torchao.quantization.quantize_.common import KernelPreference
    HAS_MX = True
    print("[System] torchao MX workflows loaded.")
except ImportError as e:
    HAS_MX = False
    print(f"[Warning] MX workflows not found: {e}. Falling back to simulation.")

# --- 2. MODEL DEFINITION ---

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
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."
        self.heads = num_heads
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Projections (Target for FP8)
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
        keyVal = keyVal.reshape(B, -1, 2, self.heads, int(C / self.heads)).permute(2, 0, 3, 1, 4).contiguous()
        k, v = keyVal[0], keyVal[1]
        
        q = self.query(x).reshape(B, N, self.heads, int(C / self.heads)).permute(0, 2, 1, 3).contiguous()
        dimHead = (C / self.heads) ** 0.5
        attention = self.smax(q @ k.transpose(-2, -1) / dimHead)
        attention = (attention @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.finalLayer(attention)
        return x

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

    def forward(self, x, state_vector=None):
        for block in self.encoder_blocks:
            x = block(x)
        x = x.flatten(1)
        x = self.decoder(x)
        if state_vector is None:
            state_vector = torch.zeros(x.shape[0], 5, device=x.device, dtype=x.dtype)
        x = torch.cat([x, state_vector], dim=1)
        x = F.relu(self.nn_fc1(x))
        x = self.nn_fc2(x)
        return x

# --- 3. APPLY HYBRID CONFIGURATION ---

def apply_quantization_configs(model):
    print("\n[Quantizer] Configuring Hybrid Quantization...")
    
    # Remove Spectral
    # Must be done BEFORE quantization and export
    # We do this while still in FP32 to preserve maximum weight precision before casting
    for module in [model.nn_fc1, model.nn_fc2]:
        remove_spectral_norm(module)

    # Cast to BF16
    model = model.to(torch.bfloat16)

    # 3. Define Configs
    mx_config = None
    if HAS_MX:
        mx_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO,
        )
    
    int8_config = Int8DynamicActivationInt8WeightConfig()

    # Apply to specific layers
    for name, mod in model.named_modules():
        
        # Attention Layers -> FP8 (MX Config)
        if isinstance(mod, EfficientSelfAttention):
            if HAS_MX and mx_config:
                print(f"   -> MX-FP8 (E4M3): {name}")
                quantize_(mod.keyValueExtractor, mx_config)
                quantize_(mod.query, mx_config)
                quantize_(mod.finalLayer, mx_config)
        
        # Feed Forward / MLP -> INT8
        elif isinstance(mod, MixFFN):
            print(f"   -> INT8:          {name}")
            quantize_(mod.mlp1, int8_config)
            quantize_(mod.mlp2, int8_config)

    #  Classifier Heads -> INT8
    print(f"   -> INT8:          decoder & heads")
    quantize_(model.decoder, int8_config)
    quantize_(model.nn_fc1, int8_config)
    quantize_(model.nn_fc2, int8_config)

    return model

if __name__ == "__main__":
    print("Initializing ViT...")
    model = ViT()
    
    # Apply Quantization
    with torch.no_grad():
        model = apply_quantization_configs(model)
    
    print("\n[Compilation] Compiling graph...")
    # fullgraph=True required for export to IREE/MLIR
    model = torch.compile(model, fullgraph=True)

    print("\n[Simulation] Running Inference...")
    dummy_input = torch.randn(1, 1, 48, 96, dtype=torch.bfloat16)
    dummy_state = torch.randn(1, 5, dtype=torch.bfloat16)
    
    try:
        with torch.no_grad():
            output = model(dummy_input, dummy_state)
        
        print(f"   Output Shape: {output.shape}")
        print(f"   Output Sample: {output[0].float().cpu().numpy()}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()