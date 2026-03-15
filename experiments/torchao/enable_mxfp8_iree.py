import iree.turbine.support.conversions as conversions
import numpy as np
import torch
from iree.compiler.extras import fx_importer
from iree.runtime import HalElementType


def enable_mxfp8_support():
    """
    Patches IREE Turbine to support OCP MXFP8/MXFP4 data types.
    """
    print("[System] Enabling MXFP8/MXFP4 support in IREE Turbine (Bitcast Mode)...")

    # --- 1. Define Missing Torch Dtypes (if needed) ---
    if not hasattr(torch, "float8_e8m0fnu"):
        # If it doesn't exist, we can't really do much, but we proceed assuming nightly torch
        print("   -> [Warning] torch.float8_e8m0fnu not found in torch.")

    # --- 2. Patch MLIR Assembly Mappings (CRITICAL CHANGE) ---
    # We map E8M0 to "ui8" (unsigned int 8) because the C++ torch dialect parser 
    # rejects "f8E8M0FNU". 'ui8' preserves the bits perfectly.
    if hasattr(torch, "float8_e8m0fnu"):
        fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[torch.float8_e8m0fnu] = "ui8"
        print("   -> Patched FX Importer: torch.float8_e8m0fnu -> ui8 (bit preservation)")

    # FP4 E2M1 is packed 2-elements-per-byte. 
    # We map this to "ui8" (or "i8") as well, since it's just a byte container.
    if hasattr(torch, "float4_e2m1fn_x2"):
        fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM[torch.float4_e2m1fn_x2] = "ui8"
        print("   -> Patched FX Importer: torch.float4_e2m1fn_x2 -> ui8 (packed)")

    # --- 3. Patch IREE Turbine Conversions ---
    
    # Map for serialization/constants
    if hasattr(torch, "float8_e8m0fnu"):
        conversions.TORCH_DTYPE_TO_IREE_TYPE_ASM[torch.float8_e8m0fnu] = "i8" # IREE uses i8 generally
        # We don't necessarily need the inverse for export
    
    if hasattr(torch, "float4_e2m1fn_x2"):
        conversions.TORCH_DTYPE_TO_IREE_TYPE_ASM[torch.float4_e2m1fn_x2] = "i8"

    # Map to HAL Element Type
    # Using UINT8 is safe for transport. The NPU instruction doesn't care about the type tag, just bits.
    if hasattr(torch, "float8_e8m0fnu"):
        conversions.DTYPE_TO_ELEMENT_TYPE[torch.float8_e8m0fnu] = HalElementType.UINT_8

    if hasattr(torch, "float4_e2m1fn_x2"):
        conversions.DTYPE_TO_ELEMENT_TYPE[torch.float4_e2m1fn_x2] = HalElementType.UINT_8

    # Map to NumPy
    if hasattr(torch, "float8_e8m0fnu"):
        conversions.TORCH_DTYPE_TO_NUMPY[torch.float8_e8m0fnu] = np.dtype("u1")
        
    if hasattr(torch, "float4_e2m1fn_x2"):
        conversions.TORCH_DTYPE_TO_NUMPY[torch.float4_e2m1fn_x2] = np.dtype("u1")

    print("✅ IREE Turbine patched (Storage=UINT8).")

if __name__ == "__main__":
    enable_mxfp8_support()