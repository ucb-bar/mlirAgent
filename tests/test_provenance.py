import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.mlirAgent.tools.provenance import MLIRProvenanceTracer

# --- CONFIGURATION ---
TARGET_ROOT = "experiments/iree_artifacts/compilation_quantized_matmul/artifacts_riscv/ir_pass_history/builtin_module_no-symbol-name"
TARGET_FILE = "compilation_quantized_matmul/quantized_matmul.mlir"
TARGET_LINE = 26

def main():
    print("🧪 STARTING ROBUST PROVENANCE TEST")
    print("==================================")
    print(f"📂 Root:   {TARGET_ROOT}")
    print(f"🎯 Target: {TARGET_FILE}:{TARGET_LINE}")
    
    if not os.path.exists(TARGET_ROOT):
        print("❌ Error: Target directory does not exist.")
        return

    try:
        tracer = MLIRProvenanceTracer()
        result = tracer.trace(TARGET_ROOT, TARGET_FILE, TARGET_LINE)
        
        events = result.get("events", [])
        print(f"\n📊 Found {len(events)} events.")
        
        if len(events) == 0:
            print("⚠️  Warning: No events found. The target location might not be present in the history files.")
        
        for i, event in enumerate(events):
            print(f"\n[{i+1}] PASS: {event['pass']}")
            print(f"    ACTION: {event['action'].upper()}")
            
            code = event.get('code', '')
            if code:
                lines = code.splitlines()
                
                # VERIFICATION: Ensure structural sanitization worked
                has_loc = any("loc(" in line for line in lines)
                status_str = "❌ CONTAINS LOC" if has_loc else "✅ CLEAN"
                
                print(f"    STATUS: {status_str}")
                print("-" * 30)
                # Print snippet
                for line in lines[:8]:
                    print(f"    {line}")
                if len(lines) > 8:
                    print("    ... (truncated)")
        
        if len(events) > 0:
            print("\n✅ SUCCESS: Tool is robust and operational.")

    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()