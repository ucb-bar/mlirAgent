import os
import sys
from pathlib import Path

# Add project root to python path to allow importing config if needed
# (Optional, but good practice for standalone scripts)
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    # Now it is a clean relative import since they are in the same folder
    from . import scip_pb2
except ImportError:
    # Fallback if run as a script directly (python src/mlirAgent/scip/inspect.py)
    import scip_pb2

def inspect_index(scip_path):
    print(f"🕵️  Inspecting: {scip_path}")
    
    if not os.path.exists(scip_path):
        print(f"❌ File not found at: {scip_path}")
        print("   Did you run the ingest_codegraph.py script yet?")
        return

    index = scip_pb2.Index()
    try:
        file_size = os.path.getsize(scip_path) / (1024 * 1024)
        print(f"   • File Size: {file_size:.2f} MB")
        
        # SCIP files can be large, read binary
        with open(scip_path, "rb") as f:
            index.ParseFromString(f.read())
    except Exception as e:
        print(f"❌ Failed to parse protobuf: {e}")
        return

    print("✅ Successfully Parsed Index!")
    print(f"   • Documents Indexed: {len(index.documents)}")
    
    # ... (Rest of the logic remains the same) ...

if __name__ == "__main__":
    # DYNAMIC PATH RESOLUTION
    # This automatically finds data/knowledge_base/scip/index_test.scip
    # regardless of where you run the script from.
    current_dir = Path(__file__).parent
    # Go up: src/mlirAgent/scip -> src/mlirAgent -> src -> mlirEvolve -> data
    default_data_path = current_dir.parent.parent.parent / "data" / "knowledge_base" / "scip" / "index_test.scip"
    
    inspect_index(str(default_data_path))