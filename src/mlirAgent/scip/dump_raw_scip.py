from mlirAgent.config import Config
from mlirAgent.scip import scip_pb2

path = str(Config.PROJECT_ROOT / "data" / "knowledge_base" / "scip" / "index_test.scip")

print(f"Reading: {path}")
index = scip_pb2.Index()
with open(path, "rb") as f:
    index.ParseFromString(f.read())

# Dump the first 3 symbols of the first document with symbols
for doc in index.documents:
    if doc.symbols:
        print(f"\nFile: {doc.relative_path}")
        for sym in doc.symbols[:3]:
            print(f"  Symbol: {sym.symbol}")
            # This prints the RAW integer from the Protobuf
            print(f"  Raw Kind Int: {sym.kind}") 
        break