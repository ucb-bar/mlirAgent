import argparse

import scip_pb2
from neo4j import GraphDatabase

from mlirAgent.config import Config


class GraphRefresher:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )

    def refresh_files(self, file_paths):
        with self.driver.session() as session:
            for file_path in file_paths:
                print(f"♻️  Refreshing: {file_path}")
                
                # 1. PURGE (Delete old nodes for this file)
                # We match the file node and all symbols defined exclusively in it
                purge_query = """
                MATCH (f:FILE {name: $path})
                OPTIONAL MATCH (f)-[:DEFINES]->(s:SYMBOL)
                DETACH DELETE f, s
                """
                session.run(purge_query, path=file_path)

        # 2. RE-INGEST (Load only these files from index)
        # Note: You still load the full index.scip (fast), but filter what you write (slow)
        self._ingest_subset(file_paths)

    def _ingest_subset(self, target_files):
        index = scip_pb2.Index()
        # Ensure this points to your actual index file
        with open("index.scip", "rb") as f:
            index.ParseFromString(f.read())
            
        print(f"📂 Loaded SCIP Index. Scanning for {len(target_files)} files...")
        
        target_set = set(target_files)
        
        for doc in index.documents:
            if doc.relative_path in target_set:
                print(f"   Found {doc.relative_path} in index. Ingesting...")
                # *** CALL YOUR EXISTING INGESTION LOGIC HERE ***
                # You can import your original ingest_codegraph.py function:
                # ingest_document(doc, self.driver) 
                pass 

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='List of files to refresh')
    args = parser.parse_args()
    
    refresher = GraphRefresher()
    refresher.refresh_files(args.files)
    refresher.close()