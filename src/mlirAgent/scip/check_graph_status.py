from neo4j import GraphDatabase

from mlirAgent.config import Config


def check_status():
    print(f"🔌 Connecting to Neo4j at {Config.NEO4J_URI}...")
    
    try:
        driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        print("✅ Connection Successful!")
        
        with driver.session() as session:
            # --- NEW: Check Server Version ---
            ver_result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()
            server_version = ver_result["v"]
            print(f"🆕 Server Version: {server_version}")
            
            if not server_version.startswith("5.26"):
                print("⚠️  WARNING: Server is NOT running the expected version (5.26.0)!")
            else:
                print("✅ Update Verified: Running Neo4j 5.26.0")
            # ---------------------------------

            # 1. Count Nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            
            # 2. Count Edges
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            edge_count = result.single()["count"]
            
            print("\n📊 Graph Statistics:")
            print(f"   • Nodes: {node_count:,}")
            print(f"   • Edges: {edge_count:,}")
            
            if node_count > 0:
                print("\n🔍 Sample Data (Last Ingested Node):")
                sample = session.run("MATCH (n) RETURN n LIMIT 1").single()["n"]
                print(f"   • Labels: {list(sample.labels)}")
                print(f"   • Properties: {dict(sample)}")
            else:
                print("\n⚠️  Database is empty! (Ingestion might be queued or failed)")

        driver.close()

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("   Make sure your Docker container is running.")
if __name__ == "__main__":
    check_status()