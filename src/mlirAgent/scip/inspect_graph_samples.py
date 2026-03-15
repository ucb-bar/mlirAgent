import sys
from pathlib import Path

from neo4j import GraphDatabase

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from src.mlirAgent.config import Config


def inspect():
    driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("\n🔎 --- INSPECTION REPORT ---\n")

        # 1. Who is the "Popular Kid"? (Most Called Function)
        print("1️⃣  TOP 5 MOST CALLED FUNCTIONS:")
        result = session.run("""
            MATCH (f)-[:CALLS]->(target)
            WHERE target.label IN ['FUNCTION', 'METHOD']
            RETURN target.name as Name, count(f) as Callers
            ORDER BY Callers DESC LIMIT 5
        """)
        for record in result:
            print(f"   • {record['Name']}: Called by {record['Callers']} places")

        # 2. Who is the "Hardest Worker"? (File with most definitions)
        print("\n2️⃣  FILES WITH MOST DEFINITIONS:")
        result = session.run("""
            MATCH (f:FILE)-[:DEFINED_IN]-(s) 
            RETURN f.name as File, count(s) as Defs
            ORDER BY Defs DESC LIMIT 5
        """)
        # Note: Direction might be (s)-[:DEFINED_IN]->(f) or (f)-[:DEFINES]->(s) 
        # based on your ingestor. Let's try the label we used: "DEFINES" from File->Symbol
        result = session.run("""
            MATCH (f:FILE)-[:DEFINES]->(s) 
            RETURN f.name as File, count(s) as Defs
            ORDER BY Defs DESC LIMIT 5
        """)
        for record in result:
            print(f"   • {record['File']}: Defines {record['Defs']} symbols")

        # 3. Can we trace a path? (Chain of reasoning)
        print("\n3️⃣  SAMPLE CALL CHAIN (Depth 2):")
        result = session.run("""
            MATCH (a:FUNCTION)-[:CALLS]->(b:FUNCTION)-[:CALLS]->(c:FUNCTION)
            RETURN a.name, b.name, c.name LIMIT 3
        """)
        for record in result:
            print(f"   • {record['a.name']} -> {record['b.name']} -> {record['c.name']}")

    driver.close()

if __name__ == "__main__":
    inspect()