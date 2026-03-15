from neo4j import GraphDatabase

# Auth from your docker-compose
NEO4J_AUTH = ("neo4j", "your_password_here") 

with GraphDatabase.driver("bolt://localhost:7687", auth=NEO4J_AUTH) as driver:
    driver.verify_connectivity()
    with driver.session() as session:
        # This query asks the internal system for its component versions
        result = session.run("CALL dbms.components() YIELD versions RETURN versions").single()
        print(f"✅ Current Neo4j Version: {result['versions'][0]}")