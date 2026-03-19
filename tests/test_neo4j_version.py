from neo4j import GraphDatabase

from mlirAgent.config import Config

with GraphDatabase.driver(
    Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
) as driver:
    driver.verify_connectivity()
    with driver.session() as session:
        result = session.run(
            "CALL dbms.components() YIELD versions RETURN versions"
        ).single()
        print(f"Current Neo4j Version: {result['versions'][0]}")
