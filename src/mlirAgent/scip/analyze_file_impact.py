import random

from neo4j import GraphDatabase

from mlirAgent.config import Config


class GraphAuditor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )

    def run_audit(self):
        with self.driver.session() as session:
            print("\n🔍 --- STARTING GRAPH INTELLIGENCE AUDIT ---\n")
            
            # --- CATEGORY 1: THE CELEBRITIES ---
            print("🏆 PHASE 1: Auditing 'Core Infrastructure' (Top 3 Most Used Functions)")
            query_celebs = """
            MATCH (target:FUNCTION)<-[:CALLS]-(caller)
            RETURN target.name as Name, target.id as ID, count(caller) as Pop
            ORDER BY Pop DESC
            LIMIT 3
            """
            celebs = list(session.run(query_celebs))
            for c in celebs:
                self._generate_report(session, c['Name'], c['ID'], "CORE INFRASTRUCTURE")

            # --- CATEGORY 2: THE LOCAL HEROES ---
            print("\n🏘️  PHASE 2: Auditing 'Standard Logic' (Functions with 2-10 Callers)")
            query_mid = """
            MATCH (target:FUNCTION)<-[:CALLS]-(caller)
            WITH target, count(caller) as Pop
            WHERE Pop > 2 AND Pop < 10
            RETURN target.name as Name, target.id as ID
            LIMIT 10
            """
            # Randomly pick 3 from the list to get variety
            mids = list(session.run(query_mid))
            if mids:
                for m in random.sample(mids, min(3, len(mids))):
                    self._generate_report(session, m['Name'], m['ID'], "STANDARD LOGIC")
            else:
                print("   (No mid-tier functions found yet. Ingestion might be too early.)")

            # --- CATEGORY 3: THE LONELY WOLVES ---
            print("\n🐺 PHASE 3: Auditing 'Leaf Nodes' (Functions with 0 Callers)")
            query_leaf = """
            MATCH (target:FUNCTION)
            WHERE NOT (target)<-[:CALLS]-(:FUNCTION)
            RETURN target.name as Name, target.id as ID
            LIMIT 10
            """
            leaves = list(session.run(query_leaf))
            if leaves:
                for l in random.sample(leaves, min(3, len(leaves))):
                    self._generate_report(session, l['Name'], l['ID'], "LEAF NODE")
            else:
                print("   (No leaf nodes found. This is statistically impossible unless empty.)")

    def _generate_report(self, session, name, scip_id, category):
        print(f"\n   >>> ANALYZING: {self._clean_name(name)} ({category})")
        
        # 1. Get Definition Location
        query_def = "MATCH (s) WHERE s.id = $id RETURN s.file_path as Path, labels(s) as Type"
        res = session.run(query_def, id=scip_id).single()
        if not res: return
        file_name = Path(res['Path']).name
        
        # 2. Get Blast Radius
        query_impact = """
        MATCH (caller)-[:CALLS]->(target)
        WHERE target.id = $id
        RETURN caller.name as Who, caller.file_path as Path
        """
        callers = list(session.run(query_impact, id=scip_id))
        
        # 3. Print The "Agent Context"
        print(f"       📍 Defined in: {file_name}")
        print(f"       🔗 Impact: Breaks {len(callers)} call sites across {len(set(c['Path'] for c in callers))} files.")
        
        if len(callers) > 0:
            # Show a sample caller to prove connectivity
            sample = callers[0]
            clean_caller = self._clean_name(sample['Who'])
            sample_file = Path(sample['Path']).name
            print(f"       ⚠️  Example usage: Called by `{clean_caller}` in `{sample_file}`")
        else:
            print("       ✅ Safe to Refactor: No incoming dependencies found.")

    def _clean_name(self, raw_name):
        return raw_name.replace('cxx . . $ ', '').split('(')[0].strip('.')

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    auditor = GraphAuditor()
    auditor.run_audit()
    auditor.close()