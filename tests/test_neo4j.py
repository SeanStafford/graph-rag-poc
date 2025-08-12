#!/usr/bin/env python3
"""Quick test of local Neo4j connection"""
import sys
sys.path.append('.')

from config import config
from neo4j import GraphDatabase

print(f"Testing Neo4j connection to: {config.NEO4J_URI}")
print(f"Username: {config.NEO4J_USERNAME}")

try:
    with GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j is working!' as message")
            message = result.single()["message"]
            print(f"✓ SUCCESS: {message}")
            
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)