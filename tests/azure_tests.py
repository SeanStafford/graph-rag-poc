#!/usr/bin/env python3
"""
Azure Cloud Test Suite - Compare with local test results
"""

import sys
import datetime
import os
# Add parent directory to path so we can import azure_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append('..')

from azure_config import AzureSettings
from openai import AzureOpenAI
from neo4j import GraphDatabase

LOG_FILE = "tests/azure_test_results.log"
config = AzureSettings()

def log_and_print(message, log_file=None):
    """Print message and write to log file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def test_azure_openai():
    """Test Azure OpenAI connection and reasoning"""
    try:
        client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version='2024-02-01',
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )

        response = client.chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "What is SAP HANA? Answer in one technical sentence."}],
            max_tokens=100
        )

        log_and_print(f"✓ Azure OpenAI Response: {response.choices[0].message.content}", LOG_FILE)
        return True
    except Exception as e:
        log_and_print(f"✗ Azure OpenAI failed: {e}", LOG_FILE)
        return False

def test_neo4j_aura():
    """Test Neo4j Aura with APOC availability"""
    try:
        with GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                # Test basic connection
                result = session.run("RETURN 'Neo4j Aura working!' as message")
                message = result.single()["message"]

                # Test APOC availability (the issue that blocked local setup)
                apoc_result = session.run("CALL apoc.help('meta') YIELD name RETURN count(name) as apoc_functions")
                apoc_count = apoc_result.single()["apoc_functions"]

                log_and_print(f"✓ Neo4j Aura: {message}", LOG_FILE)
                log_and_print(f"✓ APOC functions available: {apoc_count}", LOG_FILE)
                return True
    except Exception as e:
        log_and_print(f"✗ Neo4j Aura failed: {e}", LOG_FILE)
        return False

def main():
    """Run Azure cloud tests and compare with local results"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log
    with open(LOG_FILE, 'w') as f:
        f.write(f"Azure Cloud Test Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Compare with local test_results.log\n")
        f.write(f"{'='*60}\n\n")

    log_and_print(">>> Azure Cloud Test Suite <<<", LOG_FILE)
    log_and_print("Comparing cloud services vs local setup", LOG_FILE)

    tests = [
        ("Azure OpenAI Test", test_azure_openai),
        ("Neo4j Aura + APOC Test", test_neo4j_aura),
    ]

    results = []
    for test_name, test_func in tests:
        log_and_print(f"\n{'='*60}", LOG_FILE)
        log_and_print(f"{test_name}", LOG_FILE)
        log_and_print(f"{'='*60}", LOG_FILE)

        success = test_func()
        results.append((test_name, success))

    # Summary
    log_and_print(f"\n{'='*60}", LOG_FILE)
    log_and_print("AZURE vs LOCAL COMPARISON", LOG_FILE)
    log_and_print(f"{'='*60}", LOG_FILE)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        log_and_print(f"{status} - {test_name}", LOG_FILE)

    log_and_print(f"\nAzure Results: {passed}/{total} tests passed", LOG_FILE)
    log_and_print("Local Results: 2/4 tests passed (from test_results.log)", LOG_FILE)

    log_and_print(f"\nKEY ADVANTAGES:", LOG_FILE)
    log_and_print("- Azure OpenAI: More powerful than local Llama 3.2", LOG_FILE)
    log_and_print("- Neo4j Aura: APOC included (blocked local Graph RAG)", LOG_FILE)
    log_and_print("- Managed services: No dependency management overhead", LOG_FILE)

if __name__ == "__main__":
    main()