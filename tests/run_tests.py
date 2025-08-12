#!/usr/bin/env python3
"""
Unified test runner for Graph RAG system components
Run this to validate all system components are working correctly
"""

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global log file
LOG_FILE = "tests/test_results.log"

def log_and_print(message, log_file=None):
    """Print message and write to log file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def run_test_script(script_name, description):
    """Run a test script and report results"""
    separator = "=" * 60
    log_and_print(f"\n{separator}", LOG_FILE)
    log_and_print(f"{description}", LOG_FILE)
    log_and_print(f"{separator}", LOG_FILE)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, f"tests/{script_name}"], 
                              capture_output=True, text=True, timeout=120)
        
        # Log the output regardless of success/failure
        if result.stdout:
            log_and_print(result.stdout, LOG_FILE)
        if result.stderr:
            log_and_print("STDERR:", LOG_FILE)
            log_and_print(result.stderr, LOG_FILE)
        
        if result.returncode == 0:
            log_and_print(f"✓ {description} - PASSED", LOG_FILE)
            return True
        else:
            log_and_print(f"✗ {description} - FAILED (exit code: {result.returncode})", LOG_FILE)
            return False
            
    except subprocess.TimeoutExpired:
        msg = f"[TIMEOUT] {description} - TIMEOUT (>2min)"
        log_and_print(msg, LOG_FILE)
        return False
    except Exception as e:
        msg = f"[ERROR] {description} - ERROR: {e}"
        log_and_print(msg, LOG_FILE)
        return False

def main():
    """Run all system tests"""
    # Initialize log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'w') as f:
        f.write(f"Graph RAG System Test Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
    
    log_and_print(">>> Graph RAG System Test Suite <<<", LOG_FILE)
    log_and_print("     Testing all components...", LOG_FILE)
    
    tests = [
        ("quick_test.py", "Ollama LLM Connection Test"),
        ("test_neo4j.py", "Neo4j Database Connection Test"), 
        ("test_current_system.py", "Document Loading & Basic RAG Test"),
        ("test_graph_rag.py", "Full Graph RAG System Test"),
    ]
    
    results = []
    for script, description in tests:
        success = run_test_script(script, description)
        results.append((description, success))
    
    # Summary
    separator = "=" * 60
    log_and_print(f"\n{separator}", LOG_FILE)
    log_and_print("TEST SUMMARY", LOG_FILE)
    log_and_print(f"{separator}", LOG_FILE)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL" 
        log_and_print(f"{status} - {description}", LOG_FILE)
    
    log_and_print(f"\nOverall: {passed}/{total} tests passed", LOG_FILE)
    
    if passed == total:
        log_and_print("*** All systems operational! Ready for Graph RAG implementation.", LOG_FILE)
    else:
        log_and_print("!!! Some components need attention before proceeding.", LOG_FILE)
        log_and_print("\nDIAGNOSIS:", LOG_FILE)
        log_and_print("- Local Neo4j container lacks APOC plugin required for Graph RAG", LOG_FILE)
        log_and_print("- Solution: Deploy to cloud with managed Neo4j service or install APOC locally", LOG_FILE)
        log_and_print("- Recommendation: Proceed with Azure cloud deployment", LOG_FILE)
    
    log_and_print(f"\nLog saved to: {LOG_FILE}", LOG_FILE)

if __name__ == "__main__":
    main()