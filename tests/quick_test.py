#!/usr/bin/env python3
"""Quick test of Ollama LLM connection"""
import sys
sys.path.append('.')

from config import config
from llama_index.llms.ollama import Ollama

print(f"Testing Ollama with model: {config.OLLAMA_LLM_MODEL}")
print(f"Host: {config.OLLAMA_HOST}:{config.OLLAMA_PORT}")

try:
    llm = Ollama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
        request_timeout=90.0  # Increased for first-time model loading
    )
    
    print("Sending test query (may take 30-60 seconds for first request)...")
    response = llm.complete("What is SAP HANA? Answer briefly in one sentence.")
    print(f"✓ SUCCESS: {str(response)}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)