#!/usr/bin/env python3
"""
Test current Graph RAG system capabilities without Neo4j dependency
"""

import sys
sys.path.append('.')

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from config import config

def test_document_loading():
    """Test if we can load and chunk the PDF document"""
    print("Testing document loading...")
    try:
        loader = SimpleDirectoryReader(
            input_dir=config.DOC_DIR,
            required_exts=[".pdf"],
            recursive=True
        )
        docs = loader.load_data()
        print(f"✓ Successfully loaded {len(docs)} document chunks")
        print(f"✓ Sample text from first chunk: {docs[0].text[:150]}...")
        return docs
    except Exception as e:
        print(f"✗ Document loading failed: {e}")
        return None

def test_llm_connection():
    """Test if we can connect to Ollama LLM"""
    print("\nTesting LLM connection...")
    try:
        llm = Ollama(
            model=config.OLLAMA_LLM_MODEL, 
            base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
            request_timeout=60.0
        )
        # Simple test query
        response = llm.complete("What is SAP HANA? Answer in one sentence.")
        print(f"✓ LLM connection successful")
        print(f"✓ Sample response: {str(response)}")
        return llm
    except Exception as e:
        print(f"✗ LLM connection failed: {e}")
        return None

def test_embedding_model():
    """Test if we can use the embedding model"""
    print("\nTesting embedding model...")
    try:
        embed_model = OllamaEmbedding(
            model_name=config.OLLAMA_EMBED_MODEL,
            base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
        )
        # Test embedding
        embedding = embed_model.get_text_embedding("SAP HANA database")
        print(f"✓ Embedding model working, vector dimension: {len(embedding)}")
        return embed_model
    except Exception as e:
        print(f"✗ Embedding model failed: {e}")
        return None

def test_basic_rag_without_graph(docs, llm, embed_model):
    """Test basic RAG functionality using vector index instead of graph"""
    print("\nTesting basic RAG (vector-based, no graph)...")
    try:
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Create simple vector index
        index = VectorStoreIndex.from_documents(docs[:20])  # Limit to first 20 chunks for speed
        query_engine = index.as_query_engine()
        
        # Test query
        test_queries = [
            "What is SAP HANA?",
            "What are the memory requirements for SAP HANA?",
            "How should VMware vSphere be configured?"
        ]
        
        for query in test_queries:
            print(f"\n  Query: {query}")
            response = query_engine.query(query)
            print(f"  Response: {str(response)[:600]}...")
            
        print("✓ Basic RAG functionality working")
        return True
    except Exception as e:
        print(f"✗ Basic RAG failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Current System Capability Test ===")
    print(f"Using LLM: {config.OLLAMA_LLM_MODEL}")
    print(f"Using Embedding: {config.OLLAMA_EMBED_MODEL}")
    print(f"Document directory: {config.DOC_DIR}")
    
    # Run tests
    docs = test_document_loading()
    if not docs:
        sys.exit(1)
    
    llm = test_llm_connection() 
    if not llm:
        sys.exit(1)
        
    embed_model = test_embedding_model()
    if not embed_model:
        sys.exit(1)
    
    success = test_basic_rag_without_graph(docs, llm, embed_model)
    
    if success:
        print("\n✓ Current system basic capabilities verified!")
        print("Next step: Add Graph RAG with Neo4j for advanced querying")
    else:
        print("\n✗ System has issues that need to be resolved")