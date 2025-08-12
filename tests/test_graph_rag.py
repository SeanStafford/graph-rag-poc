#!/usr/bin/env python3
"""Quick test of full Graph RAG system"""
import sys
sys.path.append('.')

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from config import config

print("Testing full Graph RAG system...")
print(f"LLM: {config.OLLAMA_LLM_MODEL}")
print(f"Neo4j: {config.NEO4J_URI}")

try:
    # Setup models
    llm = Ollama(model=config.OLLAMA_LLM_MODEL, 
                 base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
                 request_timeout=120.0)
    embed_model = OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL,
                                  base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Load document chunks from single large PDF
    loader = SimpleDirectoryReader(input_dir=config.DOC_DIR, required_exts=[".pdf"])
    docs = loader.load_data()
    print(f"✓ Loaded {len(docs)} document chunks from PDF")
    
    # Use only first 10 chunks for quick test (instead of full 134 chunks)
    test_docs = docs[:10]
    print(f"✓ Testing with {len(test_docs)} chunks for speed (out of {len(docs)} total)")
    
    # Setup Neo4j graph store
    graph_store = Neo4jGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
        database="neo4j",
        timeout=120.0
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    
    # Build knowledge graph (this may take a minute)
    print("Building knowledge graph... (this may take 1-2 minutes)")
    kg_index = KnowledgeGraphIndex.from_documents(
        test_docs,
        storage_context=storage_context,
        max_triplets_per_chunk=3,  # Reduced for speed
        embed_model=embed_model,
        show_progress=True
    )
    
    # Create retriever and query engine
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=True,
    )
    query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever,
        embed_model=embed_model,
    )
    
    # Test query
    test_query = "What is SAP HANA?"
    print(f"\nTesting query: {test_query}")
    response = query_engine.query(test_query)
    
    print(f"\n✓ GRAPH RAG RESPONSE:")
    print(f"{str(response)}")
    print("\n*** Full Graph RAG system working!")
    
except Exception as e:
    print(f"✗ Graph RAG test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)