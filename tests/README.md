# Graph RAG System Tests

This directory contains test scripts to validate different components of the Graph RAG system.

## Quick Start

Run all tests:
```bash
python tests/run_tests.py
```

## Individual Tests

### 1. `quick_test.py`
**Purpose**: Validates Ollama LLM connection and basic response generation
- Tests model loading and inference
- Verifies Ollama service is running
- Quick validation (<30 seconds)

### 2. `test_neo4j.py` 
**Purpose**: Validates Neo4j database connection
- Tests database connectivity
- Verifies authentication
- Confirms query execution

### 3. `test_current_system.py`
**Purpose**: Comprehensive system validation
- Document loading from PDF
- Embedding model functionality  
- Basic RAG query pipeline
- Vector-based retrieval (fallback when graph unavailable)

### 4. `test_graph_rag.py`
**Purpose**: Full Graph RAG system test
- Knowledge graph construction
- Graph-based retrieval
- End-to-end query processing
- **Note**: Requires Neo4j with APOC plugin

## Configuration

Tests use configuration from `../config.py`. To run with different settings:

1. **Local Testing**: Uncomment `config = LocalNeo4jSettings()` in config.py
2. **Llama Models**: Uncomment `config = LlamaSettings()` in config.py
3. **Original Setup**: Use default `config = Settings()`

## Prerequisites

- **Ollama**: Running locally with required models
- **Neo4j**: For graph tests (local Docker or cloud instance)
- **Python Dependencies**: Installed via Poetry (`poetry install`)

## Expected Results

- All tests should pass for a fully functional system
- Graph RAG test may fail without APOC plugin in Neo4j
- Test runner provides detailed output and summary