#!/usr/bin/env python3
"""
Semantic Graph RAG Implementation
Fixes the core problem: keyword-based nodes -> semantic concept-based nodes

This implementation uses:
1. Domain-specific schema (Concept, Parameter, Component, Recommendation)
2. Structured JSON extraction prompts 
3. Multi-hop reasoning for complex SAP HANA queries

Usage: python semantic_graph_rag.py
"""

import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, ServiceContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

# Use Azure OpenAI instead of local Ollama
from openai import AzureOpenAI
from azure_config import AzureSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityExtractionResult:
    """Structured result from semantic entity extraction"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    chunk_summary: str

class SemanticEntityExtractor:
    """
    Domain-specific entity extractor for SAP HANA on VMware documentation
    Uses Azure OpenAI with structured prompts instead of generic keyword extraction
    """
    
    def __init__(self, config: AzureSettings):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version='2024-02-01',
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        
        # Define our semantic schema (from GeminiConversation analysis)
        self.schema = {
            "entity_types": {
                "Concept": "High-level technical idea or best practice (e.g., NUMA Optimization, Virtualization)",
                "Parameter": "Specific configuration setting (e.g., sched.cpu.affinity, numa.nodeAffinity)", 
                "Component": "Software/hardware piece (e.g., vSphere, SAP HANA, CPU, Guest OS)",
                "Recommendation": "Best practice rule or guideline"
            },
            "relationship_types": {
                "HAS_CONCEPT": "Document discusses concept",
                "DESCRIBES_PARAMETER": "Document describes parameter",
                "INVOLVES_COMPONENT": "Concept involves component", 
                "SETS_PARAMETER": "Recommendation sets parameter",
                "AFFECTS": "Parameter affects component",
                "FOR_CONCEPT": "Recommendation for concept"
            }
        }
    
    def extract_entities_from_chunk(self, text_chunk: str) -> EntityExtractionResult:
        """
        Extract semantic entities using structured JSON prompt
        This replaces the generic max_triplets_per_chunk approach
        """
        
        extraction_prompt = f"""You are an expert in SAP HANA on VMware systems. Extract structured information from this technical documentation chunk.

SCHEMA DEFINITION:
Entity Types:
- Concept: High-level technical ideas (e.g., "NUMA Optimization", "Memory Management")
- Parameter: Specific config settings (e.g., "sched.cpu.affinity", "numa.nodeAffinity") 
- Component: Software/hardware pieces (e.g., "vSphere", "SAP HANA", "Guest OS")
- Recommendation: Best practice rules or guidelines

Relationship Types:
- INVOLVES_COMPONENT: Concept involves component
- SETS_PARAMETER: Recommendation sets parameter
- AFFECTS: Parameter affects component
- FOR_CONCEPT: Recommendation for concept

TEXT CHUNK:
{text_chunk}

Extract entities and relationships. Output MUST be valid JSON:

{{
  "entities": [
    {{"type": "Concept", "name": "NUMA Optimization", "description": "brief description"}},
    {{"type": "Parameter", "name": "numa.nodeAffinity", "description": "controls NUMA node assignment"}}
  ],
  "relationships": [
    {{"from": "NUMA Optimization", "to": "vSphere", "type": "INVOLVES_COMPONENT"}},
    {{"from": "Configure NUMA settings", "to": "numa.nodeAffinity", "type": "SETS_PARAMETER"}}
  ],
  "chunk_summary": "Brief summary of what this chunk discusses"
}}

Focus on SAP HANA performance, VMware configuration, and system optimization concepts."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=1500,
                temperature=0.1  # Low temperature for consistent structured output
            )
            
            # Parse JSON response
            json_str = response.choices[0].message.content.strip()
            if json_str.startswith('```json'):
                json_str = json_str.split('```json')[1].split('```')[0]
            
            extracted_data = json.loads(json_str)
            
            return EntityExtractionResult(
                entities=extracted_data.get("entities", []),
                relationships=extracted_data.get("relationships", []),
                chunk_summary=extracted_data.get("chunk_summary", "")
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Fallback to empty result rather than crash
            return EntityExtractionResult(entities=[], relationships=[], chunk_summary="")

class SemanticGraphRAG:
    """
    Complete semantic Graph RAG system that fixes the barebones implementation
    """
    
    def __init__(self, config: AzureSettings):
        self.config = config
        self.extractor = SemanticEntityExtractor(config)
        self.graph_store = None
        self.query_engine = None
        
        # Azure OpenAI client for final answer generation
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version='2024-02-01', 
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
    
    def setup_graph_store(self):
        """Setup Neo4j graph store with semantic schema"""
        logger.info("Setting up Neo4j graph store...")
        
        self.graph_store = Neo4jGraphStore(
            username=self.config.NEO4J_USERNAME,
            password=self.config.NEO4J_PASSWORD,
            url=self.config.NEO4J_URI,
            database="neo4j",
            timeout=120.0
        )
        
        # Clear existing data for clean start
        with self.graph_store._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing graph data")
    
    def ingest_documents_semantically(self, documents: List):
        """
        Ingest documents using semantic extraction instead of generic triplets
        This is the core fix for the barebones version
        """
        logger.info(f"Processing {len(documents)} document chunks with semantic extraction...")
        
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        for i, doc in enumerate(documents[:20]):  # Process first 20 chunks for demo
            logger.info(f"Processing chunk {i+1}/20...")
            
            # Extract semantic entities (NOT generic keywords)
            extraction = self.extractor.extract_entities_from_chunk(doc.text)
            
            if not extraction.entities:
                continue
                
            # Create nodes in Neo4j
            with self.graph_store._driver.session() as session:
                # Create document node
                session.run(
                    "CREATE (d:Document {chunk_id: $chunk_id, summary: $summary})",
                    chunk_id=f"chunk_{i}", summary=extraction.chunk_summary
                )
                
                # Create entity nodes
                for entity in extraction.entities:
                    node_query = f"""
                    MERGE (e:{entity['type']} {{name: $name}})
                    SET e.description = $description
                    """
                    session.run(node_query, 
                               name=entity['name'], 
                               description=entity.get('description', ''))
                    
                    # Link to document
                    link_query = f"""
                    MATCH (d:Document {{chunk_id: $chunk_id}})
                    MATCH (e:{entity['type']} {{name: $name}})
                    MERGE (d)-[:HAS_{entity['type'].upper()}]->(e)
                    """
                    session.run(link_query, chunk_id=f"chunk_{i}", name=entity['name'])
                
                # Create relationships
                for rel in extraction.relationships:
                    rel_query = f"""
                    MATCH (a {{name: $from_name}})
                    MATCH (b {{name: $to_name}}) 
                    MERGE (a)-[:{rel['type']}]->(b)
                    """
                    session.run(rel_query, 
                               from_name=rel['from'], 
                               to_name=rel['to'])
        
        logger.info("Semantic ingestion complete!")
    
    def setup_query_engine(self):
        """Setup query engine with semantic graph retrieval"""
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        # Use the same retriever architecture but with semantic graph
        graph_rag_retriever = KnowledgeGraphRAGRetriever(
            storage_context=storage_context,
            verbose=True
        )
        
        self.query_engine = RetrieverQueryEngine.from_args(
            graph_rag_retriever
        )
    
    def query_with_chain_of_thought(self, question: str) -> str:
        """
        Enhanced querying with chain-of-thought reasoning
        Addresses the 'does not perform chain of thought' problem
        """
        logger.info(f"Processing query: {question}")
        
        # Get graph context using our semantic retriever
        retrieved_context = self.query_engine.retrieve(question)
        
        # Build structured context from graph traversal
        context_parts = []
        for node in retrieved_context:
            context_parts.append(f"- {node.text}")
        
        context = "\n".join(context_parts)
        
        # Chain-of-thought prompt for final answer generation
        cot_prompt = f"""You are an expert SAP HANA on VMware consultant. Answer the question using chain-of-thought reasoning.

CONTEXT from semantic knowledge graph:
{context}

QUESTION: {question}

Think step-by-step:
1. What SAP HANA concepts are involved in this question?
2. What configuration parameters or components are relevant?
3. How do these elements interact in a VMware environment?
4. What specific recommendations apply?

FINAL ANSWER: Provide a comprehensive response based on your analysis, including specific parameters and recommendations."""

        response = self.client.chat.completions.create(
            model=self.config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": cot_prompt}],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content

def main():
    """Main execution function"""
    logger.info("Starting Semantic Graph RAG system...")
    
    # Load configuration
    config = AzureSettings()
    
    # Load documents
    loader = SimpleDirectoryReader(
        input_dir=config.DOC_DIR,
        required_exts=[".pdf"],
        recursive=True
    )
    docs = loader.load_data()
    logger.info(f"Loaded {len(docs)} document chunks")
    
    # Initialize semantic Graph RAG
    rag_system = SemanticGraphRAG(config)
    
    # Setup and ingest
    rag_system.setup_graph_store()
    rag_system.ingest_documents_semantically(docs)
    rag_system.setup_query_engine()
    
    # Test queries that failed in barebones version
    test_queries = [
        "What parameters should I check to optimize SAP HANA performance on VMware vSphere?",
        "Which components are involved in setting up SAP HANA and VMware according to best practice?", 
        "What are the NUMA configuration recommendations for SAP HANA?",
        "How should I configure CPU affinity settings?"
    ]
    
    logger.info("Testing semantic Graph RAG with complex queries...")
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        try:
            response = rag_system.query_with_chain_of_thought(query)
            print(f"SEMANTIC GRAPH RAG RESPONSE:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Show graph statistics
    with rag_system.graph_store._driver.session() as session:
        stats = session.run("""
        RETURN 
          size((n:Concept)) as concepts,
          size((n:Parameter)) as parameters, 
          size((n:Component)) as components,
          size((n:Recommendation)) as recommendations
        """).single()
        
        print(f"\n{'='*80}")
        print("SEMANTIC GRAPH STATISTICS:")
        print(f"Concepts: {stats['concepts']}")
        print(f"Parameters: {stats['parameters']}")
        print(f"Components: {stats['components']}")  
        print(f"Recommendations: {stats['recommendations']}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()