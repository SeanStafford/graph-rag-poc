# Create the enhanced query processor
"""
Chain-of-Thought Query Enhancement using Azure OpenAI
Demonstrates improved reasoning for complex SAP HANA queries
"""

from azure_config import AzureSettings
from openai import AzureOpenAI
from llama_index.core import SimpleDirectoryReader
import json

config = AzureSettings()

client = AzureOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    api_version='2024-02-01',
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT
)

def enhanced_query(question, context_chunks):
    """Enhanced query with chain-of-thought reasoning"""

    # Combine context from multiple document chunks
    context = "\n".join([chunk.text[:500] for chunk in context_chunks[:5]])

    cot_prompt = f"""You are an expert SAP HANA consultant. Answer the question using chain-of-thought reasoning.

CONTEXT from SAP HANA on VMware documentation:
{context}

QUESTION: {question}

Think step-by-step:
1. What SAP HANA concepts are involved in this question?
2. What configuration parameters or components are relevant?
3. How do these elements interact in a VMware environment?
4. What specific recommendations apply?

FINAL ANSWER: Provide a comprehensive response based on your analysis."""

    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": cot_prompt}],
        max_tokens=800,
        temperature=0.3
    )

    return response.choices[0].message.content

def test_enhanced_queries():
    """Test the enhancement with complex SAP HANA queries"""

    # Load documents
    loader = SimpleDirectoryReader(input_dir=config.DOC_DIR, required_exts=[".pdf"])
    docs = loader.load_data()
    print(f"Loaded {len(docs)} document chunks")

    # Test queries that would challenge a basic system
    test_queries = [
        "What memory configuration parameters should I optimize for SAP HANA performance on VMware?",
        "How do I configure NUMA settings for SAP HANA on vSphere?",
        "What are the storage requirements and best practices for SAP HANA data files?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        try:
            response = enhanced_query(query, docs)
            print(f"ENHANCED RESPONSE:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_enhanced_queries()