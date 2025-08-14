# Create new Azure configuration
from pydantic_settings import BaseSettings

class AzureSettings(BaseSettings):
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = "ENTER OPENAI ENPOINT URL HERE"
    AZURE_OPENAI_API_KEY: str = "ENTER OPENAI API KEY HERE"
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"  # or your model deployment name

    # Neo4j Aura (includes APOC!)
    NEO4J_URI: str = "ENTER NEO4J URL HERE"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "ENTER NEO4J PASSWORD HERE"

    # Document settings
    DOC_DIR: str = "input-dir"
