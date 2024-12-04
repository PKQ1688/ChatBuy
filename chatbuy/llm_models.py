import os

from dotenv import load_dotenv
from phi.embedder.azure_openai import AzureOpenAIEmbedder
from phi.model.aws.claude import Claude
from phi.model.azure.openai_chat import AzureOpenAIChat

load_dotenv(override=True)


llm_model_4o = AzureOpenAIChat(
    id="gpt4o",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
)

llm_model_0806 = AzureOpenAIChat(
    id="4o0806",
    api_key=os.getenv("AZURE_OPENAI_API_KEY_0806"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_0806"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_0806", "2024-08-01-preview"),
    temperature=0.5,
    top_p=0.95,
)

llm_claude = Claude(
    id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_region=os.getenv("AWS_DEFAULT_REGION"),
)

embed_model = AzureOpenAIEmbedder(
    model="text-embedding-3-large",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

if __name__ == "__main__":
    from phi.agent.agent import Agent

    agent = Agent(provider=llm_model_0806, markdown=True)  # type: ignore
    agent.print_response("Share a 2 sentence horror story.")
