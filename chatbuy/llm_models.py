import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


from pydantic_ai.models.openai import OpenAIModel


load_dotenv(override=True)

client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_0806"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION_0806"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_0806"],
)

model_0806 = OpenAIModel("4o0806", openai_client=client)

if __name__ == "__main__":
    from pydantic_ai import Agent

    agent = Agent(
        model=model_0806,
        system_prompt="Be concise, reply with one sentence.",
    )

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.data)
