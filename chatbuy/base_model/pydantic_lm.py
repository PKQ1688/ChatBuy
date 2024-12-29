import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel

load_dotenv(override=True)

client = AsyncAzureOpenAI(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
)

model_4o = OpenAIModel('gpt4o', openai_client=client)
model_0806 = OpenAIModel('gpt-4o-0806', openai_client=client)
model_1120 = OpenAIModel('gpt-4o-1120', openai_client=client)
model_mini = OpenAIModel('gpt-4o-mini', openai_client=client)


if __name__ == '__main__':
    from pydantic_ai import Agent

    agent = Agent(model=model_1120)
    res = agent.run_sync('who are you?')
    print(res.data)
