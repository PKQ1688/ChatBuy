import os

from dotenv import load_dotenv
from phi.agent.agent import Agent
from phi.model.azure import AzureOpenAIChat

load_dotenv(override=True)

model_4o = AzureOpenAIChat(
    id='gpt4o',
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

model_4o_mini = AzureOpenAIChat(
    id='gpt4o-mini',
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

model_0806 = AzureOpenAIChat(
    id='4o0806',
    api_key=os.getenv('AZURE_OPENAI_API_KEY_0806'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION_0806'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT_0806'),
)

if __name__ == '__main__':
    agent = Agent(model=model_0806)
    agent.print_response('who is the president of the united states?')
