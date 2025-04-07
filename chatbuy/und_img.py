import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(override=True)

provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)
model = OpenAIModel("openrouter/quasar-alpha", provider=provider)

agent = Agent(model=model, system_prompt="Be concise, reply with one sentence.")

result = agent.run_sync("Who are you?")
print(result.data)
