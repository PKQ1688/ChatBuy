import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(override=True)


deepseek_provider = OpenAIProvider(
    api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
)

gpt_client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_API_BASE"],
    api_version=os.environ["AZURE_API_VERSION"],
    api_key=os.environ["AZURE_API_KEY"],
)

gpt_provider = OpenAIProvider(openai_client=gpt_client)

model_reason = OpenAIModel("deepseek-reasoner", provider=deepseek_provider)

model_4o = OpenAIModel("gpt4o", provider=gpt_provider)
model_0806 = OpenAIModel("gpt-4o-0806", provider=gpt_provider)
model_1120 = OpenAIModel("gpt-4o-1120", provider=gpt_provider)
model_mini = OpenAIModel("gpt-4o-mini", provider=gpt_provider)

model_qwen32 = GroqModel("qwen-2.5-32b", api_key=os.getenv("GROQ_API_KEY"))
model_qwq32 = GroqModel("qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY"))

if __name__ == "__main__":
    import time

    from pydantic_ai import Agent

    agent = Agent(model=model_qwq32)

    st = time.time()
    res = agent.run_sync("who are you?")
    print(res.data)
    print(f"Time taken: {time.time() - st:.2f} seconds")
