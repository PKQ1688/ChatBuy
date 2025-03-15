import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv(override=True)


class PydanticModel:
    """A wrapper class for handling different language model services using pydantic-ai."""

    def __init__(self, service="azure", model_id="gpt-4o-1120", **kwargs):
        self.service = service
        self.model_id = model_id
        self.kwargs = kwargs

        self.model = self.create_model()

    def create_model(self):
        if self.service == "azure":
            azure_gpt_client = AsyncAzureOpenAI(
                azure_endpoint=os.environ["AZURE_API_BASE"],
                api_version=os.environ["AZURE_API_VERSION"],
                api_key=os.environ["AZURE_API_KEY"],
            )
            gpt_provider = OpenAIProvider(openai_client=azure_gpt_client)
            return OpenAIModel(self.model_id, provider=gpt_provider, **self.kwargs)
        elif self.service == "groq":
            return GroqModel(
                self.model_id,
                provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
                **self.kwargs,
            )
        elif self.service == "deepseek":
            deepseek_provider = OpenAIProvider(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            return OpenAIModel(self.model_id, provider=deepseek_provider, **self.kwargs)
        else:
            raise ValueError(f"Service '{self.service}' not supported")


# Pre-defined models for convenience
model_reason = PydanticModel(service="deepseek", model_id="deepseek-reasoner")
model_4o = PydanticModel(service="azure", model_id="gpt4o")
model_0806 = PydanticModel(service="azure", model_id="gpt-4o-0806")
model_1120 = PydanticModel(service="azure", model_id="gpt-4o-1120")
model_mini = PydanticModel(service="azure", model_id="gpt-4o-mini")
model_qwen32 = PydanticModel(service="groq", model_id="qwen-2.5-32b")
model_qwq32 = PydanticModel(service="groq", model_id="qwen-qwq-32b")

if __name__ == "__main__":
    import time

    from pydantic_ai import Agent

    # Create models directly and pass the underlying model to Agent
    # Option 1: Create the model directly and extract the underlying model
    # groq_model = PydanticModel(service="groq", model_id="qwen-qwq-32b")
    agent = Agent(model=model_1120.model)  # Pass the actual GroqModel instance

    # Option 2: Or use model strings as expected by pydantic_ai
    # agent = Agent(model="groq:qwen-qwq-32b")

    st = time.time()
    res = agent.run_sync("who are you?")
    print(res.data)
    print(f"Time taken: {time.time() - st:.2f} seconds")

    # Or using a pre-defined model
    # agent = Agent(model=model_qwq32.model)  # Access the underlying model
