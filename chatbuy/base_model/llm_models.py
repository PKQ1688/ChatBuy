import os

from agno.models.azure import AzureOpenAI
from agno.models.deepseek import DeepSeek
from agno.models.groq import Groq
from agno.models.huggingface import HuggingFace


class AgnoModel:
    """A wrapper class for handling different language model services, currently supporting Azure OpenAI."""

    def __init__(self, service="azure", model_id="gpt-4o-1120", provider="", **kwargs):
        self.service = service
        self.model_id = model_id
        self.provider = provider
        self.kwargs = kwargs

        self.model = self.create_model()

    def create_model(self):
        if self.service == "azure":
            return AzureOpenAI(
                id=self.model_id,
                azure_endpoint=os.environ["AZURE_API_BASE"],
                api_version=os.environ["AZURE_API_VERSION"],
                api_key=os.environ["AZURE_API_KEY"],
                **self.kwargs,
            )
        elif self.service == "hf":
            # todo huggingface 的 第三方 provider 服务还不可用
            return HuggingFace(
                id=self.model_id,
                provider=self.provider,
                api_key=os.environ["HF_TOKEN"],
                **self.kwargs,
            )
        elif self.service == "groq":
            return Groq(
                id=self.model_id,
                api_key=os.environ["GROQ_API_KEY"],
                **self.kwargs,
            )
        elif self.service == "deepseek":
            return DeepSeek(
                id=self.model_id,
                api_key=os.environ["DEEPSEEK_API_KEY"],
                **self.kwargs,
            )
        else:
            raise ValueError(f"Service '{self.service}' not supported")

    def __getattr__(self, name):
        return getattr(self.model, name)


if __name__ == "__main__":
    from agno.agent import Agent

    agent = Agent(
        # model=AgnoModel(service="azure", model_id="gpt-4o-1120", temperature=0.1),
        # model=AgnoModel(
        #     service="hf",
        #     model_id="Qwen/QwQ-32B",
        #     # provider="together",
        # ),
        # model=AgnoModel(service="groq", model_id="qwen-qwq-32b"),
        model=AgnoModel(service="deepseek", model_id="deepseek-chat"),
        markdown=True,
    )

    agent.print_response(message="who are you?", stream=True)
