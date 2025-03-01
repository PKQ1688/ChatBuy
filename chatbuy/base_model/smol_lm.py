import os

from dotenv import load_dotenv
from smolagents import AzureOpenAIServerModel, LiteLLMModel

load_dotenv(override=True)

# openai_kwargs = {
#     'AZURE_API_BASE': os.environ['AZURE_OPENAI_ENDPOINT'],
#     'api_version': os.environ['AZURE_OPENAI_API_VERSION'],
#     'api_key': os.environ['AZURE_OPENAI_API_KEY'],
# }

os.environ["AZURE_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]

model_mini = LiteLLMModel("azure/gpt-4o-mini")

model_0806 = LiteLLMModel(model_id="azure/gpt-4o-0806")
# model_1120 = LiteLLMModel(model_id="azure/gpt-4o-1120")

model_1120 = AzureOpenAIServerModel(
    model_id="gpt-4o-1120",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday",
        },
    ]

    print(model_1120(messages))
