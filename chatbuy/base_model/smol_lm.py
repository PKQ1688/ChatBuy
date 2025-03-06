import os

from dotenv import load_dotenv
from smolagents import AzureOpenAIServerModel, HfApiModel, LiteLLMModel

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

# model_qwq32 = HfApiModel(
#     provider="hyperbolic", model_id="Qwen/QwQ-32B", token=os.environ.get("HF_TOKEN")
# )

model_qwq32 = LiteLLMModel(
    model_id="groq/qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY")
)

if __name__ == "__main__":
    import time

    messages = [
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday",
        },
    ]

    st = time.time()
    response = model_qwq32(messages)
    print(response)

    print(f"Time taken: {time.time() - st:.2f} seconds")
