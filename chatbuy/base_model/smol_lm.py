import os

from dotenv import load_dotenv
from smolagents import HfApiModel, LiteLLMModel

load_dotenv(override=True)


model_mini = LiteLLMModel("azure/gpt-4o-mini")
model_0806 = LiteLLMModel(model_id="azure/gpt-4o-0806")
model_1120 = LiteLLMModel(model_id="azure/gpt-4o-1120")

model_qwq32 = HfApiModel(
    provider="hyperbolic", model_id="Qwen/QwQ-32B", token=os.getenv("HF_TOKEN")
)

# model_qwq32 = LiteLLMModel(
#     model_id="groq/qwen-qwq-32b", api_key=os.getenv("GROQ_API_KEY")
# )

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
    response = model_1120(messages)
    print(response)

    print(f"Time taken: {time.time() - st:.2f} seconds")
