import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(override=True)

print(os.getenv("HUGGINGFACE_API_KEY"))
exit()

client = InferenceClient(
    provider="hyperbolic", api_key=os.getenv("HF_TOKEN")
)

messages = [{"role": "user", "content": "What is the capital of France?"}]

completion = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=messages,
    max_tokens=500,
)

print(completion.choices[0].message)
