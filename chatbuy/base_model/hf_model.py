import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI

load_dotenv(override=True)

use_openai = False

client_hf = InferenceClient(provider="hf-inference", api_key=os.getenv("HF_TOKEN"))
client_openai = OpenAI(
    base_url="https://router.huggingface.co/hf-inference/v1",
    api_key=os.getenv("HF_TOKEN"),
)

messages = [{"role": "user", "content": "帮我分析一下深圳的就业环境？"}]

if use_openai:
    client = client_openai
else:
    client = client_hf

completion = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=messages,
    max_tokens=5000,
)

print(completion.choices[0].message.content)
