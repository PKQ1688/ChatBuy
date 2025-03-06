import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# from openai import OpenAI

load_dotenv(override=True)

client = InferenceClient(provider="hyperbolic", api_key=os.getenv("HF_TOKEN"))
# client = OpenAI(
#     base_url="https://router.huggingface.co/hyperbolic/",
#     api_key=os.getenv("HF_TOKEN"),
# )

messages = [{"role": "user", "content": "帮我分析一下深圳的就业环境？"}]

completion = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=messages,
    max_tokens=500,
)

print(completion.choices[0].message)
