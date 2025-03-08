import os

from dotenv import load_dotenv
from smolagents import HfApiModel, LiteLLMModel

load_dotenv(override=True)


model_mini = LiteLLMModel("azure/gpt-4o-mini")
model_0806 = LiteLLMModel(model_id="azure/gpt-4o-0806")
model_1120 = LiteLLMModel(model_id="azure/gpt-4o-1120")

# model_deepseek_r1 = HfApiModel(
#     provider="hyperbolic",
#     model_id="deepseek-ai/DeepSeek-R1",
#     token=os.getenv("HF_TOKEN"),
# )


model_deepseek_r1 = HfApiModel(
    model_id="deepseek-ai/DeepSeek-R1",
    provider="together",
)

model_qwen32 = LiteLLMModel(
    model_id="groq/qwen-2.5-32b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

model_qwq32 = LiteLLMModel(
    model_id="groq/qwen-qwq-32b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

model_qwen_code = LiteLLMModel(
    model_id="groq/qwen-2.5-coder-32b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)


if __name__ == "__main__":
    import time

    from smolagents import CodeAgent

    agent = CodeAgent(
        tools=[],  # No tools needed to demonstrate the issue
        model=model_qwq32,
        add_base_tools=False,
        verbosity_level=2,
    )

    st = time.time()
    # Try to run a simple task
    try:
        result = agent.run("Say hello!")
        print(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {str(e)}")

    print(f"Time taken: {time.time() - st:.2f} seconds")
