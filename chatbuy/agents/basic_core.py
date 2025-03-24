import logfire
from pydantic_ai import Agent  # , Tool

from chatbuy.base_model.pydantic_lm import PydanticModel

logfire.configure()

model = PydanticModel(service="openrouter", model_id="qwen/qwq-32b:free")

basic_agent = Agent(
    model=model,
    # system_prompt='Make a trading decision based on the provided data.',
    # tools=[
    #     Tool(fetch_historical_data, takes_ctx=False),
    #     Tool(fetch_current_price, takes_ctx=False),
    #     Tool(fetch_and_calculate_indicators, takes_ctx=False),
    # ],
)


res = basic_agent.run_sync("帮我查一下今天ETH的价格")
print(res.data)
