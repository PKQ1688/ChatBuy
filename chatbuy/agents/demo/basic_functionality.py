import logfire
from pydantic_ai import Agent, Tool

from chatbuy.base_model.pydantic_lm import model_1120 as model
from chatbuy.tool.demo.data_fetcher import (
    fetch_and_calculate_indicators,
    fetch_current_price,
    fetch_historical_data,
)

logfire.configure()

basic_agent = Agent(
    model=model,
    # system_prompt='Make a trading decision based on the provided data.',
    tools=[
        Tool(fetch_historical_data, takes_ctx=False),
        Tool(fetch_current_price, takes_ctx=False),
        Tool(fetch_and_calculate_indicators, takes_ctx=False),
    ],
)


res = basic_agent.run_sync("帮我查一下今天ETH的价格")
print(res.data)
