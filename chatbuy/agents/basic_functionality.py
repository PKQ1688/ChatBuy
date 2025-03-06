import logfire
from base_model.pydantic_lm import model_qwen32
from pydantic_ai import Agent, Tool
from tool.data_indicators_fetcher import fetch_current_price, fetch_historical_data

logfire.configure()

basic_agent = Agent(
    model=model_qwen32,
    # system_prompt='Make a trading decision based on the provided data.',
    tools=[
        Tool(fetch_historical_data, takes_ctx=False),
        Tool(fetch_current_price, takes_ctx=False),
    ],
)


res = basic_agent.run_sync("帮我查一下今天BTC的价格")
print(res.data)
