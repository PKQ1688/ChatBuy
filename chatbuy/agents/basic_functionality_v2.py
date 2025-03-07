from base_model.smol_lm import model_qwen_code as model
from smolagents import tool

# from smolagents.agents import ToolCallingAgent
from smolagents.agents import CodeAgent
from tool.data_indicators_fetcher import fetch_current_price, fetch_historical_data

agent = CodeAgent(
    tools=[tool(fetch_current_price), tool(fetch_historical_data)],
    model=model,
    # grammar=TradingDecision,
)
response = agent.run("请查询今天ETH的价格")
