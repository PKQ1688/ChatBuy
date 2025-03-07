from base_model.smol_lm import model_qwen32 as model
from smolagents import tool

# from smolagents.agents import ToolCallingAgent
from smolagents.agents import CodeAgent
from tool.data_indicators_fetcher import fetch_current_price, fetch_historical_data
from utils import logger

agent = CodeAgent(
    tools=[tool(fetch_current_price), tool(fetch_historical_data)],
    model=model,
    # grammar=TradingDecision,
)
response = agent.run("请查询今天ETH的价格")
logger.info(f"Agent response: {response}")
