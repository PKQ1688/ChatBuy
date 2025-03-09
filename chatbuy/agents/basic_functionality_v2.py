from base_model.smol_lm import model_deepseek_r1 as model
from smolagents import GradioUI, tool

# from smolagents.agents import ToolCallingAgent
from smolagents.agents import CodeAgent
from tool.data_fetcher import fetch_current_price, fetch_historical_data

agent = CodeAgent(
    tools=[
        tool(fetch_current_price),
        tool(fetch_historical_data),
    ],
    model=model,
    verbosity_level=1,
    name="query_agent",
    description="This is an agent used to query cryptocurrency prices and technical indicators.",
)
# response = agent.run("请查询今天ETH的价格")

GradioUI(agent, file_upload_folder="data").launch(share=False)
