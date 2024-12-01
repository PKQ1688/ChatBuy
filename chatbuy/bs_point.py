from pathlib import Path
from phi.agent import Agent
from phi.tools.pandas import PandasTools
from phi.model.azure import AzureOpenAIChat
import pandas as pd


csv_path = Path("data/BTC_USDT_1d_with_indicators.csv")
dataframes = {"btc": pd.read_csv(csv_path)}

agent = Agent(
    model=AzureOpenAIChat(id="gpt4o"),
    tools=[PandasTools()],
    markdown=True,
    show_tool_calls=True,
    # instructions=[
    #     # "First always get the list of files",
    #     # "Then check the columns in the file",
    #     # "Then run the query to answer the question",
    # ],
)
agent.print_response(
    """找到其中列名为histogram指标由负变正和由正变负的临界点从 `csv_data`, 
    use: {"dataframe_name": "csv_data", "operation": "head", "operation_parameters": {"n": 5}}""", 
    stream=True
)
