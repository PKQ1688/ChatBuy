from pathlib import Path
from phi.agent import Agent
import pandas as pd
from chatbuy.llm_models import llm_model_0806 as llm_model

csv_path = Path("data/BTC_USDT_1d_with_indicators.csv")
dataframes = pd.read_csv(csv_path)

agent = Agent(
    model=llm_model,
    markdown=True,
    show_tool_calls=True,
    instructions=[
        "data:",
        dataframes[-60:].to_markdown(index=False),
    ],
    debug_mode=True,
)

agent.print_response(
    """找到其中列名为histogram指标由负变正和由正变负的所有临界点""", stream=False
)
