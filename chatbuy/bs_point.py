import pandas as pd
from scripts.llm_models import model_4o_mini as model
from pathlib import Path

from pydantic_ai import Agent

csv_path = Path("data/BTC_USDT_1d_with_indicators.csv")
dataframes = pd.read_csv(csv_path)

agent = Agent(
    model=model,
    deps_type=str,
    result_type=str,
    system_prompt="你是一名优秀的交易员,现在你需要找到根据我给你提供的数据来找到关键位置.",
)


@agent.system_prompt
async def system_prompt() -> str:
    df = dataframes[["timestamp", "histogram"]]
    return df[-60:].to_markdown(index=False)


if __name__ == "__main__":
    # import asyncio
    from rich import print

    # result = asyncio.run(
    #     agent.run("找到其中列名为histogram指标由负变正和由正变负的所有临界点")
    # )
    result = agent.run_sync("找到其中列名为histogram指标由负变正和由正变负的所有临界点")
    print(result.all_messages())
    print(result.data)
    # print(result.cost)
