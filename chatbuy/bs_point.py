import pandas as pd
from chatbuy.llm_models import model_0806 as model
from pathlib import Path

from pydantic_ai import Agent

csv_path = Path("data/BTC_USDT_1d_with_indicators.csv")
dataframes = pd.read_csv(csv_path)

agent = Agent(
    model=model,
    deps_type=str,
    result_type=str,
)


@agent.system_prompt
async def system_prompt() -> str:
    df = dataframes[["timestamp", "histogram"]]
    return df[-60:].to_markdown(index=False)


if __name__ == "__main__":
    import asyncio

    result = asyncio.run(
        agent.run("找到其中列名为histogram指标由负变正和由正变负的所有临界点")
    )

    print(result.data)
    # print(result.cost)
