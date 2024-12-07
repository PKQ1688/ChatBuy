from scripts.llm_engines import llm_engine_0806 as llm_engine

from transformers.agents import ReactCodeAgent

csv_path = "data/BTC_USDT_1d_with_indicators.csv"
# dataframes = pd.read_csv(csv_path)[-60:]

agent = ReactCodeAgent(
    llm_engine=llm_engine,
    tools=[],
    max_iterations=12,
    verbose=1,
    additional_authorized_imports=[
        "os",
        "pathlib",
        "numpy",
        "pandas",
        "PIL",
    ],
    planning_interval=3,
    plan_type="default",
)


if __name__ == "__main__":
    # import asyncio
    from rich import print

    # result = asyncio.run(
    #     agent.run("找到其中列名为histogram指标由负变正和由正变负的所有临界点")
    # )
    result = agent.run(
        task=(
            "你要根据我给你提供的`csv_path`,找到其中列名为histogram指标由负变正和由正变负的所有临界点"
            "你只要看最后60行数据,不需要看完整的表格"
        ),
        csv_path=csv_path,
    )

    print(result)
