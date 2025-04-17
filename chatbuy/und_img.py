import os
from textwrap import dedent
from typing import Literal

from agno.agent import Agent, RunResponse
from agno.media import Image

# from agno.models.openai.like import OpenAILike
from agno.models.azure import AzureOpenAI
from agno.models.openrouter import OpenRouter
from pydantic import BaseModel


class TradeAdvice(BaseModel):
    """Represents trade advice based on the analysis of a chart.

    action : str
        The recommended action, which can be "hold", "sell", or "buy".
    reason : str
        The reason for the recommended action.
    """

    action: Literal["hold", "sell", "buy"]  # Restrict to specific values
    reason: str


class TradePipeline:
    """Unified trading inference pipeline, supports two input modes: image K-line and text K-line."""

    def __init__(
        self,
        debug_mode: bool = False,
        use_openrouter: bool = False,
    ):
        if use_openrouter:
            model = OpenRouter(
                # id="openai/gpt-4.1",
                # id="openai/o4-mini",
                id="openai/o3",
                # id="google/gemma-3-27b-it:free",
                # id="google/gemini-2.0-flash-001",
                # id="google/gemini-2.5-pro-preview-03-25",
                # id="deepseek/deepseek-chat-v3-0324",
                # id="openai/gpt-4.1-nano",
                # temperature=0.1,
            )
        else:
            # model = OpenAILike(
            #     id="deepseek-ai/DeepSeek-V3-0324",
            #     base_url="https://router.huggingface.co/hyperbolic/v1",
            #     api_key=os.environ["HF_TOKEN"],
            # )
            model = AzureOpenAI(id="gpt-4o-1120")

        self.agent = Agent(
            model=model,
            response_model=TradeAdvice,
            description=dedent("""\
                You are a trading assistant. You can analyze K-line charts and provide trading advice.
                You can only analyze the K-line chart and the strategy provided by the user.
                You cannot analyze other factors, such as news, market sentiment, or other indicators.
                You must carefully analyze the provided K-line chart image. 
                Do not fabricate or assume any information that is not clearly visible in the image. 
                Only base your advice on what you can actually see in the chart.
            """),
            instructions=[
                "Note:",
                "1.	Green candles indicate that the closing price is higher than the opening price, while red candles indicate that the closing price is lower than the opening price.",
                "2.	The upper shadow of the candle represents the highest price during the period, and the lower shadow represents the lowest price.",
                "3.	The top and bottom of the candle body represent the opening and closing prices, depending on the price movement. For green candles (uptrend), the bottom is the opening price; for red candles (downtrend), the top is the opening price.",
                "4.	The chart layout consists of the candlestick chart at the top, volume in the middle, and the MACD indicator at the bottom.",
                "5.	Bollinger Bands consist of an upper band, a lower band, and a middle band, where the middle band represents the average price.",
                "6.	The MACD indicator is composed of three elements: the blue line is the MACD line, the orange line is the signal line, and the histogram shows the difference between the two lines.",
            ],
            debug_mode=debug_mode,
            # use_json_mode=True,
        )

    def run_pipeline(
        self,
        strategy: str = "Buy when the price hits the lower Bollinger Band, sell when it hits the upper band, otherwise hold.",
        image_path: str | None = None,
        markdown_text: str | None = None,
    ):
        """Unified entry point, supports three modes.

        - Only pass image_path (image K-line)
        - Only pass markdown_text (text K-line)
        - Pass both image_path and markdown_text (image + text joint inference)

        Parameters:
            image_path: Image file path, optional
            markdown_text: K-line markdown text, optional

        Returns:
            TradeAdvice
        """
        if not image_path and not markdown_text:
            raise ValueError(
                "image_path and markdown_text cannot both be empty, at least one must be provided"
            )
        msg = f"Strategy: {strategy}."
        #
        if markdown_text:
            msg += f"\n\nHere is the recent K-line data in markdown table:\n{markdown_text}"
        images = [Image(filepath=image_path)] if image_path else None
        response: RunResponse = self.agent.run(
            msg,
            images=images,
        )
        return response.content

    async def a_run_pipeline(
        self,
        strategy: str = "Buy when the price hits the lower Bollinger Band, sell when it hits the upper band, otherwise hold.",
        image_path: str | None = None,
        markdown_text: str | None = None,
    ):
        """Asynchronous unified entry point, supports three modes.

        - Only pass image_path (image K-line)
        - Only pass markdown_text (text K-line)
        - Pass both image_path and markdown_text (image + text joint inference)

        Parameters:
            image_path: Image file path, optional
            markdown_text: K-line markdown text, optional

        Returns:
            TradeAdvice
        """
        if not image_path and not markdown_text:
            raise ValueError(
                "image_path and markdown_text cannot both be empty, at least one must be provided"
            )
        msg = f"Strategy: {strategy}."
        if markdown_text:
            msg += f"\n\nHere is the recent K-line data in markdown table:\n{markdown_text}"
        images = [Image(filepath=image_path)] if image_path else None
        response: RunResponse = await self.agent.arun(
            msg,
            images=images,
        )
        return response.content


if __name__ == "__main__":
    import pandas as pd

    csv_data = pd.read_csv("data/BTC_USDT_1d_with_indicators.csv")

    csv_data = csv_data[csv_data["timestamp"] >= "2021-07-12"]
    csv_data = csv_data[csv_data["timestamp"] <= "2021-11-08"]

    pipe = TradePipeline(
        debug_mode=False,
        use_openrouter=False,
    )
    image_dir = "data/btc_daily"
    test_image_name = [
        "coin_120_20210712_20211108.png",
        "coin_120_20210713_20211109.png",
        "coin_120_20210714_20211110.png",
        "coin_120_20210722_20211118.png",
        "coin_120_20210726_20211122.png",
        "coin_120_20210801_20211128.png",
    ]
    for i in sorted(os.listdir(image_dir)):
        if i.endswith(".png"):
            # if image_path =[image_path]
            if i not in test_image_name:
                continue

            image_path = os.path.join(image_dir, i)
            print(f"Processing image: {image_path}")
            res = pipe.run_pipeline(
                strategy="只分析最后一天的K线数据。当天的收盘价格跌破布林线下轨时买入，当价格升至布林线上轨时卖出，否则持有",
                image_path=image_path,
                # markdown_text=csv_data.to_markdown(index=False),
            )
            print(f"Advice: {res.action}, Reason: {res.reason}")
            print("=" * 50)

    # res = pipe.run_pipeline(
    #     strategy="只分析最后一天的K线数据。当天的收盘价格跌破布林线下轨时买入，当价格升至布林线上轨时卖出，否则持有",
    #     image_path="data/btc_daily/coin_120_20210712_20211108.png",
    #     # markdown_text=csv_data.to_markdown(index=False),
    # )
    # print(res)
