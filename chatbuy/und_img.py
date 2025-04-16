import os
from typing import Literal

from agno.agent import Agent, RunResponse
from agno.media import Image
from agno.models.openai.like import OpenAILike
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
                # id="google/gemma-3-27b-it:free",
                id="google/gemini-2.0-flash-001",
                # id="google/gemini-2.5-pro-preview-03-25",
                # id="deepseek/deepseek-chat-v3-0324",
                # id="openai/gpt-4.1-nano",
            )
        else:
            model = OpenAILike(
                id="deepseek-ai/DeepSeek-V3-0324",
                base_url="https://router.huggingface.co/hyperbolic/v1",
                api_key=os.environ["HF_TOKEN"],
            )

        self.agent = Agent(
            model=model,
            response_model=TradeAdvice,
            description="Provide a trading decision based on the strategy I provide.",
            instructions=[
                "Note:",
                "1. Green candles indicate that the closing price is higher than the opening price, red candles indicate that the closing price is lower than the opening price.",
                "2. The upper shadow of the candle represents the highest price, and the lower shadow represents the lowest price.",
                "3. The opening and closing prices are the bottom and top of the candle, respectively.",
            ],
            debug_mode=debug_mode,
            use_json_mode=True,
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
        debug_mode=True,
        use_openrouter=True,
    )
    res = pipe.run_pipeline(
        strategy="只分析最后一天的K线数据。当天的收盘价格跌破布林线下轨时买入，当价格升至布林线上轨时卖出，否则持有。",
        image_path="data/btc_daily/coin_120_20210712_20211108.png",
        # markdown_text=csv_data.to_markdown(index=False),
    )
    print(res)
