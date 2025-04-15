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
                id="openai/gpt-4.1",
                # id="deepseek/deepseek-chat-v3-0324:free",
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
            debug_mode=debug_mode,
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
    pipe = TradePipeline(
        debug_mode=False,
        use_openrouter=True,
    )
    res = pipe.run_pipeline(
        strategy="Buy when the price hits the lower Bollinger Band, sell when it hits the upper band, otherwise hold.",
        image_path="data/btc_daily/coin_120_20210630_20211027.png",
    )
    print(res)
