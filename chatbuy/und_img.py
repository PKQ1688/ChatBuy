import asyncio
import glob
import os
from typing import Literal

import pandas as pd
from agno.agent import Agent, RunResponse
from agno.media import Image
from agno.models.openrouter import OpenRouter
from pydantic import BaseModel
from tqdm import tqdm


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
    """Represents a trading pipeline with a specific strategy.

    strategy : str
        The trading strategy being used.
    """

    def __init__(
        self,
        strategy: str = "Buy when the price hits the lower Bollinger Band, sell when it hits the upper band, otherwise hold.",
        debug_mode: bool = False,
    ):
        self.agent = Agent(
            model=OpenRouter(
                # id="openai/gpt-4.1-nano",
                # id="openai/gpt-4.1-mini",
                id="openai/gpt-4.1",
                # id="google/gemini-2.5-pro-preview-03-25"
            ),
            response_model=TradeAdvice,
            description="Provide a trading decision based on the strategy I provide.",
            debug_mode=debug_mode,
        )
        self.strategy = strategy

    def run_pipeline(self, image_path: str) -> TradeAdvice:
        """Run the trading pipeline with the provided image path.

        Parameters
        ----------
        image_path : str
            The path to the image file.

        Returns:
        -------
        TradeAdvice
            The trade advice based on the analysis of the chart.
        """
        response: RunResponse = self.agent.run(
            f"Strategy: {self.strategy}.",
            images=[Image(filepath=image_path)],
        )
        return response.content

    async def a_run_pipeline(self, image_path: str) -> TradeAdvice:
        """Asynchronous version, suitable for agents that support async."""
        response: RunResponse = await self.agent.arun(
            f"Strategy: {self.strategy}.",
            images=[Image(filepath=image_path)],
        )
        return response.content


async def batch_process_images(image_dir: str, output_csv: str, strategy: str):
    """Batch process all images in the given directory asynchronously, obtain trade advice for each, and save the results to a CSV file.

    Args:
        image_dir (str): Directory containing images.
        output_csv (str): Output CSV file path.
        strategy (str): Trading strategy description.
    """
    pipeline = TradePipeline(strategy=strategy)
    image_paths = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(
        os.path.join(image_dir, "*.jpg")
    )
    # image_paths = image_paths[:5]

    async def process(img_path):
        advice = await pipeline.a_run_pipeline(img_path)

        action = str(advice.action)
        reason = str(advice.reason)

        filename = os.path.splitext(os.path.basename(img_path))[0]
        trade_time = filename.rsplit("_", 1)[-1]
        return {
            "trade_time": trade_time,
            "action": action,
            "reason": reason,
            "image": os.path.basename(img_path),
        }

    tasks = [process(img_path) for img_path in image_paths]
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"
    ):
        res = await coro
        results.append(res)

    df = pd.DataFrame(results)
    df.sort_values("trade_time", inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Processed {len(results)} images, results saved to {output_csv}")


if __name__ == "__main__":
    # Configuration parameters
    image_dir = "data/btc_daily"  # Replace with your image folder path
    output_csv = "output/trade_advice_results.csv"  # Output CSV path
    strategy = "Buy when the lowest price of the cryptocurrency falls below the lower Bollinger Band, sell when the highest price rises above the upper band, otherwise hold."

    # res = TradePipeline(strategy=strategy, debug_mode=True).run_pipeline(
    #     image_path="data/btc_daily/coin_120_20210712_20211108.png"
    # )
    # print(res)
    # Run the asynchronous batch processing
    asyncio.run(batch_process_images(image_dir, output_csv, strategy))
