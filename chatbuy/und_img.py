from typing import Literal

from agno.agent import Agent
from agno.media import Image
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
    """Represents a trading pipeline with a specific strategy.

    strategy : str
        The trading strategy being used.
    """

    def __init__(
        self, strategy: str = "如果到布林道下轨就买入，上轨到达时卖出，剩下的时候就观望"
    ):
        self.agent = Agent(
            model=OpenRouter(
                id="openrouter/optimus-alpha",
            ),
            response_model=TradeAdvice,
            instructions="根据我提供给的策略，给出交易判断",
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
        result = self.agent.run(
            f"策略如下:{self.strategy}，请根据策略给出交易判断",
            images=[Image(filepath=image_path)],
        ).content

        print(result)


if __name__ == "__main__":
    image_path = (
        "data/btc_daily/coin_120_20210101_20210430.png"  # 替换为你的本地图片路径
    )

    pipeline = TradePipeline(
        strategy="如果到布林道下轨就买入，上轨到达时卖出，剩下的时候就观望"
    )
    pipeline.run_pipeline(image_path)
