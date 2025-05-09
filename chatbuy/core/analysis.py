# from textwrap import dedent

import asyncio
import os

from agno.agent import Agent
from agno.media import Image
from agno.models.azure import AzureOpenAI
from agno.team.team import Team


class BasicTechnicalAnalysis:
    """Represents basic analysis of a chart."""

    def __init__(
        self,
        model_name: str = "gpt-4o-1120",
        temperature: float = 0.1,
        macd_prompt: str = None,
        bb_prompt: str = None,
        rsi_prompt: str = None,
    ):
        if macd_prompt is None:
            macd_prompt = "You are a trading assistant.You provide analysis based on MACD indicators."
        if bb_prompt is None:
            bb_prompt = "You are a trading assistant.You provide analysis based on Bollinger Bands indicators."
        if rsi_prompt is None:
            rsi_prompt = "You are a trading assistant.You provide analysis based on RSI indicators."

        self.model = AzureOpenAI(id=model_name, temperature=temperature)

        macd_agent = Agent(
            name="MACD Agent",
            role=macd_prompt,
            model=self.model,
        )
        bb_agent = Agent(
            name="BB Agent",
            role=bb_prompt,
            model=self.model,
        )
        rsi_agent = Agent(
            name="RSI Agent",
            role=rsi_prompt,
            model=self.model,
        )

        self.technical_team = Team(
            name="Technical Analysis Team",
            mode="collaborate",
            model=self.model,
            members=[macd_agent, bb_agent, rsi_agent],
            show_tool_calls=True,
            markdown=True,
            description="You are a technical analysis team that collaborates on trading insights.",
            instructions=[
                "You will analyze the K-line chart and provide trading advice.",
                "You will use MACD, Bollinger Bands, and RSI indicators for your analysis.",
                "You will work together to provide a comprehensive analysis.",
                "You will provide a summary of your analysis and trading advice.",
            ],
            show_members_responses=True,
        )

    def analyze_image(
        self,
        kline_date: str = None,
        image_path: str = None,
        prompt: str = "Please analyze this K-line chart and provide your trading advice.",
    ):
        """Analyzes a single image and returns its path and AI analysis result."""
        if image_path is None:
            image_path = f"data/btc_daily_refactored/btc_daily_{kline_date}_len120.png"

        if not os.path.exists(image_path):
            return {"image": image_path, "result": f"Image not found at {image_path}."}

        if not os.path.isfile(image_path):
            return {"image": image_path, "result": f"Path {image_path} is not a file."}

        images = [Image(filepath=image_path)]
        response = asyncio.run(
            self.technical_team.arun(
                message=prompt,
                images=images,
                stream=False,
                stream_intermediate_steps=False,
            )
        )
        return response.content


if __name__ == "__main__":
    # Example usage
    analysis = BasicTechnicalAnalysis()
    result = analysis.analyze_image(kline_date="20180116")
    print(result)
