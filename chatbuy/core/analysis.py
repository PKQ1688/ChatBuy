# from textwrap import dedent

import asyncio
import os
from typing import Literal

from agno.agent import Agent
from agno.media import Image
from agno.models.azure import AzureOpenAI
from agno.team.team import Team
from agno.workflow import RunResponse, Workflow
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


class BasicTechnicalAnalysis(Workflow):
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
            macd_prompt = "Provide analysis based on MACD indicators."
        if bb_prompt is None:
            bb_prompt = "Provide analysis based on Bollinger Bands indicators."
        if rsi_prompt is None:
            rsi_prompt = "Provide analysis based on RSI indicators."

        self.model = AzureOpenAI(id=model_name, temperature=temperature)

        macd_agent: Agent = Agent(
            name="MACD Agent",
            role=macd_prompt,
            model=self.model,
        )
        bb_agent: Agent = Agent(
            name="BB Agent",
            role=bb_prompt,
            model=self.model,
        )
        rsi_agent: Agent = Agent(
            name="RSI Agent",
            role=rsi_prompt,
            model=self.model,
        )

        self.technical_team: Team = Team(
            name="Technical Analysis Team",
            mode="coordinate",  # collaborate, coordinate, route
            model=self.model,
            members=[macd_agent, bb_agent, rsi_agent],
            show_tool_calls=True,
            # markdown=True,
            description="You are a technical analysis team that collaborates on trading insights.",
            instructions=[
                "You will analyze the K-line chart and provide trading advice.",
                "You will use MACD, Bollinger Bands, and RSI indicators for your analysis.",
                "You will work together to provide a comprehensive analysis.",
                "You will provide a summary of your analysis and trading advice.",
            ],
            show_members_responses=False,
            debug_mode=False,
        )

    def run(
        self,
        kline_date: str = None,
        image_path: str = None,
        prompt: str = "Please analyze this K-line chart and provide your trading advice.",
    ) -> TradeAdvice:
        """Analyzes a single image and returns its path and AI analysis result."""
        if image_path is None:
            image_path = f"data/btc_daily_refactored/btc_daily_{kline_date}_len120.png"

        if not os.path.exists(image_path):
            raise {"image": image_path, "result": f"Image not found at {image_path}."}

        if not os.path.isfile(image_path):
            raise {"image": image_path, "result": f"Path {image_path} is not a file."}

        images = [Image(filepath=image_path)]
        response: RunResponse = asyncio.run(
            self.technical_team.arun(
                message=prompt,
                images=images,
                stream=False,
                stream_intermediate_steps=False,
            )
        )

        final_analysis = response.content
        return final_analysis


class AnalysisRunner:
    """A class to run the analysis and provide trading advice."""

    def __init__(self, model_name: str = "gpt-4o-1120", temperature: float = 0.1):
        self.analysis = BasicTechnicalAnalysis(
            model_name=model_name, temperature=temperature
        )
        self.model = AzureOpenAI(id=model_name, temperature=temperature)
        self.final_agent = Agent(
            description="You need to provide the final trading advice based on the previous analysis.",
            model=self.model,
            response_model=TradeAdvice,
            debug_mode=True,
            # use_json_mode=True,
        )

    def run(
        self,
        kline_date: str = None,
        image_path: str = None,
        prompt: str = "Please analyze this K-line chart and provide your trading advice.",
    ) -> TradeAdvice:
        analysis_res = self.analysis.run(
            kline_date=kline_date,
            image_path=image_path,
            prompt=prompt,
        )

        final_response = self.final_agent.run(message=analysis_res)
        final_res = final_response.content

        return final_res


if __name__ == "__main__":
    # Example usage
    analysis = AnalysisRunner()
    result = analysis.run(kline_date="20180116")
    print(result)
