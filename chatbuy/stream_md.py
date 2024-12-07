import asyncio

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from pydantic_ai import Agent
from chatbuy.llm_models import model_4o_mini, model_0806, model_4o

agent = Agent()


models = [model_4o_mini, model_0806, model_4o]


async def main():
    prettier_code_blocks()
    console = Console()
    prompt = "Show me a short example of using Pydantic."
    console.log(f"Asking: {prompt}...", style="cyan")
    for model in models:
        console.log(f"Using model: {model}")
        with Live("", console=console, vertical_overflow="visible") as live:
            async with agent.run_stream(prompt, model=model) as result:
                async for message in result.stream():
                    live.update(Markdown(message))
        console.log(result.cost())


def prettier_code_blocks():
    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style="dim")
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color="default",
                word_wrap=True,
            )
            yield Text(f"/{self.lexer_name}", style="dim")

    Markdown.elements["fence"] = SimpleCodeBlock


if __name__ == "__main__":
    asyncio.run(main())
