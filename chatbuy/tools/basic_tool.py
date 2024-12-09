import pandas as pd

from transformers.agents.tools import Tool
from transformers.agents.llm_engine import MessageRole

from scripts.llm_engines import llm_engine_0806 as llm_engine


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".csv"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "text",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.csv'.",
            "type": "text",
        },
    }
    output_type = "text"

    def forward(
        self,
        file_path,
        question: str | None = None,
    ) -> str:
        result = pd.read_csv(file_path).to_markdown(index=False)

        if not question:
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": "Here is a file:\n### "
                + str(result.title)
                + "\n\n"
                + result.text_content,
            },
            {
                "role": MessageRole.USER,
                "content": question,
            },
        ]
        return llm_engine(messages)
