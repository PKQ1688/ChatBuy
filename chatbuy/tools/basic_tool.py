class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "text",
        },
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!",
            "type": "text",
        },
    }
    output_type = "text"
    md_converter = MarkdownConverter()

    def forward(self, file_path, question: Optional[str] = None, initial_exam_mode: Optional[bool] = False) -> str:

        result = self.md_converter.convert(file_path)

        if file_path[-4:] in ['.png', '.jpg']:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content
        
        if not question:
            return result.text_content
        
        if initial_exam_mode:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": "Here is a file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": MessageRole.USER,
                    "content": question,
                },
            ]
            return websurfer_llm_engine(messages)
        else:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": "You will have to write a short caption for this file, then answer this question:"
                    + question,
                },
                {
                    "role": MessageRole.USER,
                    "content": "Here is the complete file:\n### "
                    + str(result.title)
                    + "\n\n"
                    + result.text_content[:70000],
                },
                {
                    "role": MessageRole.USER,
                    "content": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                    + question,
                },
            ]
            return websurfer_llm_engine(messages)