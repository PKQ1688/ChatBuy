#!/usr/bin/env python3
"""Launch script for ChatBuy Gradio web interface."""

from chatbuy.ui.gradio_app import ChatBuyGradio


def main():
    """Main entry point for the Gradio app."""
    app = ChatBuyGradio()
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Use port 7861 to avoid conflicts
        share=False,  # Set to True to create public link
        inbrowser=True,  # Auto-open browser
        show_error=True,  # Show detailed errors
    )


if __name__ == "__main__":
    main()
