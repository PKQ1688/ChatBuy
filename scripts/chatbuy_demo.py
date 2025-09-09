#!/usr/bin/env python3
"""Demo script for ChatBuy - Interactive Quantitative Trading System."""

from chatbuy.ui.cli import ChatBuyCLI


def main():
    """Main entry point for the demo."""
    cli = ChatBuyCLI()
    cli.run()


if __name__ == "__main__":
    main()