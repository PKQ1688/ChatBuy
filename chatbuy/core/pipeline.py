import os

import pandas as pd

from chatbuy.logger import logger

try:
    from scripts.get_crypto_data import fetch_historical_data
except ImportError:
    logger.warning(
        "Warning: Could not import 'fetch_historical_data' from 'scripts.get_crypto_data'. Step 1 will be unavailable."
    )
    fetch_historical_data = None

try:
    # Fix: Import the actual existing function visualize_btc_with_indicators
    from chatbuy.core.visualize_indicators import visualize_btc_with_indicators
except ImportError:
    logger.warning(
        "Warning: Could not import 'visualize_btc_with_indicators' from 'chatbuy.core.visualize_indicators'. Step 2 will be unavailable."
    )
    visualize_btc_with_indicators = None

try:
    # Fix: Import the TradePipeline class from und_img
    from chatbuy.core.und_img import (
        TradePipeline as UndImgTradePipeline,
    )  # Use an alias to avoid conflict with this class name
except ImportError:
    logger.warning(
        "Warning: Could not import 'TradePipeline' from 'chatbuy.core.und_img'. Step 3 will be unavailable."
    )
    UndImgTradePipeline = None

try:
    # Fix: Import the evaluate_signals function
    from chatbuy.core.evaluate_trade_signals import evaluate_signals
except ImportError:
    logger.warning(
        "Warning: Could not import 'evaluate_signals' from 'chatbuy.core.evaluate_trade_signals'. Step 4 will be unavailable."
    )
    evaluate_signals = None


# --- Configuration (consider moving to a dedicated config file or class attributes) ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
IMAGE_FILE = os.path.join(OUTPUT_DIR, "kline_plot.png")
AI_RESULT_FILE = os.path.join(
    OUTPUT_DIR, "trade_advice_results.csv"
)  # Currently unused, AI function might return results directly
REPORT_FILE = os.path.join(
    OUTPUT_DIR, "evaluation_report.txt"
)  # Currently unused, report function might return content directly


class TradingAnalysisPipeline:
    """Encapsulates the entire trading strategy analysis workflow.

    Manages state and passes data between steps.
    Returns a dictionary containing status and results/errors.
    """

    def __init__(self, use_openrouter: bool = False):  # Allow configuration of the AI model
        """Initialize, ensure directories exist, and initialize the AI Pipeline."""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Initialize AI Pipeline instance
        if UndImgTradePipeline:
            try:
                # Pass the use_openrouter parameter
                self.ai_pipeline = UndImgTradePipeline(use_openrouter=use_openrouter)
                logger.info(
                    f"AI Pipeline initialized using {'OpenRouter' if use_openrouter else 'Azure OpenAI (default)'}."
                )
            except Exception as e:
                logger.error(f"Failed to initialize AI Pipeline from und_img: {e}")
                self.ai_pipeline = None
        else:
            self.ai_pipeline = None

    def run_step_1_fetch_data(self, **kwargs):
        """Execute Step 1: Fetch Candlestick Data.

        Args:
            **kwargs: Keyword arguments passed to fetch_historical_data (e.g., symbol, timeframe, start_date).

        Returns:
            A dictionary containing the execution status and results:
            {"success": bool, "result": pd.DataFrame | None, "error": str | None}.
        """
        if not fetch_historical_data:
            return {
                "success": False,
                "result": None,
                "error": "The function to fetch data (fetch_historical_data) could not be imported successfully.",
            }

        try:
            # Fix: Call fetch_historical_data and provide default parameters (or get from kwargs)
            symbol = kwargs.get("symbol", "BTC/USDT")  # Note: ccxt uses '/'
            timeframe = kwargs.get("timeframe", "1d")
            start_date = kwargs.get("start_date", "2017-07-01T00:00:00Z")

            logger.info(
                f"Pipeline: Calling fetch_historical_data(symbol='{symbol}', timeframe='{timeframe}', start_date='{start_date}')..."
            )
            result_df = fetch_historical_data(
                symbol=symbol, timeframe=timeframe, start_date=start_date
            )
            logger.info(
                f"Pipeline: fetch_historical_data returned type: {type(result_df)}"
            )

            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                return {"success": True, "result": result_df, "error": None}
            elif isinstance(result_df, pd.DataFrame) and result_df.empty:
                return {
                    "success": False,
                    "result": None,
                    "error": "Data fetched successfully, but the returned data is empty.",
                }
            else:
                # This should not happen, as the function is designed to return a DataFrame
                return {
                    "success": False,
                    "result": None,
                    "error": f"fetch_historical_data returned an unexpected type: {type(result_df)}",
                }
        except Exception as e:
            error_msg = f"Error calling `fetch_historical_data`:\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "result": None, "error": error_msg}

    def run_step_2_generate_image(self, data_input):
        """Execute Step 2: Generate Candlestick Image.

        Args:
            data_input: Data obtained from Step 1 (DataFrame or path).

        Returns:
            A dictionary containing the execution status and results:
            {"success": bool, "image_path": str | None, "error": str | None}.
        """
        if not visualize_btc_with_indicators:
            return {
                "success": False,
                "image_path": None,
                "error": "The function to generate images (visualize_btc_with_indicators) could not be imported successfully.",
            }
        if data_input is None:
            return {
                "success": False,
                "image_path": None,
                "error": "Error: No valid data input provided for image generation.",
            }

        try:
            # Fix: Call visualize_btc_with_indicators and pass the preset full output path
            logger.info(
                f"Pipeline: Calling visualize_btc_with_indicators with output path: {IMAGE_FILE}..."
            )
            # Function now accepts data and output_file_path
            returned_path = visualize_btc_with_indicators(
                data_input, output_file_path=IMAGE_FILE
            )
            logger.info(
                f"Pipeline: visualize_btc_with_indicators returned: {returned_path}"
            )

            # Verify if the returned path matches the expectation and the file exists
            if returned_path == IMAGE_FILE and os.path.exists(returned_path):
                logger.info(
                    f"Pipeline: Image successfully generated and found at {returned_path}"
                )
                return {"success": True, "image_path": returned_path, "error": None}
            elif returned_path == IMAGE_FILE and not os.path.exists(returned_path):
                error_msg = (
                    f"Function call succeeded and returned expected path {returned_path}, but the file was not found."
                )
                logger.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}
            elif returned_path != IMAGE_FILE:
                error_msg = f"Function returned an unexpected path '{returned_path}' instead of the expected '{IMAGE_FILE}'."
                logger.warning(f"Pipeline Warning: {error_msg}")
                # Try checking if the returned path exists
                if os.path.exists(returned_path):
                    logger.info(
                        f"Pipeline: Image found at unexpected path {returned_path}. Using this path."
                    )
                    return {
                        "success": True,
                        "image_path": returned_path,
                        "error": None,
                    }  # Still considered successful, but the path is unexpected
                else:
                    logger.error(
                        f"Pipeline Error: Image not found at unexpected path {returned_path} either."
                    )
                    return {
                        "success": False,
                        "image_path": None,
                        "error": error_msg + " File also not found.",
                    }
            else:  # returned_path is None or not a string (should not happen based on function modification)
                error_msg = "visualize_btc_with_indicators returned None or a non-string value."
                logger.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error calling `visualize_btc_with_indicators`:\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "image_path": None, "error": error_msg}

    def run_step_3_analyze_signals(self, image_path: str, strategy: str | None = None):
        """Execute Step 3: Analyze buy/sell points using the AI Pipeline.

        Args:
            image_path: Path to the image generated in Step 2.
            strategy: Optional trading strategy description string.

        Returns:
            A dictionary containing the execution status and results:
            {"success": bool, "result": TradeAdvice | None, "error": str | None}.
        """
        if not self.ai_pipeline:
            return {
                "success": False,
                "result": None,
                "error": "AI Analysis Pipeline failed to initialize successfully.",
            }
        if not image_path or not os.path.exists(image_path):
            return {
                "success": False,
                "result": None,
                "error": f"Error: No valid image path provided for AI analysis: {image_path}",
            }

        try:
            # Fix: Call the run_pipeline method of und_img.TradePipeline
            logger.info(
                f"Pipeline: Calling AI Pipeline (und_img) with image: {image_path}..."
            )
            # If no strategy is provided, use the default strategy from und_img
            call_args = {"image_path": image_path}
            if strategy:
                call_args["strategy"] = strategy
                logger.info(f"Using custom strategy: {strategy}")
            else:
                logger.info("Using default strategy from und_img.")

            trade_advice = self.ai_pipeline.run_pipeline(**call_args)
            logger.info(
                f"Pipeline: AI Pipeline returned: Action={trade_advice.action}, Reason={trade_advice.reason}"
            )

            # run_pipeline returns a TradeAdvice object on success
            return {"success": True, "result": trade_advice, "error": None}
        except Exception as e:
            error_msg = f"Error calling AI Pipeline (und_img):\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "result": None, "error": error_msg}

    def run_step_4_generate_report(self, price_df: pd.DataFrame):
        """Execute Step 4: Evaluate trading signals using evaluate_signals.

        Note:
            Currently assumes the signal file 'output/trade_advice_unified_results_one.csv' exists.

        Args:
            price_df: Price DataFrame obtained from Step 1.

        Returns:
            A dictionary containing the execution status and report:
            {"success": bool, "report": dict | None, "error": str | None}.
            The report dictionary contains evaluation results (total_trades, total_profit, etc.).
        """
        if not evaluate_signals:
            return {
                "success": False,
                "report": None,
                "error": "The function to evaluate signals (evaluate_signals) could not be imported successfully.",
            }
        if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
            return {
                "success": False,
                "report": None,
                "error": "Error: No valid price DataFrame provided for evaluation.",
            }

        signals_path = "output/trade_advice_unified_results_one.csv"
        # Need a temporary price file path
        temp_prices_path = os.path.join(OUTPUT_DIR, "temp_prices_for_eval.csv")

        try:
            # Check if the signal file exists
            if not os.path.exists(signals_path):
                return {
                    "success": False,
                    "report": None,
                    "error": f"Signal file not found: {signals_path}",
                }

            # Save the price DataFrame to a temporary file
            logger.info(
                f"Pipeline: Saving price data to temporary file: {temp_prices_path}"
            )
            price_df.to_csv(temp_prices_path, index=False)

            # Call the evaluation function
            logger.info(
                f"Pipeline: Calling evaluate_signals with signals='{signals_path}' and prices='{temp_prices_path}'..."
            )
            evaluation_result = evaluate_signals(
                signals_path=signals_path, prices_path=temp_prices_path
            )
            logger.info(
                f"Pipeline: evaluate_signals returned success={evaluation_result.get('success')}"
            )

            # evaluate_signals internally handles file reading errors etc., and returns success/error
            if evaluation_result.get("success"):
                # Wrap the evaluation result dictionary under the 'report' key
                return {"success": True, "report": evaluation_result, "error": None}
            else:
                # Directly return the error message from evaluate_signals
                return {
                    "success": False,
                    "report": None,
                    "error": evaluation_result.get("error", "Unknown error occurred during signal evaluation."),
                }

        except Exception as e:
            # Catch unexpected errors when saving the temporary file or calling the evaluation function
            error_msg = f"Error executing evaluation step:\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "report": None, "error": error_msg}
        finally:
            # Clean up the temporary price file
            if os.path.exists(temp_prices_path):
                try:
                    os.remove(temp_prices_path)
                    logger.info(
                        f"Pipeline: Removed temporary price file: {temp_prices_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Pipeline Warning: Failed to remove temporary price file {temp_prices_path}: {e}"
                    )


# --- Helper functions (if needed) ---
# e.g., helper functions for reading files or processing data can be placed here
