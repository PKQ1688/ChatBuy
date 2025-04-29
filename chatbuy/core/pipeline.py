import os

import pandas as pd
from scripts.get_crypto_data import fetch_historical_data

# from chatbuy.core.evaluate_trade_signals import evaluate_signals
from chatbuy.core.und_img import TradePipeline as UndImgTradePipeline
from chatbuy.core.visualize_indicators import IndicatorVisualizer
from chatbuy.logger import log

# --- Configuration (consider moving to a dedicated config file or class attributes) ---
DATA_DIR = "data"
CACHE_SUBDIR = os.path.join(DATA_DIR, "cache")
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

    def __init__(
        self, use_openrouter: bool = False
    ):  # Allow configuration of the AI model
        """Initialize, ensure directories exist, and initialize the AI Pipeline."""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CACHE_SUBDIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Initialize AI Pipeline instance
        if UndImgTradePipeline:
            try:
                # Pass the use_openrouter parameter
                self.ai_pipeline = UndImgTradePipeline(use_openrouter=use_openrouter)
                log.info(
                    f"AI Pipeline initialized using {'OpenRouter' if use_openrouter else 'Azure OpenAI (default)'}."
                )
            except Exception as e:
                log.error(f"Failed to initialize AI Pipeline from und_img: {e}")
                self.ai_pipeline = None
        else:
            self.ai_pipeline = None

    def run_step_1_fetch_data(self, **kwargs):
        """Execute Step 1: Fetch Candlestick Data, with local cache support.

        Args:
            **kwargs: Keyword arguments passed to fetch_historical_data (e.g., symbol, timeframe, start_date, end_date).

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
            import hashlib

            symbol = kwargs.get("symbol", "BTC/USDT")
            timeframe = kwargs.get("timeframe", "1d")
            start_date = kwargs.get("start_date", "2017-07-01T00:00:00Z")
            end_date = kwargs.get("end_date", None)

            # 生成唯一缓存文件名
            from datetime import datetime

            def safe_str(s, only_date=False):
                s = str(s).replace("/", "_").replace(":", "-")
                if only_date:
                    # 只保留到日，优先直接截取前10位
                    if len(s) >= 10:
                        return s[:10]
                    try:
                        dt = datetime.fromisoformat(s.replace("Z", ""))
                        return dt.strftime("%Y-%m-%d")
                    except Exception:
                        log.error(f"Failed to parse date string: {s}")
                return s

            cache_key = f"{safe_str(symbol)}_{safe_str(timeframe)}_{safe_str(start_date, only_date=True)}"
            if end_date:
                cache_key += f"_{safe_str(end_date, only_date=True)}"
            # 防止文件名过长
            cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
            cache_file = os.path.join(CACHE_SUBDIR, f"{cache_key}_{cache_hash}.csv")

            # 检查本地缓存
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, parse_dates=["timestamp"])
                    # end_date 过滤（兼容旧缓存）
                    if end_date:
                        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
                    if not df.empty:
                        log.info(f"Loaded cached data from {cache_file}")
                        return {"success": True, "result": df, "error": None}
                    else:
                        log.warning(f"Cached file {cache_file} is empty, will refetch.")
                except Exception as e:
                    log.warning(
                        f"Failed to read cache file {cache_file}: {e}, will refetch."
                    )

            # 获取数据
            log.info(
                f"Pipeline: Calling fetch_historical_data(symbol='{symbol}', timeframe='{timeframe}', start_date='{start_date}')..."
            )
            result_df = fetch_historical_data(
                symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
            )
            # end_date 过滤
            if end_date and isinstance(result_df, pd.DataFrame):
                result_df = result_df[
                    result_df["timestamp"] <= pd.to_datetime(end_date)
                ]

            log.info(
                f"Pipeline: fetch_historical_data returned type: {type(result_df)}"
            )

            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                # 计算技术指标
                try:
                    from talipp.indicators import BB, MACD

                    macd = MACD(12, 26, 9, result_df["close"])[:]
                    bb = BB(20, 2, result_df["close"])[:]
                    for index, row in result_df.iterrows():
                        if macd[index] is not None:
                            result_df.at[index, "macd"] = macd[index].macd
                            result_df.at[index, "signal"] = macd[index].signal
                            result_df.at[index, "histogram"] = macd[index].histogram
                        if bb[index] is not None:
                            result_df.at[index, "bb_upper"] = bb[index].ub
                            result_df.at[index, "bb_middle"] = bb[index].cb
                            result_df.at[index, "bb_lower"] = bb[index].lb
                    result_df = result_df.infer_objects(copy=False)
                    result_df.fillna(0, inplace=True)
                except Exception as e:
                    log.warning(f"Failed to calculate indicators: {e}")

                # 保存到本地
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    result_df.to_csv(cache_file, index=False)
                    log.info(f"Saved fetched data to {cache_file}")
                except Exception as e:
                    log.warning(f"Failed to save data to cache file {cache_file}: {e}")
                return {"success": True, "result": result_df, "error": None}
            elif isinstance(result_df, pd.DataFrame) and result_df.empty:
                return {
                    "success": False,
                    "result": None,
                    "error": "Data fetched successfully, but the returned data is empty.",
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"fetch_historical_data returned an unexpected type: {type(result_df)}",
                }
        except Exception as e:
            error_msg = f"Error calling `fetch_historical_data`:\n{e}"
            log.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "result": None, "error": error_msg}

    def run_step_2_generate_image(self, data_input):
        """Execute Step 2: Generate Candlestick Image.

        Args:
            data_input: Data obtained from Step 1 (DataFrame or path).

        Returns:
            A dictionary containing the execution status and results:
            {"success": bool, "image_path": str | None, "error": str | None}.
        """
        if not IndicatorVisualizer:
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
            log.info(
                f"Pipeline: Calling IndicatorVisualizer().visualize with output path: {IMAGE_FILE}..."
            )
            visualizer = IndicatorVisualizer()
            returned_path = visualizer.visualize(
                data_input, output_file_path=IMAGE_FILE
            )
            log.info(
                f"Pipeline: IndicatorVisualizer.visualize returned: {returned_path}"
            )

            # Verify if the returned path matches the expectation and the file exists
            if returned_path == IMAGE_FILE and os.path.exists(returned_path):
                log.info(
                    f"Pipeline: Image successfully generated and found at {returned_path}"
                )
                return {"success": True, "image_path": returned_path, "error": None}
            elif returned_path == IMAGE_FILE and not os.path.exists(returned_path):
                error_msg = f"Function call succeeded and returned expected path {returned_path}, but the file was not found."
                log.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}
            elif returned_path != IMAGE_FILE:
                error_msg = f"Function returned an unexpected path '{returned_path}' instead of the expected '{IMAGE_FILE}'."
                log.warning(f"Pipeline Warning: {error_msg}")
                # Try checking if the returned path exists
                if os.path.exists(returned_path):
                    log.info(
                        f"Pipeline: Image found at unexpected path {returned_path}. Using this path."
                    )
                    return {
                        "success": True,
                        "image_path": returned_path,
                        "error": None,
                    }  # Still considered successful, but the path is unexpected
                else:
                    log.error(
                        f"Pipeline Error: Image not found at unexpected path {returned_path} either."
                    )
                    return {
                        "success": False,
                        "image_path": None,
                        "error": error_msg + " File also not found.",
                    }
            else:  # returned_path is None or not a string (should not happen based on function modification)
                error_msg = (
                    "visualize_btc_with_indicators returned None or a non-string value."
                )
                log.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}

        except Exception as e:
            error_msg = f"Error calling `visualize_btc_with_indicators`:\n{e}"
            log.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "image_path": None, "error": error_msg}

    def run_step_2_generate_images_batch(
        self,
        data_input,
        output_dir: str,
        length: int = 120,
        step: int = 1,
        start_time: str = None,
        end_time: str = None,
        filename_prefix: str = "chart",
    ):
        """Batch generate candlestick images using a sliding window approach and save them to a specified folder.
        
        Args:
            data_input: DataFrame or path to a CSV file
            output_dir: Output folder for the images
            length: Number of candlesticks per image
            step: Sliding window step size
            start_time: Start time (optional, YYYY-MM-DD)
            end_time: End time (optional, YYYY-MM-DD)
            filename_prefix: Prefix for the filenames
        Returns:
            dict: {success, output_dir, count, error}
        """
        import shutil
        import tempfile

        visualizer = IndicatorVisualizer()
        temp_csv = None
        try:
            # 判断输入类型
            if isinstance(data_input, pd.DataFrame):
                # 保存为临时csv
                temp_dir = tempfile.mkdtemp()
                temp_csv = os.path.join(temp_dir, "temp_data.csv")
                data_input.to_csv(temp_csv, index=False)
                data_path = temp_csv
            elif isinstance(data_input, str) and os.path.exists(data_input):
                data_path = data_input
            else:
                return {
                    "success": False,
                    "output_dir": None,
                    "count": 0,
                    "error": "data_input 必须为DataFrame或有效csv路径",
                }

            os.makedirs(output_dir, exist_ok=True)
            visualizer.batch_generate(
                data_path=data_path,
                output_dir=output_dir,
                length=length,
                step=step,
                start_time=start_time,
                end_time=end_time,
                filename_prefix=filename_prefix,
                show=False,
            )
            # 统计生成图片数量
            count = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
            return {
                "success": True,
                "output_dir": output_dir,
                "count": count,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "output_dir": output_dir,
                "count": 0,
                "error": str(e),
            }
        finally:
            if temp_csv and os.path.exists(temp_csv):
                shutil.rmtree(os.path.dirname(temp_csv), ignore_errors=True)
