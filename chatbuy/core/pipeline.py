import os
import logging

import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(
            os.path.join("output", "pipeline.log"), mode="a"
        ),  # 输出到文件
    ],
)
logger = logging.getLogger("TradingPipeline")

# --- 导入底层实现函数 ---
# !! 这些导入需要根据实际情况调整 !!
# !! 假设这些函数存在且可导入 !!
try:
    # 修正：导入实际存在的函数 fetch_historical_data
    from scripts.get_crypto_data import fetch_historical_data
except ImportError:
    logger.warning(
        "Warning: Could not import 'fetch_historical_data' from 'scripts.get_crypto_data'. Step 1 will be unavailable."
    )
    fetch_historical_data = None

try:
    # 修正：导入实际存在的函数 visualize_btc_with_indicators
    from chatbuy.core.visualize_indicators import visualize_btc_with_indicators
except ImportError:
    logger.warning(
        "Warning: Could not import 'visualize_btc_with_indicators' from 'chatbuy.core.visualize_indicators'. Step 2 will be unavailable."
    )
    visualize_btc_with_indicators = None

try:
    # 修正：导入 und_img 中的 TradePipeline 类
    from chatbuy.core.und_img import (
        TradePipeline as UndImgTradePipeline,
    )  # 使用别名避免与此类名冲突
except ImportError:
    logger.warning(
        "Warning: Could not import 'TradePipeline' from 'chatbuy.core.und_img'. Step 3 will be unavailable."
    )
    UndImgTradePipeline = None

try:
    # 修正：导入 evaluate_signals 函数
    from chatbuy.core.evaluate_trade_signals import evaluate_signals
except ImportError:
    logger.warning(
        "Warning: Could not import 'evaluate_signals' from 'chatbuy.core.evaluate_trade_signals'. Step 4 will be unavailable."
    )
    evaluate_signals = None


# --- 配置 (可以考虑移到专门的配置文件或类属性) ---
DATA_DIR = "data"
OUTPUT_DIR = "output"
IMAGE_FILE = os.path.join(OUTPUT_DIR, "kline_plot.png")
AI_RESULT_FILE = os.path.join(
    OUTPUT_DIR, "trade_advice_results.csv"
)  # 暂时未使用，AI函数可能直接返回结果
REPORT_FILE = os.path.join(
    OUTPUT_DIR, "evaluation_report.txt"
)  # 暂时未使用，报告函数可能直接返回内容


class TradingAnalysisPipeline:
    """
    封装交易策略分析的整个流程。
    管理状态并在步骤之间传递数据。
    返回包含状态和结果/错误的字典。
    """

    def __init__(self, use_openrouter: bool = False):  # 允许配置 AI 模型
        """初始化，确保目录存在，并初始化 AI Pipeline。"""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 初始化 AI Pipeline 实例
        if UndImgTradePipeline:
            try:
                # 传递 use_openrouter 参数
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
        """
        执行第一步：获取K线数据。
        返回: {"success": bool, "result": pd.DataFrame | None, "error": str | None}
        """
        if not fetch_historical_data:
            return {
                "success": False,
                "result": None,
                "error": "获取数据的函数 (fetch_historical_data) 未能成功导入。",
            }

        try:
            # 修正：调用 fetch_historical_data 并提供默认参数 (或从 kwargs 获取)
            symbol = kwargs.get("symbol", "BTC/USDT")  # 注意 ccxt 使用 '/'
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
                    "error": "获取数据成功，但返回的数据为空。",
                }
            else:
                # 这不应该发生，因为函数设计为返回 DataFrame
                return {
                    "success": False,
                    "result": None,
                    "error": f"fetch_historical_data 返回了意外的类型: {type(result_df)}",
                }
        except Exception as e:
            error_msg = f"调用 `fetch_historical_data` 出错：\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "result": None, "error": error_msg}

    def run_step_2_generate_image(self, data_input):
        """
        执行第二步：生成K线图片。
        参数: data_input: 第一步获取的数据 (DataFrame 或路径)
        返回: {"success": bool, "image_path": str | None, "error": str | None}
        """
        if not visualize_btc_with_indicators:
            return {
                "success": False,
                "image_path": None,
                "error": "生成图片的函数 (visualize_btc_with_indicators) 未能成功导入。",
            }
        if data_input is None:
            return {
                "success": False,
                "image_path": None,
                "error": "错误：未提供有效的数据输入用于生成图片。",
            }

        try:
            # 修正：调用 visualize_btc_with_indicators 并传递预设的完整输出路径
            logger.info(
                f"Pipeline: Calling visualize_btc_with_indicators with output path: {IMAGE_FILE}..."
            )
            # 函数现在接受 data 和 output_file_path
            returned_path = visualize_btc_with_indicators(
                data_input, output_file_path=IMAGE_FILE
            )
            logger.info(
                f"Pipeline: visualize_btc_with_indicators returned: {returned_path}"
            )

            # 验证返回的路径是否与预期一致且文件存在
            if returned_path == IMAGE_FILE and os.path.exists(returned_path):
                logger.info(
                    f"Pipeline: Image successfully generated and found at {returned_path}"
                )
                return {"success": True, "image_path": returned_path, "error": None}
            elif returned_path == IMAGE_FILE and not os.path.exists(returned_path):
                error_msg = (
                    f"函数调用成功并返回预期路径 {returned_path}，但文件未找到。"
                )
                logger.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}
            elif returned_path != IMAGE_FILE:
                error_msg = f"函数返回了意外的路径 '{returned_path}' 而不是预期的 '{IMAGE_FILE}'。"
                logger.warning(f"Pipeline Warning: {error_msg}")
                # 尝试检查返回的路径是否存在
                if os.path.exists(returned_path):
                    logger.info(
                        f"Pipeline: Image found at unexpected path {returned_path}. Using this path."
                    )
                    return {
                        "success": True,
                        "image_path": returned_path,
                        "error": None,
                    }  # 仍然认为是成功，但路径非预期
                else:
                    logger.error(
                        f"Pipeline Error: Image not found at unexpected path {returned_path} either."
                    )
                    return {
                        "success": False,
                        "image_path": None,
                        "error": error_msg + " 文件也未找到。",
                    }
            else:  # returned_path is None or not a string (should not happen based on function modification)
                error_msg = f"visualize_btc_with_indicators 返回了 None 或非字符串值。"
                logger.error(f"Pipeline Error: {error_msg}")
                return {"success": False, "image_path": None, "error": error_msg}

        except Exception as e:
            error_msg = f"调用 `visualize_btc_with_indicators` 出错：\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "image_path": None, "error": error_msg}

    def run_step_3_analyze_signals(self, image_path: str, strategy: str | None = None):
        """
        执行第三步：使用 und_img.py 中的 AI Pipeline 分析买卖点。
        参数:
            image_path: 第二步生成的图片路径
            strategy: 可选的交易策略描述字符串
        返回: {"success": bool, "result": TradeAdvice | None, "error": str | None}
        """
        if not self.ai_pipeline:
            return {
                "success": False,
                "result": None,
                "error": "AI 分析 Pipeline 未能成功初始化。",
            }
        if not image_path or not os.path.exists(image_path):
            return {
                "success": False,
                "result": None,
                "error": f"错误：未提供有效的图片路径用于AI分析: {image_path}",
            }

        try:
            # 修正：调用 und_img.TradePipeline 的 run_pipeline 方法
            logger.info(
                f"Pipeline: Calling AI Pipeline (und_img) with image: {image_path}..."
            )
            # 如果未提供策略，使用 und_img 中的默认策略
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

            # run_pipeline 成功时返回 TradeAdvice 对象
            return {"success": True, "result": trade_advice, "error": None}
        except Exception as e:
            error_msg = f"调用 AI Pipeline (und_img) 出错：\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "result": None, "error": error_msg}

    def run_step_4_generate_report(self, price_df: pd.DataFrame):
        """
        执行第四步：使用 evaluate_signals 评估交易信号。
        注意：当前假设信号文件 'output/trade_advice_unified_results_one.csv' 已存在。
        参数: price_df: 第一步获取的价格 DataFrame
        返回: {"success": bool, "report": dict | None, "error": str | None}
              report 字典包含评估结果 (total_trades, total_profit, etc.)
        """
        if not evaluate_signals:
            return {
                "success": False,
                "report": None,
                "error": "评估信号的函数 (evaluate_signals) 未能成功导入。",
            }
        if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
            return {
                "success": False,
                "report": None,
                "error": "错误：未提供有效的价格 DataFrame 用于评估。",
            }

        signals_path = "output/trade_advice_unified_results_one.csv"
        # 需要一个临时的价格文件路径
        temp_prices_path = os.path.join(OUTPUT_DIR, "temp_prices_for_eval.csv")

        try:
            # 检查信号文件是否存在
            if not os.path.exists(signals_path):
                return {
                    "success": False,
                    "report": None,
                    "error": f"信号文件未找到: {signals_path}",
                }

            # 保存价格 DataFrame 到临时文件
            logger.info(
                f"Pipeline: Saving price data to temporary file: {temp_prices_path}"
            )
            price_df.to_csv(temp_prices_path, index=False)

            # 调用评估函数
            logger.info(
                f"Pipeline: Calling evaluate_signals with signals='{signals_path}' and prices='{temp_prices_path}'..."
            )
            evaluation_result = evaluate_signals(
                signals_path=signals_path, prices_path=temp_prices_path
            )
            logger.info(
                f"Pipeline: evaluate_signals returned success={evaluation_result.get('success')}"
            )

            # evaluate_signals 内部已经处理了文件读取等错误，并返回了 success/error
            if evaluation_result.get("success"):
                # 将评估结果字典包装在 report 键下
                return {"success": True, "report": evaluation_result, "error": None}
            else:
                # 直接返回 evaluate_signals 的错误信息
                return {
                    "success": False,
                    "report": None,
                    "error": evaluation_result.get("error", "评估信号时发生未知错误。"),
                }

        except Exception as e:
            # 捕获保存临时文件或调用评估函数时的意外错误
            error_msg = f"执行评估步骤时出错：\n{e}"
            logger.error(f"Pipeline Error: {error_msg}")
            return {"success": False, "report": None, "error": error_msg}
        finally:
            # 清理临时价格文件
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
# 例如，用于读取文件或处理数据的辅助函数可以放在这里
