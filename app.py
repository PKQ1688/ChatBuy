import os

import gradio as gr
import pandas as pd
from chatbuy.logger import log # 导入日志记录器

# --- 导入 Pipeline ---
try:
    from chatbuy.core.pipeline import TradingAnalysisPipeline
except ImportError as e:
    log.error(f"错误：无法导入 TradingAnalysisPipeline", exc_info=True) # 使用 log.error
    # 在 Gradio 中，我们不能像 Streamlit 那样直接停止应用，
    # 但可以在界面上显示错误信息。
    pipeline_import_error = f"无法导入核心处理模块: {e}"
    pipeline = None  # 设置为 None 以便后续检查
else:
    pipeline_import_error = None
    # --- 初始化 Pipeline ---
    # Gradio 应用通常在启动时初始化一次
    pipeline = TradingAnalysisPipeline()

# --- Gradio 应用 ---


def create_gradio_app():
    """创建 Gradio 应用界面和逻辑."""
    if pipeline is None:
        with gr.Blocks() as app:
            gr.Markdown("# 交易策略分析流程 (Pipeline 版)")
            gr.Error(f"应用初始化失败: {pipeline_import_error}")
        return app

    with gr.Blocks(title="交易策略分析流程") as app:
        gr.Markdown("# 交易策略分析流程 (Pipeline 版)")

        # --- 状态管理 (使用 gr.State) ---
        data_result_state = gr.State(None)
        image_path_state = gr.State(None)
        analysis_result_state = gr.State(None)
        report_content_state = gr.State(None)
        # 状态标志 (虽然 Gradio 的流程控制不同，但保留可能有用)
        data_fetched_state = gr.State(False)
        image_generated_state = gr.State(False)
        analysis_done_state = gr.State(False)
        report_generated_state = gr.State(False)

        # --- 步骤一：获取K线数据 ---
        with gr.Tab("第一步：获取数据"):
            with gr.Row():
                fetch_button = gr.Button("获取数据", variant="primary")
                fetch_status = gr.Textbox(
                    "点击按钮开始获取数据...", label="状态", interactive=False
                )
            fetch_output_df = gr.DataFrame(label="数据预览 (前5行)", visible=False)
            fetch_output_path = gr.Textbox(
                label="数据文件路径", visible=False, interactive=False
            )

            def run_fetch_data():
                status_update = gr.update(
                    value="正在调用 Pipeline 获取数据...", interactive=False
                )
                df_update = gr.update(visible=False)
                path_update = gr.update(visible=False)
                next_button_update = gr.update(interactive=False)  # 禁用下一步按钮
                report_button_update = gr.update(interactive=False)  # 禁用报告按钮

                pipeline_result = pipeline.run_step_1_fetch_data()

                if pipeline_result["success"]:
                    result = pipeline_result["result"]
                    data_fetched = True
                    status_update = gr.update(value="数据获取成功！", interactive=False)
                    next_button_update = gr.update(interactive=True)  # 启用生成图片按钮
                    report_button_update = gr.update(interactive=True)  # 启用报告按钮

                    if isinstance(result, pd.DataFrame):
                        df_update = gr.update(value=result.head(), visible=True)
                        path_update = gr.update(visible=False)
                        data_result = result  # 直接存储 DataFrame
                    elif isinstance(result, str) and os.path.exists(result):
                        path_update = gr.update(
                            value=f"数据已保存到: {result}", visible=True
                        )
                        try:
                            df_update = gr.update(
                                value=pd.read_csv(result).head(), visible=True
                            )
                        except Exception as e:
                            log.warning("数据获取成功，但预览失败", exc_info=True) # 添加日志
                            status_update = gr.update(
                                value=f"数据获取成功，但预览失败: {e}",
                                interactive=False,
                            )
                        data_result = result  # 存储文件路径
                    else:
                        status_update = gr.update(
                            value=f"数据获取成功，函数返回: {result}", interactive=False
                        )
                        data_result = result  # 存储其他类型结果
                else:
                    data_fetched = False
                    log.error(f"数据获取失败：{pipeline_result['error']}") # 添加日志
                    status_update = gr.update(
                        value=f"数据获取失败：\n{pipeline_result['error']}",
                        interactive=False,
                    )
                    data_result = None

                # 返回所有需要更新的组件和状态
                return (
                    status_update,
                    df_update,
                    path_update,
                    data_result,  # 更新 data_result_state
                    data_fetched,  # 更新 data_fetched_state
                    next_button_update,  # 更新生成图片按钮状态
                    report_button_update,  # 更新报告按钮状态
                )

        # --- 步骤二：生成K线图片 ---
        with gr.Tab("第二步：生成图片"):
            with gr.Row():
                generate_image_button = gr.Button(
                    "生成图片", variant="primary", interactive=False
                )  # 初始禁用
                image_status = gr.Textbox(
                    "请先完成第一步获取数据。", label="状态", interactive=False
                )
            generated_image = gr.Image(
                label="生成的K线图", type="filepath", visible=False
            )

            def run_generate_image(current_data_result, is_data_fetched):
                if not is_data_fetched:
                    log.warning("需要先获取数据才能生成图片") # 添加日志
                    return (
                        gr.update(value="错误：需要先获取数据。", interactive=False),
                        gr.update(visible=False),
                        None,  # image_path_state
                        False,  # image_generated_state
                        gr.update(interactive=False),  # analyze_button
                    )

                status_update = gr.update(
                    value="正在调用 Pipeline 生成图片...", interactive=False
                )
                image_update = gr.update(visible=False)
                next_button_update = gr.update(interactive=False)  # 禁用下一步按钮

                pipeline_result = pipeline.run_step_2_generate_image(
                    current_data_result
                )

                if pipeline_result["success"]:
                    image_path = pipeline_result["image_path"]
                    image_generated = True
                    status_update = gr.update(value="图片生成成功！", interactive=False)
                    image_update = gr.update(value=image_path, visible=True)
                    next_button_update = gr.update(interactive=True)  # 启用下一步按钮
                else:
                    image_path = None
                    image_generated = False
                    log.error(f"图片生成失败：{pipeline_result['error']}") # 添加日志
                    status_update = gr.update(
                        value=f"图片生成失败：\n{pipeline_result['error']}",
                        interactive=False,
                    )

                return (
                    status_update,
                    image_update,
                    image_path,  # 更新 image_path_state
                    image_generated,  # 更新 image_generated_state
                    next_button_update,  # 更新 AI 分析按钮状态
                )

        # --- 步骤三：AI分析买卖点 ---
        with gr.Tab("第三步：AI分析"):
            with gr.Row():
                analyze_button = gr.Button(
                    "AI分析", variant="primary", interactive=False
                )  # 初始禁用
                analyze_status = gr.Textbox(
                    "请先完成第二步生成图片。", label="状态", interactive=False
                )
            with gr.Row():
                analysis_action = gr.Textbox(
                    label="建议操作", interactive=False, visible=False
                )
                analysis_reason = gr.Textbox(
                    label="原因", interactive=False, visible=False
                )
            analysis_raw_output = gr.Textbox(
                label="原始输出 (如果非预期格式)", interactive=False, visible=False
            )

            def run_ai_analysis(current_image_path, is_image_generated):
                if not is_image_generated:
                    log.warning("需要先生成图片才能进行AI分析") # 添加日志
                    return (
                        gr.update(value="错误：需要先生成图片。", interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,  # analysis_result_state
                        False,  # analysis_done_state
                    )
                if not current_image_path:
                    log.error("无法找到用于AI分析的图片路径") # 添加日志
                    return (
                        gr.update(
                            value="错误：无法找到用于AI分析的图片路径。",
                            interactive=False,
                        ),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,
                        False,
                    )

                status_update = gr.update(
                    value="正在调用 Pipeline 进行AI分析...", interactive=False
                )
                action_update = gr.update(visible=False)
                reason_update = gr.update(visible=False)
                raw_update = gr.update(visible=False)

                pipeline_result = pipeline.run_step_3_analyze_signals(
                    current_image_path
                )

                if pipeline_result["success"]:
                    analysis_result = pipeline_result["result"]
                    analysis_done = True
                    status_update = gr.update(value="AI分析成功！", interactive=False)

                    # 假设 trade_advice 是 und_img.TradeAdvice 的实例或类似结构
                    # Gradio 的 Textbox 输入需要是字符串
                    if hasattr(analysis_result, "action") and hasattr(
                        analysis_result, "reason"
                    ):
                        action_update = gr.update(
                            value=str(analysis_result.action).upper(), visible=True
                        )
                        reason_update = gr.update(
                            value=str(analysis_result.reason), visible=True
                        )
                        raw_update = gr.update(visible=False)
                    else:
                        log.warning(f"AI 返回了非预期的结果: {analysis_result}") # 添加日志
                        raw_update = gr.update(
                            value=f"AI 返回了非预期的结果: {analysis_result}",
                            visible=True,
                        )
                        action_update = gr.update(visible=False)
                        reason_update = gr.update(visible=False)

                else:
                    analysis_result = None
                    analysis_done = False
                    log.error(f"AI分析失败：{pipeline_result['error']}") # 添加日志
                    status_update = gr.update(
                        value=f"AI分析失败：\n{pipeline_result['error']}",
                        interactive=False,
                    )
                    action_update = gr.update(visible=False)
                    reason_update = gr.update(visible=False)
                    raw_update = gr.update(visible=False)

                return (
                    status_update,
                    action_update,
                    reason_update,
                    raw_update,
                    analysis_result,  # 更新 analysis_result_state
                    analysis_done,  # 更新 analysis_done_state
                )

        # --- 步骤四：生成评估报告 ---
        with gr.Tab("第四步：生成报告"):
            with gr.Row():
                report_button = gr.Button(
                    "生成评估报告", variant="primary", interactive=False
                )  # 初始禁用
                report_status = gr.Textbox(
                    "请先完成第一步获取数据。", label="状态", interactive=False
                )
            with gr.Row(visible=False) as report_metrics_row:
                report_trades = gr.Number(label="总交易次数", interactive=False)
                report_profit = gr.Number(label="总收益", interactive=False)
                report_win_rate = gr.Number(label="胜率 (%)", interactive=False)
                report_avg_profit = gr.Number(label="平均单笔收益", interactive=False)
            report_details_df = gr.DataFrame(label="交易明细", visible=False)
            final_message = gr.Markdown("", visible=False)

            def run_generate_report(current_data_result, is_data_fetched):
                metrics_row_update = gr.update(visible=False)
                details_df_update = gr.update(visible=False)
                final_msg_update = gr.update(visible=False)
                trades, profit, win_rate, avg_profit = (
                    None,
                    None,
                    None,
                    None,
                )  # Default values
                evaluation_data = None  # Default value
                report_generated = False  # Default value

                if not is_data_fetched:
                    log.warning("需要先获取数据才能生成报告") # 添加日志
                    status_update = gr.update(
                        value="错误：需要先获取数据。", interactive=False
                    )
                    return (
                        status_update,
                        metrics_row_update,
                        trades,
                        profit,
                        win_rate,
                        avg_profit,  # metrics
                        details_df_update,
                        evaluation_data,  # report_content_state
                        report_generated,  # report_generated_state
                        final_msg_update,
                    )

                # 检查数据是否为 DataFrame
                if not isinstance(current_data_result, pd.DataFrame):
                    log.error("无法执行评估，因为第一步获取的数据不是 DataFrame") # 添加日志
                    status_update = gr.update(
                        value="错误：无法执行评估，因为第一步获取的数据不是 DataFrame。",
                        interactive=False,
                    )
                    return (
                        status_update,
                        metrics_row_update,
                        trades,
                        profit,
                        win_rate,
                        avg_profit,  # metrics
                        details_df_update,
                        evaluation_data,
                        report_generated,
                        final_msg_update,
                    )

                status_update = gr.update(
                    value="正在调用 Pipeline 生成评估报告...", interactive=False
                )

                pipeline_result = pipeline.run_step_4_generate_report(
                    current_data_result
                )

                if pipeline_result["success"]:
                    evaluation_data = pipeline_result["report"]
                    report_generated = True
                    status_update = gr.update(
                        value="评估报告生成成功！", interactive=False
                    )
                    metrics_row_update = gr.update(visible=True)
                    details_df_update = gr.update(
                        value=pd.DataFrame(
                            evaluation_data.get("trade_details", []),
                            columns=["Timestamp", "Action", "Price"],
                        ),
                        visible=True,
                    )
                    final_msg_update = gr.update(
                        value="**所有步骤已完成！** 🎉", visible=True
                    )

                    # 更新指标
                    trades = evaluation_data.get("total_trades", "N/A")
                    # Ensure metrics are numbers or handle N/A for gr.Number
                    try:
                        profit = float(f"{evaluation_data.get('total_profit', 0):.2f}")
                    except (ValueError, TypeError):
                        profit = None
                    try:
                        # Gradio Number doesn't support '%', convert to float
                        win_rate_str = f"{evaluation_data.get('win_rate', 0):.2%}"
                        win_rate = (
                            float(win_rate_str.replace("%", ""))
                            if "%" in win_rate_str
                            else float(win_rate_str)
                        )
                    except (ValueError, TypeError):
                        win_rate = None
                    try:
                        avg_profit = float(
                            f"{evaluation_data.get('avg_profit', 0):.2f}"
                        )
                    except (ValueError, TypeError):
                        avg_profit = None

                else:
                    evaluation_data = None
                    report_generated = False
                    log.error(f"评估报告生成失败：{pipeline_result['error']}") # 添加日志
                    status_update = gr.update(
                        value=f"评估报告生成失败：\n{pipeline_result['error']}",
                        interactive=False,
                    )
                    metrics_row_update = gr.update(visible=False)
                    details_df_update = gr.update(visible=False)
                    final_msg_update = gr.update(visible=False)

                return (
                    status_update,
                    metrics_row_update,
                    trades,
                    profit,
                    win_rate,
                    avg_profit,  # metrics
                    details_df_update,
                    evaluation_data,  # 更新 report_content_state
                    report_generated,  # 更新 report_generated_state
                    final_msg_update,
                )

        # --- 连接按钮和函数 ---
        fetch_button.click(
            fn=run_fetch_data,
            inputs=[],
            outputs=[
                fetch_status,
                fetch_output_df,
                fetch_output_path,
                data_result_state,
                data_fetched_state,
                generate_image_button,  # 更新按钮状态
                report_button,  # 更新报告按钮状态 (也依赖第一步)
            ],
        )

        generate_image_button.click(
            fn=run_generate_image,
            inputs=[data_result_state, data_fetched_state],
            outputs=[
                image_status,
                generated_image,
                image_path_state,
                image_generated_state,
                analyze_button,  # 更新按钮状态
            ],
        )

        analyze_button.click(
            fn=run_ai_analysis,
            inputs=[image_path_state, image_generated_state],
            outputs=[
                analyze_status,
                analysis_action,
                analysis_reason,
                analysis_raw_output,
                analysis_result_state,
                analysis_done_state,
            ],
        )

        report_button.click(
            fn=run_generate_report,
            inputs=[data_result_state, data_fetched_state],
            outputs=[
                report_status,
                report_metrics_row,
                report_trades,
                report_profit,
                report_win_rate,
                report_avg_profit,
                report_details_df,
                report_content_state,
                report_generated_state,
                final_message,
            ],
        )

    return app


# --- 主程序入口 ---
if __name__ == "__main__":
    gradio_app = create_gradio_app()
    gradio_app.launch(share=False)
