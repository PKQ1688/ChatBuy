import datetime  # Make sure datetime import is at the top level
import os

import gradio as gr
import pandas as pd

from chatbuy.core.pipeline import TradingAnalysisPipeline
from chatbuy.logger import log

pipeline = TradingAnalysisPipeline()

# --- Gradio App ---


def create_gradio_app():
    """Create the Gradio application interface and logic."""
    if pipeline is None:
        with gr.Blocks() as app:
            gr.Markdown("# Trading Strategy Analysis Pipeline (Pipeline Version)")
            gr.Error("Application initialization failed: pipeline_import_error")
        return app

    with gr.Blocks(title="Trading Strategy Analysis Pipeline") as app:
        gr.Markdown("# Trading Strategy Analysis Pipeline (Pipeline Version)")

        # --- State Management (using gr.State) ---
        data_result_state = gr.State(None)
        image_path_state = gr.State(None)
        analysis_result_state = gr.State(None)
        report_content_state = gr.State(None)
        # State flags (might be useful although Gradio's flow control differs)
        data_fetched_state = gr.State(False)
        image_generated_state = gr.State(False)
        analysis_done_state = gr.State(False)
        report_generated_state = gr.State(False)

        # --- Step 1: Fetch Candlestick Data ---
        with gr.Tab("Step 1: Fetch Data"):
            with gr.Row():
                symbol_input = gr.Textbox(
                    value="BTC/USDT", label="Symbol (标的)", interactive=True
                )
                timeframe_input = gr.Dropdown(
                    choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                    value="1d",
                    label="Timeframe (K线周期)",
                    interactive=True,
                )
                # 使用普通的 Textbox 组件代替 DateTime 组件
                today = datetime.date.today()
                default_start_date = (today - datetime.timedelta(days=365)).strftime(
                    "%Y-%m-%d"
                )
                default_end_date = today.strftime("%Y-%m-%d")

                start_date_input = gr.Textbox(
                    label="起始时间 (YYYY-MM-DD)",
                    value=default_start_date,
                    placeholder="例如: 2023-01-01",
                )
                end_date_input = gr.Textbox(
                    label="结束时间 (YYYY-MM-DD)",
                    value=default_end_date,
                    placeholder="例如: 2024-01-01",
                )
            with gr.Row():
                fetch_button = gr.Button("Fetch Data", variant="primary")
                fetch_status = gr.Textbox(
                    "Click the button to start fetching data...",
                    label="Status",
                    interactive=False,
                )
            fetch_output_df = gr.DataFrame(
                label="Data Preview (first 5 rows)", visible=False
            )
            fetch_output_path = gr.Textbox(
                label="Data File Path", visible=False, interactive=False
            )

            def run_fetch_data(symbol, timeframe, start_date, end_date):
                status_update = gr.update(
                    value="Calling Pipeline to fetch data...", interactive=False
                )
                df_update = gr.update(visible=False)
                path_update = gr.update(visible=False)
                next_button_update = gr.update(
                    interactive=False
                )  # Disable the next button
                report_button_update = gr.update(
                    interactive=False
                )  # Disable the report button

                # Process the text date inputs
                try:
                    # Convert string dates to datetime objects
                    if start_date:
                        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                    else:
                        raise ValueError("Start date is required")

                    if end_date:
                        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
                    else:
                        end_dt = None
                except Exception as e:
                    log.error(f"Error processing date inputs: {e}")
                    return (
                        gr.update(
                            value=f"错误：日期格式错误。请使用YYYY-MM-DD格式。详情: {e}",
                            interactive=False,
                        ),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,
                        False,
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )

                kwargs = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_dt,
                }
                if end_dt:
                    kwargs["end_date"] = end_dt

                pipeline_result = pipeline.run_step_1_fetch_data(**kwargs)

                if pipeline_result["success"]:
                    result = pipeline_result["result"]
                    data_fetched = True
                    status_update = gr.update(
                        value="Data fetched successfully!", interactive=False
                    )
                    next_button_update = gr.update(
                        interactive=True
                    )  # Enable the generate image button
                    report_button_update = gr.update(
                        interactive=True
                    )  # Enable the report button

                    if isinstance(result, pd.DataFrame):
                        df_update = gr.update(value=result.tail(), visible=True)
                        path_update = gr.update(visible=False)
                        data_result = result  # Store DataFrame directly
                    elif isinstance(result, str) and os.path.exists(result):
                        path_update = gr.update(
                            value=f"Data saved to: {result}", visible=True
                        )
                        try:
                            df_update = gr.update(
                                value=pd.read_csv(result).tail(), visible=True
                            )
                        except Exception as e:
                            log.warning(
                                "Data fetched successfully, but preview failed",
                                exc_info=True,
                            )  # Add log
                            status_update = gr.update(
                                value=f"Data fetched successfully, but preview failed: {e}",
                                interactive=False,
                            )
                        data_result = result  # Store file path
                    else:
                        status_update = gr.update(
                            value=f"Data fetched successfully, function returned: {result}",
                            interactive=False,
                        )
                        data_result = result  # Store other result types
                else:
                    data_fetched = False
                    log.error(
                        f"Data fetch failed: {pipeline_result['error']}"
                    )  # Add log
                    status_update = gr.update(
                        value=f"Data fetch failed:\n{pipeline_result['error']}",
                        interactive=False,
                    )
                    data_result = None

                # Return all components and states that need updating
                return (
                    status_update,
                    df_update,
                    path_update,
                    data_result,  # Update data_result_state
                    data_fetched,  # Update data_fetched_state
                    next_button_update,  # Update generate image button state
                    report_button_update,  # Update report button state
                )

        # --- Step 2: Generate Candlestick Image ---
        with gr.Tab("Step 2: Generate Image"):
            with gr.Row():
                generate_image_button = gr.Button(
                    "Generate Images", variant="primary", interactive=False
                )  # Initially disabled
                image_status = gr.Textbox(
                    "Please complete Step 1 (Fetch Data) first.",
                    label="Status",
                    interactive=False,
                )

            with gr.Row():
                length_input = gr.Slider(
                    minimum=30,
                    maximum=200,
                    step=10,
                    value=120,
                    label="Candlesticks per image (length)",
                    interactive=True,
                )
                step_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=1,
                    label="Sliding window step size",
                    interactive=True,
                )

            with gr.Row():
                filename_prefix = gr.Textbox(
                    value="chart", label="Filename prefix", interactive=True
                )
                output_dir = gr.Textbox(
                    value="output/batch_images",
                    label="Output directory",
                    interactive=True,
                )

            generation_result = gr.Markdown("", visible=False)
            sample_image = gr.Image(
                label="Sample Generated Image", type="filepath", visible=False
            )

            def run_generate_image(
                current_data_result,
                is_data_fetched,
                length,
                step,
                output_folder,
                prefix,
            ):
                if not is_data_fetched:
                    log.warning("Need to fetch data before generating image")
                    return (
                        gr.update(value="错误: 请先获取数据", interactive=False),
                        gr.update(visible=False),
                        None,  # image_path_state
                        False,  # image_generated_state
                        gr.update(interactive=False),  # analyze_button
                    )

                status_update = gr.update(
                    value="正在批量生成图表...", interactive=False
                )

                # Make sure output directory exists
                os.makedirs(output_folder, exist_ok=True)

                # Call batch generation function
                pipeline_result = pipeline.run_step_2_generate_images_batch(
                    data_input=current_data_result,
                    output_dir=output_folder,
                    length=length,
                    step=step,
                    filename_prefix=prefix,
                )

                if pipeline_result["success"]:
                    image_generated = True

                    # Find first image to display as sample
                    sample_img_path = None
                    try:
                        files = [
                            f for f in os.listdir(output_folder) if f.endswith(".png")
                        ]
                        if files:
                            sample_img_path = os.path.join(output_folder, files[0])
                    except Exception as e:
                        log.warning(f"Failed to find sample image: {e}")

                    status_update = gr.update(
                        value=f"成功生成 {pipeline_result['count']} 张图表! 保存至: {output_folder}",
                        interactive=False,
                    )

                    image_update = gr.update(
                        value=sample_img_path if sample_img_path else None,
                        visible=True if sample_img_path else False,
                    )

                    result_markdown = f"""
### 批量图表生成结果
- **总计图表数**: {pipeline_result["count"]} 张
- **保存路径**: {output_folder}
- **每张图表K线数**: {length}
- **滑动窗口步长**: {step}
                    """

                    next_button_update = gr.update(
                        interactive=True
                    )  # Enable the next button

                    return (
                        status_update,
                        image_update,
                        result_markdown,  # Update result markdown
                        True,  # Enable visibility
                        sample_img_path,  # Update image_path_state (first image as sample)
                        image_generated,  # Update image_generated_state
                        next_button_update,  # Update AI analysis button state
                    )
                else:
                    image_generated = False
                    log.error(f"Image generation failed: {pipeline_result['error']}")
                    status_update = gr.update(
                        value=f"图表生成失败:\n{pipeline_result['error']}",
                        interactive=False,
                    )

                    return (
                        status_update,
                        gr.update(visible=False),
                        "",  # Empty markdown
                        False,  # Hide markdown
                        None,  # Update image_path_state
                        False,  # Update image_generated_state
                        gr.update(interactive=False),  # Update AI analysis button state
                    )

        # --- Step 3: AI Analysis for Buy/Sell Points ---
        with gr.Tab("Step 3: AI Analysis"):
            with gr.Row():
                analyze_button = gr.Button(
                    "AI Analysis", variant="primary", interactive=False
                )  # Initially disabled
                analyze_status = gr.Textbox(
                    "Please complete Step 2 (Generate Image) first.",
                    label="Status",
                    interactive=False,
                )
            with gr.Row():
                analysis_action = gr.Textbox(
                    label="Suggested Action", interactive=False, visible=False
                )
                analysis_reason = gr.Textbox(
                    label="Reason", interactive=False, visible=False
                )
            analysis_raw_output = gr.Textbox(
                label="Raw Output (if unexpected format)",
                interactive=False,
                visible=False,
            )

            def run_ai_analysis(current_image_path, is_image_generated):
                if not is_image_generated:
                    log.warning("Need to generate image before AI analysis")  # Add log
                    return (
                        gr.update(
                            value="Error: Need to generate image first.",
                            interactive=False,
                        ),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,  # analysis_result_state
                        False,  # analysis_done_state
                    )
                if not current_image_path:
                    log.error("Could not find image path for AI analysis")  # Add log
                    return (
                        gr.update(
                            value="Error: Could not find image path for AI analysis.",
                            interactive=False,
                        ),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,
                        False,
                    )

                status_update = gr.update(
                    value="Calling Pipeline for AI analysis...", interactive=False
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
                    status_update = gr.update(
                        value="AI analysis successful!", interactive=False
                    )

                    # Assume trade_advice is an instance of und_img.TradeAdvice or similar structure
                    # Gradio Textbox input needs to be a string
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
                        log.warning(
                            f"AI returned an unexpected result: {analysis_result}"
                        )  # Add log
                        raw_update = gr.update(
                            value=f"AI returned an unexpected result: {analysis_result}",
                            visible=True,
                        )
                        action_update = gr.update(visible=False)
                        reason_update = gr.update(visible=False)

                else:
                    analysis_result = None
                    analysis_done = False
                    log.error(
                        f"AI analysis failed: {pipeline_result['error']}"
                    )  # Add log
                    status_update = gr.update(
                        value=f"AI analysis failed:\n{pipeline_result['error']}",
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
                    analysis_result,  # Update analysis_result_state
                    analysis_done,  # Update analysis_done_state
                )

        # --- Step 4: Generate Evaluation Report ---
        with gr.Tab("Step 4: Generate Report"):
            with gr.Row():
                report_button = gr.Button(
                    "Generate Evaluation Report", variant="primary", interactive=False
                )  # Initially disabled
                report_status = gr.Textbox(
                    "Please complete Step 1 (Fetch Data) first.",
                    label="Status",
                    interactive=False,
                )
            with gr.Row(visible=False) as report_metrics_row:
                report_trades = gr.Number(label="Total Trades", interactive=False)
                report_profit = gr.Number(label="Total Profit", interactive=False)
                report_win_rate = gr.Number(label="Win Rate (%)", interactive=False)
                report_avg_profit = gr.Number(
                    label="Average Profit per Trade", interactive=False
                )
            report_details_df = gr.DataFrame(label="Trade Details", visible=False)
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
                    log.warning(
                        "Need to fetch data before generating report"
                    )  # Add log
                    status_update = gr.update(
                        value="Error: Need to fetch data first.", interactive=False
                    )
                    return (
                        status_update,
                        metrics_row_update,
                        trades,
                        profit,
                        win_rate,
                        avg_profit,  # metrics
                        details_df_update,
                        evaluation_data,  # Update report_content_state
                        report_generated,  # Update report_generated_state
                        final_msg_update,
                    )

                # Check if data is a DataFrame
                if not isinstance(current_data_result, pd.DataFrame):
                    log.error(
                        "Cannot perform evaluation because the data fetched in Step 1 is not a DataFrame"
                    )  # Add log
                    status_update = gr.update(
                        value="Error: Cannot perform evaluation because the data fetched in Step 1 is not a DataFrame.",
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
                    value="Calling Pipeline to generate evaluation report...",
                    interactive=False,
                )

                pipeline_result = pipeline.run_step_4_generate_report(
                    current_data_result
                )

                if pipeline_result["success"]:
                    evaluation_data = pipeline_result["report"]
                    report_generated = True
                    status_update = gr.update(
                        value="Evaluation report generated successfully!",
                        interactive=False,
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
                        value="**All steps completed!** 🎉", visible=True
                    )

                    # Update metrics
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
                    log.error(
                        f"Evaluation report generation failed: {pipeline_result['error']}"
                    )  # Add log
                    status_update = gr.update(
                        value=f"Evaluation report generation failed:\n{pipeline_result['error']}",
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
                    evaluation_data,  # Update report_content_state
                    report_generated,  # Update report_generated_state
                    final_msg_update,
                )

        # --- Connect Buttons and Functions ---
        fetch_button.click(
            fn=run_fetch_data,
            inputs=[
                symbol_input,
                timeframe_input,
                start_date_input,
                end_date_input,
            ],
            outputs=[
                fetch_status,
                fetch_output_df,
                fetch_output_path,
                data_result_state,
                data_fetched_state,
                generate_image_button,  # Update button state
                report_button,  # Update report button state (also depends on Step 1)
            ],
        )

        generate_image_button.click(
            fn=run_generate_image,
            inputs=[
                data_result_state,
                data_fetched_state,
                length_input,
                step_input,
                output_dir,
                filename_prefix,
            ],
            outputs=[
                image_status,
                sample_image,
                generation_result,
                generation_result,  # 用 markdown 组件对象替换 gr.update(visible=True)
                image_path_state,
                image_generated_state,
                analyze_button,  # Update button state
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


# --- Main Program Entry Point ---
if __name__ == "__main__":
    gradio_app = create_gradio_app()
    gradio_app.launch(share=False)
