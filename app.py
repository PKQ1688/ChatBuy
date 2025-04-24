import os

import gradio as gr
import pandas as pd

from chatbuy.logger import log

# --- Import Pipeline ---
try:
    from chatbuy.core.pipeline import TradingAnalysisPipeline
except ImportError as e:
    log.error("Error: Failed to import TradingAnalysisPipeline", exc_info=True) # Use log.error
    # In Gradio, we can't stop the app directly like in Streamlit,
    # but we can display an error message in the interface.
    pipeline_import_error = f"Failed to import core processing module: {e}"
    pipeline = None  # Set to None for later checks
else:
    pipeline_import_error = None
    # --- Initialize Pipeline ---
    # Gradio apps are typically initialized once on startup
    pipeline = TradingAnalysisPipeline()

# --- Gradio App ---


def create_gradio_app():
    """Create the Gradio application interface and logic."""
    if pipeline is None:
        with gr.Blocks() as app:
            gr.Markdown("# Trading Strategy Analysis Pipeline (Pipeline Version)")
            gr.Error(f"Application initialization failed: {pipeline_import_error}")
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
                fetch_button = gr.Button("Fetch Data", variant="primary")
                fetch_status = gr.Textbox(
                    "Click the button to start fetching data...", label="Status", interactive=False
                )
            fetch_output_df = gr.DataFrame(label="Data Preview (first 5 rows)", visible=False)
            fetch_output_path = gr.Textbox(
                label="Data File Path", visible=False, interactive=False
            )

            def run_fetch_data():
                status_update = gr.update(
                    value="Calling Pipeline to fetch data...", interactive=False
                )
                df_update = gr.update(visible=False)
                path_update = gr.update(visible=False)
                next_button_update = gr.update(interactive=False)  # Disable the next button
                report_button_update = gr.update(interactive=False)  # Disable the report button

                pipeline_result = pipeline.run_step_1_fetch_data()

                if pipeline_result["success"]:
                    result = pipeline_result["result"]
                    data_fetched = True
                    status_update = gr.update(value="Data fetched successfully!", interactive=False)
                    next_button_update = gr.update(interactive=True)  # Enable the generate image button
                    report_button_update = gr.update(interactive=True)  # Enable the report button

                    if isinstance(result, pd.DataFrame):
                        df_update = gr.update(value=result.head(), visible=True)
                        path_update = gr.update(visible=False)
                        data_result = result  # Store DataFrame directly
                    elif isinstance(result, str) and os.path.exists(result):
                        path_update = gr.update(
                            value=f"Data saved to: {result}", visible=True
                        )
                        try:
                            df_update = gr.update(
                                value=pd.read_csv(result).head(), visible=True
                            )
                        except Exception as e:
                            log.warning("Data fetched successfully, but preview failed", exc_info=True) # Add log
                            status_update = gr.update(
                                value=f"Data fetched successfully, but preview failed: {e}",
                                interactive=False,
                            )
                        data_result = result  # Store file path
                    else:
                        status_update = gr.update(
                            value=f"Data fetched successfully, function returned: {result}", interactive=False
                        )
                        data_result = result  # Store other result types
                else:
                    data_fetched = False
                    log.error(f"Data fetch failed: {pipeline_result['error']}") # Add log
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
                    "Generate Image", variant="primary", interactive=False
                )  # Initially disabled
                image_status = gr.Textbox(
                    "Please complete Step 1 (Fetch Data) first.", label="Status", interactive=False
                )
            generated_image = gr.Image(
                label="Generated Candlestick Chart", type="filepath", visible=False
            )

            def run_generate_image(current_data_result, is_data_fetched):
                if not is_data_fetched:
                    log.warning("Need to fetch data before generating image") # Add log
                    return (
                        gr.update(value="Error: Need to fetch data first.", interactive=False),
                        gr.update(visible=False),
                        None,  # image_path_state
                        False,  # image_generated_state
                        gr.update(interactive=False),  # analyze_button
                    )

                status_update = gr.update(
                    value="Calling Pipeline to generate image...", interactive=False
                )
                image_update = gr.update(visible=False)
                next_button_update = gr.update(interactive=False)  # Disable the next button

                pipeline_result = pipeline.run_step_2_generate_image(
                    current_data_result
                )

                if pipeline_result["success"]:
                    image_path = pipeline_result["image_path"]
                    image_generated = True
                    status_update = gr.update(value="Image generated successfully!", interactive=False)
                    image_update = gr.update(value=image_path, visible=True)
                    next_button_update = gr.update(interactive=True)  # Enable the next button
                else:
                    image_path = None
                    image_generated = False
                    log.error(f"Image generation failed: {pipeline_result['error']}") # Add log
                    status_update = gr.update(
                        value=f"Image generation failed:\n{pipeline_result['error']}",
                        interactive=False,
                    )

                return (
                    status_update,
                    image_update,
                    image_path,  # Update image_path_state
                    image_generated,  # Update image_generated_state
                    next_button_update,  # Update AI analysis button state
                )

        # --- Step 3: AI Analysis for Buy/Sell Points ---
        with gr.Tab("Step 3: AI Analysis"):
            with gr.Row():
                analyze_button = gr.Button(
                    "AI Analysis", variant="primary", interactive=False
                )  # Initially disabled
                analyze_status = gr.Textbox(
                    "Please complete Step 2 (Generate Image) first.", label="Status", interactive=False
                )
            with gr.Row():
                analysis_action = gr.Textbox(
                    label="Suggested Action", interactive=False, visible=False
                )
                analysis_reason = gr.Textbox(
                    label="Reason", interactive=False, visible=False
                )
            analysis_raw_output = gr.Textbox(
                label="Raw Output (if unexpected format)", interactive=False, visible=False
            )

            def run_ai_analysis(current_image_path, is_image_generated):
                if not is_image_generated:
                    log.warning("Need to generate image before AI analysis") # Add log
                    return (
                        gr.update(value="Error: Need to generate image first.", interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,  # analysis_result_state
                        False,  # analysis_done_state
                    )
                if not current_image_path:
                    log.error("Could not find image path for AI analysis") # Add log
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
                    status_update = gr.update(value="AI analysis successful!", interactive=False)

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
                        log.warning(f"AI returned an unexpected result: {analysis_result}") # Add log
                        raw_update = gr.update(
                            value=f"AI returned an unexpected result: {analysis_result}",
                            visible=True,
                        )
                        action_update = gr.update(visible=False)
                        reason_update = gr.update(visible=False)

                else:
                    analysis_result = None
                    analysis_done = False
                    log.error(f"AI analysis failed: {pipeline_result['error']}") # Add log
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
                    "Please complete Step 1 (Fetch Data) first.", label="Status", interactive=False
                )
            with gr.Row(visible=False) as report_metrics_row:
                report_trades = gr.Number(label="Total Trades", interactive=False)
                report_profit = gr.Number(label="Total Profit", interactive=False)
                report_win_rate = gr.Number(label="Win Rate (%)", interactive=False)
                report_avg_profit = gr.Number(label="Average Profit per Trade", interactive=False)
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
                    log.warning("Need to fetch data before generating report") # Add log
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
                    log.error("Cannot perform evaluation because the data fetched in Step 1 is not a DataFrame") # Add log
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
                    value="Calling Pipeline to generate evaluation report...", interactive=False
                )

                pipeline_result = pipeline.run_step_4_generate_report(
                    current_data_result
                )

                if pipeline_result["success"]:
                    evaluation_data = pipeline_result["report"]
                    report_generated = True
                    status_update = gr.update(
                        value="Evaluation report generated successfully!", interactive=False
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
                    log.error(f"Evaluation report generation failed: {pipeline_result['error']}") # Add log
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
            inputs=[],
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
            inputs=[data_result_state, data_fetched_state],
            outputs=[
                image_status,
                generated_image,
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
