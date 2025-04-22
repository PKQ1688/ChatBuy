import os

<<<<<<< HEAD
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

=======
import gradio as gr
import pandas as pd
>>>>>>> 62ba0cab244184c3a94aa293bc4d70fd2b9fd246

# --- 导入 Pipeline ---
try:
    from chatbuy.core.pipeline import TradingAnalysisPipeline
except ImportError as e:
<<<<<<< HEAD
    st.error(f"无法导入 TradingAnalysisPipeline: {e}")
    st.stop()  # 如果核心 Pipeline 无法导入，则停止应用

# --- 初始化 Pipeline (确保只执行一次) ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = TradingAnalysisPipeline()

pipeline = st.session_state.pipeline

# --- Streamlit 应用 ---
st.title("交易策略分析流程 (Pipeline 版)")

# --- 状态管理 ---
# 使用 session_state 在页面刷新和步骤间传递数据
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False
if "image_generated" not in st.session_state:
    st.session_state.image_generated = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# 存储中间结果
if "data_result" not in st.session_state:
    st.session_state.data_result = None  # 可以是 DataFrame 或文件路径
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None  # 可以是 DataFrame 或文件路径
if "report_content" not in st.session_state:
    st.session_state.report_content = None  # 可以是报告文本或文件路径


# --- 步骤一：获取K线数据 ---
st.header("第一步：获取K线数据")
col1, col2 = st.columns([1, 3])
with col1:
    # 按钮现在总是可用，除非 pipeline 初始化失败 (已在顶部处理)
    fetch_button = st.button("获取数据", key="fetch")
with col2:
    fetch_output_area = st.empty()
    fetch_output_area.info("点击按钮开始获取数据...")


if fetch_button:
    fetch_output_area.info("正在调用 Pipeline 获取数据...")
    # !! 如果 fetch_data_function 需要参数，需要在这里传递 !!
    # 例如: pipeline_result = pipeline.run_step_1_fetch_data(symbol='BTCUSDT', interval='1d')
    pipeline_result = pipeline.run_step_1_fetch_data()

    if pipeline_result["success"]:
        st.session_state.data_result = pipeline_result["result"]
        st.session_state.data_fetched = True
        fetch_output_area.success("数据获取成功！")

        # 显示结果预览
        result = pipeline_result["result"]
        if isinstance(result, pd.DataFrame):
            st.dataframe(result.head())
        elif isinstance(result, str) and os.path.exists(result):
            st.success(f"数据已保存到: {result}")
            try:
                st.dataframe(pd.read_csv(result).head())
            except Exception as e:
                st.warning(f"尝试读取数据显示预览失败: {e}")
        else:
            st.info(f"函数返回: {result}")
    else:
        st.session_state.data_fetched = False
        fetch_output_area.error(f"数据获取失败：\n{pipeline_result['error']}")


# --- 步骤二：生成K线图片 ---
st.header("第二步：生成K线图片")
col3, col4 = st.columns([1, 3])
with col3:
    generate_image_button = st.button(
        "生成图片", key="generate", disabled=(not st.session_state.data_fetched)
    )
with col4:
    image_output_area = st.empty()
    if not st.session_state.data_fetched:
        image_output_area.info("请先完成第一步获取数据。")
    else:
        image_output_area.info("点击按钮生成K线图...")

if generate_image_button and st.session_state.data_fetched:
    image_output_area.info("正在调用 Pipeline 生成图片...")
    pipeline_result = pipeline.run_step_2_generate_image(st.session_state.data_result)

    if pipeline_result["success"]:
        st.session_state.image_path = pipeline_result["image_path"]
        st.session_state.image_generated = True
        image_output_area.success(f"图片生成成功！")
        st.image(st.session_state.image_path, caption="生成的K线图")
    else:
        st.session_state.image_generated = False
        image_output_area.error(f"图片生成失败：\n{pipeline_result['error']}")


# --- 步骤三：AI分析买卖点 ---
st.header("第三步：AI分析买卖点")
col5, col6 = st.columns([1, 3])
with col5:
    analyze_button = st.button(
        "AI分析", key="analyze", disabled=(not st.session_state.image_generated)
    )
with col6:
    analyze_output_area = st.empty()
    if not st.session_state.image_generated:
        analyze_output_area.info("请先完成第二步生成图片。")
    else:
        analyze_output_area.info("点击按钮进行AI分析...")

if analyze_button and st.session_state.image_generated:
    analyze_output_area.info("正在调用 Pipeline 进行AI分析...")
    # 假设 AI 总是使用图片路径
    input_for_ai = st.session_state.image_path
    if input_for_ai:
        pipeline_result = pipeline.run_step_3_analyze_signals(input_for_ai)

        if pipeline_result["success"]:
            st.session_state.analysis_result = pipeline_result["result"]
            st.session_state.analysis_done = True
            analyze_output_area.success("AI分析成功！")

            # 修正：显示 TradeAdvice 对象的内容
            trade_advice = pipeline_result["result"]
            # 假设 trade_advice 是 und_img.TradeAdvice 的实例或类似结构
            if hasattr(trade_advice, "action") and hasattr(trade_advice, "reason"):
                st.metric("建议操作", trade_advice.action.upper())
                st.info(f"原因: {trade_advice.reason}")
            else:
                # 如果返回的不是预期的对象，显示原始结果
                st.info(f"AI 返回了非预期的结果: {trade_advice}")
        else:
            st.session_state.analysis_done = False
            analyze_output_area.error(f"AI分析失败：\n{pipeline_result['error']}")
    else:
        analyze_output_area.error("错误：无法找到用于AI分析的图片路径。")


# --- 步骤四：生成评估报告 ---
st.header("第四步：生成评估报告")
col7, col8 = st.columns([1, 3])
with col7:
    # 修正：按钮依赖于第一步数据获取完成
    report_button = st.button(
        "生成评估报告", key="report", disabled=(not st.session_state.data_fetched)
    )
with col8:
    report_output_area = st.empty()
    # 修正：提示依赖于第一步
    if not st.session_state.data_fetched:
        report_output_area.info("请先完成第一步获取数据。")
    else:
        report_output_area.info("点击按钮使用已有信号和价格数据生成评估报告...")

# 修正：按钮点击条件依赖于第一步
if report_button and st.session_state.data_fetched:
    report_output_area.info("正在调用 Pipeline 生成评估报告...")
    # 修正：传递第一步获取的价格 DataFrame
    price_data_for_eval = st.session_state.data_result
    if isinstance(price_data_for_eval, pd.DataFrame):
        pipeline_result = pipeline.run_step_4_generate_report(price_data_for_eval)

        if pipeline_result["success"]:
            # 修正：返回的是包含评估结果的字典
            evaluation_data = pipeline_result["report"]
            st.session_state.report_content = evaluation_data  # 存储评估结果字典
            st.session_state.report_generated = True
            report_output_area.success("评估报告生成成功！")

            # 修正：显示评估指标
            st.subheader("交易评估结果")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("总交易次数", evaluation_data.get("total_trades", "N/A"))
            col_m2.metric("总收益", f"{evaluation_data.get('total_profit', 0):.2f}")
            col_m3.metric("胜率", f"{evaluation_data.get('win_rate', 0):.2%}")
            col_m4.metric("平均单笔收益", f"{evaluation_data.get('avg_profit', 0):.2f}")

            st.subheader("交易明细")
            trade_details_df = pd.DataFrame(
                evaluation_data.get("trade_details", []),
                columns=["Timestamp", "Action", "Price"],
            )
            st.dataframe(trade_details_df)

        else:
            st.session_state.report_generated = False
            report_output_area.error(f"评估报告生成失败：\n{pipeline_result['error']}")
    else:
        # 如果第一步的结果不是 DataFrame (例如是文件路径)，则无法执行评估
        st.session_state.report_generated = False
        report_output_area.error(
            "错误：无法执行评估，因为第一步获取的数据不是 DataFrame。"
        )


# --- 结束语 ---
if st.session_state.report_generated:
    st.balloons()
    st.success("所有步骤已完成！")
=======
    print(f"错误：无法导入 TradingAnalysisPipeline: {e}")
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
                    return (
                        gr.update(value="错误：需要先生成图片。", interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None,  # analysis_result_state
                        False,  # analysis_done_state
                    )
                if not current_image_path:
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
                        raw_update = gr.update(
                            value=f"AI 返回了非预期的结果: {analysis_result}",
                            visible=True,
                        )
                        action_update = gr.update(visible=False)
                        reason_update = gr.update(visible=False)

                else:
                    analysis_result = None
                    analysis_done = False
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
>>>>>>> 62ba0cab244184c3a94aa293bc4d70fd2b9fd246
