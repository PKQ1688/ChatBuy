import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import os

# --- 导入 Pipeline ---
try:
    from chatbuy.core.pipeline import TradingAnalysisPipeline
except ImportError as e:
    st.error(f"无法导入 TradingAnalysisPipeline: {e}")
    st.stop() # 如果核心 Pipeline 无法导入，则停止应用

# --- 初始化 Pipeline (确保只执行一次) ---
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = TradingAnalysisPipeline()

pipeline = st.session_state.pipeline

# --- Streamlit 应用 ---
st.title("交易策略分析流程 (Pipeline 版)")

# --- 状态管理 ---
# 使用 session_state 在页面刷新和步骤间传递数据
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'image_generated' not in st.session_state:
    st.session_state.image_generated = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# 存储中间结果
if 'data_result' not in st.session_state:
    st.session_state.data_result = None # 可以是 DataFrame 或文件路径
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None # 可以是 DataFrame 或文件路径
if 'report_content' not in st.session_state:
    st.session_state.report_content = None # 可以是报告文本或文件路径


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
    generate_image_button = st.button("生成图片", key="generate",
                                      disabled=(not st.session_state.data_fetched))
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
    analyze_button = st.button("AI分析", key="analyze",
                               disabled=(not st.session_state.image_generated))
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
            if hasattr(trade_advice, 'action') and hasattr(trade_advice, 'reason'):
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
    report_button = st.button("生成评估报告", key="report",
                              disabled=(not st.session_state.data_fetched))
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
            st.session_state.report_content = evaluation_data # 存储评估结果字典
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
                columns=["Timestamp", "Action", "Price"]
            )
            st.dataframe(trade_details_df)

        else:
            st.session_state.report_generated = False
            report_output_area.error(f"评估报告生成失败：\n{pipeline_result['error']}")
    else:
         # 如果第一步的结果不是 DataFrame (例如是文件路径)，则无法执行评估
         st.session_state.report_generated = False
         report_output_area.error("错误：无法执行评估，因为第一步获取的数据不是 DataFrame。")


# --- 结束语 ---
if st.session_state.report_generated:
    st.balloons()
    st.success("所有步骤已完成！")