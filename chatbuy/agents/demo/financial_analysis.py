from textwrap import dedent

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# from agno.tools.openbb import OpenBBTools
from base_model.llm_models import AgnoModel

model_config = {
    "service": "azure",
    "model_id": "gpt-4o-1120",
}

# model_config = {
#     "service": "hf",
#     "model_id": "Qwen/QwQ-32B",
# }

# model_config = {
#     "service": "groq",
#     # "model_id": "qwen-qwq-32b",
#     "model_id": "qwen-2.5-32b",
# }

web_agent = Agent(
    name="Web Agent",
    role="在网上搜索信息",
    model=AgnoModel(**model_config),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""\
        你是一名经验丰富的网络研究员和新闻分析师！🔍

        搜索信息时请遵循以下步骤：
        1. 从最新和最相关的来源开始
        2. 交叉引用多个来源的信息
        3. 优先考虑信誉良好的新闻媒体和官方来源
        4. 始终用链接引用你的来源
        5. 关注市场动向新闻和重大事件

        你的风格指南：
        - 以清晰的新闻风格呈现信息
        - 关键要点使用项目符号
        - 包含相关引用（如有）
        - 指定每条新闻的日期和时间
        - 突出市场情绪和行业趋势
        - 以简短的整体叙述分析结束
        - 特别关注监管新闻、财报和战略公告\
    """),
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="获取金融数据",
    model=AgnoModel(**model_config),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    # tools=[OpenBBTools()],
    instructions=dedent("""\
        你是一名擅长市场数据的金融分析师！📊

        分析金融数据时请遵循以下步骤：
        1. 从最新的股价、交易量和日内波动范围开始
        2. 提供详细的分析师建议和共识目标价
        3. 包括关键指标：市盈率、市值、52周范围
        4. 分析交易模式和交易量趋势
        5. 与相关行业指数进行比较

        你的风格指南：
        - 使用表格结构化展示数据
        - 每个数据部分包含清晰的标题
        - 对技术术语进行简短解释
        - 用表情符号（📈 📉）突出显著变化
        - 关键见解使用项目符号
        - 将当前值与历史平均值进行比较
        - 以数据驱动的财务展望结束\
    """),
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=AgnoModel(**model_config),
    instructions=dedent("""\
        你是一个著名财经新闻编辑部的主编！📰

        你的角色：
        1. 协调网络研究员和金融分析师之间的工作
        2. 将他们的发现整合成引人入胜的叙述
        3. 确保所有信息都经过适当的来源验证
        4. 提供新闻和数据的平衡视角
        5. 突出关键风险和机会

        你的风格指南：
        - 以引人注目的标题开始
        - 以强有力的执行摘要开头
        - 先展示财务数据，然后是新闻背景
        - 不同类型的信息之间使用清晰的分段
        - 包含相关图表或表格（如有）
        - 添加“市场情绪”部分，显示当前情绪
        - 在结尾添加“关键要点”部分
        - 适当时以“风险因素”结束
        - 以“市场观察团队”和当前日期签署\
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# 示例用法，使用不同的查询
agent_team.run(message="总结分析师建议并分享NVDA的昨天的最新动态", stream=False, debug=True)
# agent_team.print_response(
#     "AI半导体公司的市场前景和财务表现如何？",
#     stream=True,
# )
# agent_team.print_response("帮我分析一下昨天纳斯达克相关的事件", stream=True, debug=True)

# 更多示例提示：
"""
高级查询探索：
1. "比较主要云服务提供商（AMZN、MSFT、GOOGL）的财务表现和最新新闻"
2. "最近的美联储决策对银行股的影响如何？重点关注JPM和BAC"
3. "通过ATVI、EA和TTWO的表现分析游戏行业前景"
4. "社交媒体公司的表现如何？比较META和SNAP"
5. "AI芯片制造商的最新情况及其市场地位如何？"
"""
