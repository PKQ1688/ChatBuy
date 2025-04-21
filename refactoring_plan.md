# 代码重构优化计划

## 分析总结

当前 `app.py` 存在以下问题：

1.  **Streamlit 运行时错误:** `st.set_page_config()` 调用位置不当。
2.  **硬编码与灵活性:** 导入的函数名和路径硬编码在 `app.py` 中。
3.  **单一文件职责过重:** `app.py` 承担过多职责，难以维护。
4.  **函数接口不清晰:** 底层函数签名不确定。
5.  **代码重复:** 各步骤 UI 和逻辑相似。
6.  **错误处理:** 可以更友好。

## 优化计划步骤

1.  **修复 Streamlit 错误:** 将 `st.set_page_config(layout="wide")` 移动到 `app.py` 文件顶部。
2.  **抽象核心逻辑:**
    *   在 `chatbuy/core/` 下创建 `pipeline.py`。
    *   定义 `TradingAnalysisPipeline` 类或函数封装流程。
    *   `pipeline.py` 负责与具体实现交互，提供稳定接口。
3.  **简化 `app.py`:**
    *   `app.py` 负责 UI 布局和调用 `pipeline` 接口。
    *   提取重复 UI 代码为可重用函数。
4.  **规范函数接口:** 明确 `chatbuy/core/` 各模块函数的输入输出。
5.  **改进状态管理:** 简化 `st.session_state` 初始化。
6.  **增强错误处理:** 在 `pipeline.py` 中集中处理错误，返回友好信息。

## 计划图示 (Mermaid)

```mermaid
graph TD
    A[app.py (Streamlit UI & Flow Control)] --> B{chatbuy.core.pipeline.TradingAnalysisPipeline};

    subgraph Core Logic Abstraction
        B --> C[Step 1: Fetch Data Interface];
        B --> D[Step 2: Generate Image Interface];
        B --> E[Step 3: AI Analysis Interface];
        B --> F[Step 4: Generate Report Interface];
    end

    subgraph Concrete Implementations
        C --> G[scripts.get_crypto_data];
        D --> H[chatbuy.core.visualize_indicators];
        E --> I[chatbuy.core.evaluate_trade_signals / und_img];
        F --> J[chatbuy.core.evaluate_trade_signals];
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px