# Code Refactoring and Optimization Plan

## Analysis Summary

The current `app.py` has the following issues:

1.  **Streamlit Runtime Error:** Improper placement of the `st.set_page_config()` call.
2.  **Hardcoding and Lack of Flexibility:** Imported function names and paths are hardcoded in `app.py`.
3.  **Single File Overloaded:** `app.py` handles too many responsibilities, making it difficult to maintain.
4.  **Unclear Function Interfaces:** The signatures of underlying functions are uncertain.
5.  **Code Duplication:** UI and logic for various steps are similar.
6.  **Error Handling:** Can be made more user-friendly.

## Optimization Plan Steps

1.  **Fix Streamlit Error:** Move `st.set_page_config(layout="wide")` to the top of the `app.py` file.
2.  **Abstract Core Logic:**
    *   Create `pipeline.py` under `chatbuy/core/`.
    *   Define a `TradingAnalysisPipeline` class or function to encapsulate the workflow.
    *   `pipeline.py` will be responsible for interacting with specific implementations and providing a stable interface.
3.  **Simplify `app.py`:**
    *   `app.py` will handle UI layout and call the `pipeline` interface.
    *   Extract repetitive UI code into reusable functions.
4.  **Standardize Function Interfaces:** Clearly define the inputs and outputs of functions in the `chatbuy/core/` modules.
5.  **Improve State Management:** Simplify the initialization of `st.session_state`.
6.  **Enhance Error Handling:** Centralize error handling in `pipeline.py` and return user-friendly messages.

## Plan Diagram (Mermaid)

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