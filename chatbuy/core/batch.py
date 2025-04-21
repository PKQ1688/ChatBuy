import asyncio
import glob
import os

import pandas as pd
from tqdm import tqdm

from chatbuy.core.und_img import TradePipeline


def get_image_paths(image_dir: str, exts=None):
    """获取指定目录下所有图片文件路径."""
    if exts is None:
        exts = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def get_image_paths_by_timestamp(image_dir: str, exts=None):
    """Return a dict mapping timestamp to image path."""
    if exts is None:
        exts = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    img_map = {}
    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        # Assume timestamp is after last underscore
        timestamp = filename.rsplit("_", 1)[-1]
        # 转换20210510为2021-05-10格式
        if len(timestamp) == 8 and timestamp.isdigit():
            timestamp_fmt = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:]}"
        else:
            timestamp_fmt = timestamp
        img_map[timestamp_fmt] = path
    return img_map


def _prepare_batch_data(
    image_dir: str | None = None,
    csv_path: str | None = None,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
):
    img_timestamps = set()
    md_timestamps = set()
    img_map = {}
    md_windows = {}

    if image_dir:
        img_map = get_image_paths_by_timestamp(image_dir)
        img_timestamps = set(img_map.keys())

    if csv_path:
        df = pd.read_csv(csv_path)
        if start_timestamp or end_timestamp:
            df["timestamp"] = df["timestamp"].astype(str)
            if start_timestamp:
                df = df[df["timestamp"] >= start_timestamp]
            if end_timestamp:
                df = df[df["timestamp"] <= end_timestamp]
        df["timestamp"] = df["timestamp"].astype(str)
        md_timestamps = set(df["timestamp"])
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i : i + window_size]
            ts = str(window.iloc[-1]["timestamp"])
            md_windows[ts] = window

    if not image_dir and not csv_path:
        raise ValueError("Either image_dir or csv_path must be provided.")

    # 合并并去重时间戳，然后排序
    # 注意：这里合并了图像和CSV的时间戳，但后续处理只使用了存在图像或CSV窗口的时间戳
    all_timestamps_combined = img_timestamps | md_timestamps
    all_timestamps = sorted(list(all_timestamps_combined), key=lambda x: x)

    # 根据 start_timestamp 和 end_timestamp 过滤 all_timestamps
    if start_timestamp:
        all_timestamps = [ts for ts in all_timestamps if ts >= start_timestamp]
    if end_timestamp:
        all_timestamps = [ts for ts in all_timestamps if ts <= end_timestamp]

    # 过滤 img_map 和 md_windows，确保只包含在 all_timestamps 中的键
    filtered_img_map = {ts: path for ts, path in img_map.items() if ts in all_timestamps}
    filtered_md_windows = {ts: window for ts, window in md_windows.items() if ts in all_timestamps}

    # 重新生成 all_timestamps，只包含实际有数据的时间戳
    final_timestamps = sorted(list(set(filtered_img_map.keys()) | set(filtered_md_windows.keys())), key=lambda x: x)


    return final_timestamps, filtered_img_map, filtered_md_windows


async def batch_process(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_unified_results.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
):
    """Unified batch process: for each timestamp, process image and/or markdown if available."""
    pipeline = TradePipeline(use_openrouter=use_openrouter)
    all_timestamps, img_map, md_windows = _prepare_batch_data(
        image_dir, csv_path, window_size, start_timestamp, end_timestamp
    )

    async def process_one(ts, img_path, md_window):
        md_text = None
        if md_window is not None:
            md_text = md_window.to_markdown(index=False)
        
        max_retries = 3
        retry_delay = 2 # seconds
        advice = None # Initialize advice to None

        for attempt in range(max_retries):
            try:
                advice = await pipeline.a_run_pipeline(
                    strategy=strategy,
                    image_path=img_path,
                    markdown_text=md_text,
                )
                if advice is not None:
                    # Successfully got advice, break the retry loop
                    break
                else:
                    # Received None, log and prepare for retry
                    print(f"Warning: Received None advice for timestamp {ts} on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...")
                    
            except Exception as e:
                # Catch potential exceptions during the pipeline run
                print(f"Error processing timestamp {ts} on attempt {attempt + 1}/{max_retries}: {e}. Retrying after {retry_delay}s...")
            
            # If not the last attempt, wait before retrying
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                 # Last attempt failed
                 print(f"Error: Failed to get advice for timestamp {ts} after {max_retries} attempts.")


        # After the loop, check if advice was successfully obtained
        if advice is not None:
            action = str(advice.action)
            reason = str(advice.reason)
            return {
                "trade_time": ts,
                "action": action,
                "reason": reason,
            }
        else:
            # All retries failed
            return {
                "trade_time": ts,
                "action": "error",
                "reason": f"Failed after {max_retries} attempts",
            }

    tasks = []
    # 确保只为实际存在数据的时间戳创建任务
    for ts in all_timestamps: # 使用 _prepare_batch_data 返回的 final_timestamps
        img_path = img_map.get(ts)
        md_window = md_windows.get(ts)
        # 确保至少有一个数据源存在才处理
        if img_path or md_window is not None:
             # 确保 pipeline 实例对于每个并发任务是独立的，或者其方法是无状态且线程安全的
            tasks.append(process_one(ts, img_path, md_window))

    results = []
    # 使用 asyncio.Semaphore 来限制并发数量，防止外部服务过载
    concurrency_limit = 10 # Set for validation run
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_with_progress_and_limit(task): # Use this function
        async with semaphore: # 应用并发限制
            result = await task
            pbar.update(1)
            return result

    # 辅助函数，用于在异步代码中更新 tqdm 进度条 (需要定义或传递 pbar)
    # async def process_with_progress(task, pbar_instance):
    #     result = await task
    #     pbar_instance.update(1)
    #     return result

    with tqdm(total=len(tasks), desc="Processing unified batch") as pbar:
        progress_tasks = [process_with_progress_and_limit(task) for task in tasks] # 使用带限制的函数
        # 使用原始的并发处理，如果怀疑是并发问题，再启用 Semaphore
        # 将 pbar 实例传递给 process_with_progress
        # progress_tasks = [process_with_progress(task, pbar) for task in tasks]
        results = await asyncio.gather(*progress_tasks)


    out_df = pd.DataFrame(results)
    # 确保结果按时间排序
    if not out_df.empty:
        out_df.sort_values("trade_time", inplace=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Processed {len(results)} timestamps, results saved to {output_csv}")


def batch_process_sync(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_unified_results.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
):
    """同步版本：逐个顺序处理每个 timestamp."""
    pipeline = TradePipeline(use_openrouter=use_openrouter)
    all_timestamps, img_map, md_windows = _prepare_batch_data(
        image_dir, csv_path, window_size, start_timestamp, end_timestamp
    )

    results = []
    # 使用 _prepare_batch_data 返回的过滤后的 all_timestamps
    with tqdm(
        total=len(all_timestamps), desc="Processing unified batch (sync)"
    ) as pbar:
        for ts in all_timestamps: # 使用 _prepare_batch_data 返回的 final_timestamps
            img_path = img_map.get(ts)
            md_window = md_windows.get(ts)
            # 确保至少有一个数据源存在才处理
            if img_path or md_window is not None:
                md_text = None
                if md_window is not None:
                    md_text = md_window.to_markdown(index=False)
                advice = pipeline.run_pipeline(
                    strategy=strategy,
                    image_path=img_path,
                    markdown_text=md_text,
                )
                action = str(advice.action)
                reason = str(advice.reason)
                results.append(
                    {
                        "trade_time": ts,
                        "action": action,
                        "reason": reason,
                    }
                )
            pbar.update(1) # 每次循环更新进度

    out_df = pd.DataFrame(results)
    # 确保结果按时间排序
    if not out_df.empty:
        out_df.sort_values("trade_time", inplace=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Processed {len(results)} timestamps, results saved to {output_csv}")


# --- New function for Validation Retry Process ---
async def run_validation_retry_process(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_final_results.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
    async_concurrency: int = 10, # Concurrency for the initial async run
):
    """Runs batch process with validation and retry."""
    baseline_csv = "output/_temp_baseline_sync.csv"
    async_temp_csv = f"output/_temp_async_limit_{async_concurrency}.csv"

    print("--- Step 1: Running Synchronous Baseline ---")
    batch_process_sync(
        image_dir=image_dir,
        csv_path=csv_path,
        output_csv=baseline_csv,
        strategy=strategy,
        use_openrouter=use_openrouter,
        window_size=window_size,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    print(f"\n--- Step 2: Running Asynchronous Batch (Concurrency={async_concurrency}) ---")
    # Ensure the async batch process uses the specified concurrency
    # Temporarily modify the concurrency limit within batch_process or pass it as an argument if refactored
    # For simplicity here, we assume batch_process is modified or we modify it before calling
    # Let's modify it directly for this example (needs refactoring for cleaner approach)
    
    # --- Quick modification to set concurrency for the async run ---
    # This is not ideal, refactoring batch_process to accept concurrency is better
    original_concurrency = 10 # Default or previous value, store to restore later if needed
    # Find and replace the concurrency limit line (this is fragile)
    try:
        with open("chatbuy/core/batch.py", "r+") as f:
            content = f.readlines()
            for i, line in enumerate(content):
                if "concurrency_limit =" in line and "semaphore = asyncio.Semaphore" in content[i+1]:
                    content[i] = f"    concurrency_limit = {async_concurrency} # Set for validation run\n"
                    break
            f.seek(0)
            f.writelines(content)
            f.truncate()
    except Exception as e:
        print(f"Warning: Could not dynamically set concurrency limit in file: {e}")
        print(f"Proceeding with potentially incorrect concurrency limit for async run.")
        
    await batch_process(
        image_dir=image_dir,
        csv_path=csv_path,
        output_csv=async_temp_csv,
        strategy=strategy,
        use_openrouter=use_openrouter,
        window_size=window_size,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        # If batch_process is refactored, pass concurrency: concurrency=async_concurrency
    )
    
    # --- Restore original concurrency if needed (or handle via proper refactoring) ---
    # try:
    #     with open("chatbuy/core/batch.py", "r+") as f:
    #         # ... restore logic ...
    # except Exception as e:
    #     print(f"Warning: Could not restore concurrency limit in file: {e}")


    print("\n--- Step 3: Comparing Results and Identifying Retries ---")
    try:
        baseline_df = pd.read_csv(baseline_csv)
        async_df = pd.read_csv(async_temp_csv)
    except FileNotFoundError:
        print("Error: Baseline or async temp file not found. Aborting.")
        return

    # Merge results to easily compare actions
    merged_df = pd.merge(
        baseline_df,
        async_df,
        on="trade_time",
        suffixes=('_baseline', '_async'),
        how="left" # Keep all baseline entries
    )

    # Identify timestamps needing retry
    # Condition 1: Async result is 'error'
    # Condition 2: Async result action is different from baseline action
    retry_conditions = (merged_df['action_async'] == 'error') | \
                       (merged_df['action_async'].fillna('') != merged_df['action_baseline'].fillna(''))

    retry_timestamps = merged_df.loc[retry_conditions, 'trade_time'].tolist()

    if not retry_timestamps:
        print("No discrepancies found. Async results match baseline (excluding errors). Using async results.")
        # If async had no errors and matched baseline where not error, we can just use async_df
        # However, since async might have errors, we start with baseline and update
        final_df = baseline_df.copy()
        # Overwrite with async results where they are not errors and match baseline (or if baseline is preferred)
        # Simplest: just use baseline if no retries needed and baseline is considered ground truth
        print(f"Saving baseline results to {output_csv}")
        final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    else:
        print(f"Found {len(retry_timestamps)} timestamps requiring synchronous retry: {retry_timestamps}")
        print("\n--- Step 4: Performing Synchronous Retries ---")

        # Prepare data needed for retry
        all_timestamps_full, img_map_full, md_windows_full = _prepare_batch_data(
             image_dir, csv_path, window_size, start_timestamp, end_timestamp
        ) # Reuse data prep

        retry_pipeline = TradePipeline(use_openrouter=use_openrouter) # New pipeline for sync retry
        retry_results = []

        with tqdm(total=len(retry_timestamps), desc="Retrying failed/mismatched timestamps") as pbar:
            for ts in retry_timestamps:
                img_path = img_map_full.get(ts)
                md_window = md_windows_full.get(ts)
                action = "error"
                reason = "Retry failed"
                try:
                    if img_path or md_window is not None:
                        md_text = md_window.to_markdown(index=False) if md_window is not None else None
                        advice = retry_pipeline.run_pipeline( # Use sync run_pipeline
                            strategy=strategy,
                            image_path=img_path,
                            markdown_text=md_text,
                        )
                        if advice:
                            action = str(advice.action)
                            reason = str(advice.reason) + " (Retried)" # Mark as retried
                        else:
                             reason = "Retry returned None"
                    else:
                        reason = "No data found for retry timestamp" # Should not happen if logic is correct

                except Exception as e:
                    print(f"Error during retry for timestamp {ts}: {e}")
                    reason = f"Retry Exception: {e}"

                retry_results.append({
                    "trade_time": ts,
                    "action": action,
                    "reason": reason,
                })
                pbar.update(1)

        print("\n--- Step 5: Merging Final Results ---")
        retry_df = pd.DataFrame(retry_results)

        # Start with the initial async results
        final_df = async_df.copy()
        # Update the rows that needed retry with the retry results
        final_df.set_index('trade_time', inplace=True)
        retry_df.set_index('trade_time', inplace=True)
        final_df.update(retry_df)
        final_df.reset_index(inplace=True)

        # Ensure sorting
        final_df.sort_values("trade_time", inplace=True)

        print(f"Saving final combined results to {output_csv}")
        final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Clean up temporary files
    try:
        os.remove(baseline_csv)
        os.remove(async_temp_csv)
        print("Temporary files removed.")
    except OSError as e:
        print(f"Warning: Could not remove temporary files: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    image_dir = "data/btc_daily"
    csv_path = "data/BTC_USDT_1d_with_indicators.csv"
    strategy = "只分析最后一天的K线数据。当天的收盘价格跌破布林线下轨时买入，当收盘价格超过布林线上轨时卖出，否则持有"

    asyncio.run(
        run_validation_retry_process(
            image_dir=image_dir,
            # csv_path=csv_path,
            output_csv="output/trade_advice_final_results.csv", # Final output file
            strategy=strategy,
            window_size=120,
            use_openrouter=False,
            start_timestamp="2021-06-30",
            end_timestamp="2021-12-31",
            async_concurrency=10 # Use concurrency 10 for the fast async pass
        )
    )
