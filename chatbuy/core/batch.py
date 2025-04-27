import asyncio
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from chatbuy.core.und_img import TradePipeline
from chatbuy.logger import log


def get_image_paths(image_dir: str, exts=None):
    """Get all image file paths in the specified directory."""
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
        timestamp = filename.rsplit("_", 1)[-1]
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
    img_map = {}
    md_windows = {}

    if image_dir:
        img_map = get_image_paths_by_timestamp(image_dir)

    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            df["timestamp"] = df["timestamp"].astype(str)  # Ensure string type early
            if start_timestamp:
                df = df[df["timestamp"] >= start_timestamp]
            if end_timestamp:
                df = df[df["timestamp"] <= end_timestamp]

            # Check if df is empty after filtering before proceeding
            if not df.empty:
                # Ensure windowing doesn't go out of bounds if df is small
                for i in range(max(0, len(df) - window_size + 1)):
                    window = df.iloc[i : i + window_size]
                    # Ensure window is not empty and has 'timestamp' column
                    if not window.empty and "timestamp" in window.columns:
                        ts = str(window.iloc[-1]["timestamp"])
                        md_windows[ts] = window
            else:
                log.warning(
                    "DataFrame is empty after timestamp filtering. No market data windows will be created."
                )  # Use log.warning
        except FileNotFoundError:
            log.warning(
                f"CSV file not found at {csv_path}. Proceeding without market data."
            )  # Use log.warning
            df = pd.DataFrame()  # Ensure df exists even if file not found
        except Exception as e:
            log.error(e)
            log.error(
                f"Error reading or processing CSV {csv_path}", exc_info=True
            )  # Use log.error
            df = pd.DataFrame()  # Ensure df exists

    if not image_dir and not csv_path:
        raise ValueError("Either image_dir or csv_path must be provided.")

    # Determine timestamps present in the filtered data
    actual_img_timestamps = set(img_map.keys())
    actual_md_timestamps = set(md_windows.keys())

    # Filter timestamps based on start/end AFTER reading data
    combined_timestamps = actual_img_timestamps | actual_md_timestamps
    if start_timestamp:
        combined_timestamps = {
            ts for ts in combined_timestamps if ts >= start_timestamp
        }
    if end_timestamp:
        combined_timestamps = {ts for ts in combined_timestamps if ts <= end_timestamp}

    # Filter the maps/windows based on the final set of timestamps
    filtered_img_map = {
        ts: path for ts, path in img_map.items() if ts in combined_timestamps
    }
    filtered_md_windows = {
        ts: window for ts, window in md_windows.items() if ts in combined_timestamps
    }

    # Final list of timestamps to process is the union of keys from filtered data
    final_timestamps = sorted(
        list(set(filtered_img_map.keys()) | set(filtered_md_windows.keys())),
        key=lambda x: x,
    )

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
    concurrency_limit: int = 10,  # Pass concurrency as argument
):
    """Unified batch process: Creates isolated pipeline instances for each task."""
    all_timestamps, img_map, md_windows = _prepare_batch_data(
        image_dir, csv_path, window_size, start_timestamp, end_timestamp
    )

    async def process_one(ts, img_path, md_window):
        # CREATE INSTANCE INSIDE TASK: Ensures isolation
        pipeline = TradePipeline(use_openrouter=use_openrouter)
        md_text = None
        if md_window is not None:
            md_text = md_window.to_markdown(index=False)

        max_retries = 3
        retry_delay = 2  # seconds
        advice = None

        for attempt in range(max_retries):
            try:
                advice = (
                    await pipeline.a_run_pipeline(  # Use the isolated pipeline instance
                        strategy=strategy,
                        image_path=img_path,
                        markdown_text=md_text,
                    )
                )
                if advice is not None:
                    break  # Success
                else:
                    log.warning(
                        f"Received None advice for timestamp {ts} on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s..."
                    )  # Use log.warning
            except Exception as e:
                log.error(e)
                log.error(
                    f"Error processing timestamp {ts} on attempt {attempt + 1}/{max_retries}. Retrying after {retry_delay}s...",
                    exc_info=True,
                )  # Use log.error

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                log.error(
                    f"Failed to get advice for timestamp {ts} after {max_retries} attempts."
                )  # Use log.error

        if advice is not None:
            action = str(advice.action)
            reason = str(advice.reason)
            return {"trade_time": ts, "action": action, "reason": reason}
        else:
            return {
                "trade_time": ts,
                "action": "error",
                "reason": f"Failed after {max_retries} attempts",
            }

    tasks = []
    for ts in all_timestamps:
        img_path = img_map.get(ts)
        md_window = md_windows.get(ts)
        if img_path or md_window is not None:
            tasks.append(process_one(ts, img_path, md_window))

    results = []
    if not tasks:
        log.info(
            "No tasks to process based on available data and timestamps."
        )  # Use log.info
    else:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_with_progress_and_limit(task, pbar_instance):
            async with semaphore:
                result = await task
                pbar_instance.update(1)
                return result

        with tqdm(
            total=len(tasks),
            desc=f"Processing unified batch (Async, Concurrency={concurrency_limit})",
        ) as pbar:
            progress_tasks = [
                process_with_progress_and_limit(task, pbar) for task in tasks
            ]
            results = await asyncio.gather(*progress_tasks)

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df.sort_values("trade_time", inplace=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure output dir exists
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log.info(
        f"Processed {len(results)} timestamps asynchronously, results saved to {output_csv}"
    )  # Use log.info


def batch_process_sync(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_unified_results_sync.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
):
    """Synchronous version: Process each timestamp sequentially one by one."""
    pipeline = TradePipeline(use_openrouter=use_openrouter)
    all_timestamps, img_map, md_windows = _prepare_batch_data(
        image_dir, csv_path, window_size, start_timestamp, end_timestamp
    )

    results = []
    if not all_timestamps:
        log.info("No timestamps to process based on available data.")  # Use log.info
    else:
        with tqdm(
            total=len(all_timestamps), desc="Processing unified batch (Sync)"
        ) as pbar:
            for ts in all_timestamps:
                img_path = img_map.get(ts)
                md_window = md_windows.get(ts)
                action = "error"
                reason = "Processing skipped or failed"
                advice = None

                if img_path or md_window is not None:
                    md_text = (
                        md_window.to_markdown(index=False)
                        if md_window is not None
                        else None
                    )
                    try:
                        advice = pipeline.run_pipeline(
                            strategy=strategy,
                            image_path=img_path,
                            markdown_text=md_text,
                        )
                        if advice:
                            action = str(advice.action)
                            reason = str(advice.reason)
                        else:
                            action = "error"
                            reason = "Sync pipeline returned None"
                    except Exception as e:
                        log.error(
                            f"Error processing timestamp {ts} synchronously",
                            exc_info=True,
                        )  # Use log.error
                        action = "error"
                        reason = f"Sync Exception: {e}"
                else:
                    reason = "No data found for timestamp (sync)"

                results.append({"trade_time": ts, "action": action, "reason": reason})
                pbar.update(1)

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df.sort_values("trade_time", inplace=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure output dir exists
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log.info(
        f"Processed {len(results)} timestamps synchronously, results saved to {output_csv}"
    )  # Use log.info


def batch_process_threaded(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_unified_results_threaded.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
    max_workers: int = 10,
):
    """Process trading data using threading for concurrency.

    Parameters
    ----------
    image_dir : str | None, optional
        Directory containing image files, by default None.
    csv_path : str | None, optional
        Path to the CSV file containing market data, by default None.
    output_csv : str, optional
        Path to save the output CSV file, by default "output/trade_advice_unified_results_threaded.csv".
    strategy : str | None, optional
        Trading strategy description, by default None.
    use_openrouter : bool, optional
        Whether to use OpenRouter for processing, by default True.
    window_size : int, optional
        Size of the data window for processing, by default 30.
    start_timestamp : str, optional
        Start timestamp for filtering data, by default None.
    end_timestamp : str, optional
        End timestamp for filtering data, by default None.
    max_workers : int, optional
        Maximum number of threads to use, by default 10.

    Returns:
    -------
    None
        Saves the processed results to the specified output CSV file.
    """
    all_timestamps, img_map, md_windows = _prepare_batch_data(
        image_dir, csv_path, window_size, start_timestamp, end_timestamp
    )
    results = []

    def process_one(ts, img_path, md_window):
        pipeline = TradePipeline(use_openrouter=use_openrouter)
        md_text = md_window.to_markdown(index=False) if md_window is not None else None
        try:
            advice = pipeline.run_pipeline(
                strategy=strategy,
                image_path=img_path,
                markdown_text=md_text,
            )
            if advice:
                return {
                    "trade_time": ts,
                    "action": str(advice.action),
                    "reason": str(advice.reason),
                }
            else:
                return {
                    "trade_time": ts,
                    "action": "error",
                    "reason": "Sync pipeline returned None",
                }
        except Exception as e:
            return {
                "trade_time": ts,
                "action": "error",
                "reason": f"Sync Exception: {e}",
            }

    if not all_timestamps:
        log.info("No timestamps to process based on available data.")  # Use log.info
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ts = {
                executor.submit(
                    process_one, ts, img_map.get(ts), md_windows.get(ts)
                ): ts
                for ts in all_timestamps
            }
            for future in tqdm(
                as_completed(future_to_ts),
                total=len(future_to_ts),
                desc="Processing (Threaded)",
            ):
                results.append(future.result())

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df.sort_values("trade_time", inplace=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log.info(
        f"Processed {len(results)} timestamps with threading, results saved to {output_csv}"
    )


async def run_validation_retry_process(
    image_dir: str | None = None,
    csv_path: str | None = None,
    output_csv: str = "output/trade_advice_final_results.csv",
    strategy: str | None = None,
    use_openrouter: bool = True,
    window_size: int = 30,
    start_timestamp: str = None,
    end_timestamp: str = None,
    async_concurrency: int = 10,
):
    """Optimized: Run async batch first, then only retry failed timestamps synchronously.

    This avoids redundant pipeline runs and saves time/tokens.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure output dir exists
    async_temp_csv = f"output/_temp_async_limit_{async_concurrency}.csv"
    os.makedirs(os.path.dirname(async_temp_csv), exist_ok=True)

    log.info(
        f"--- Step 1: Running Asynchronous Batch (Concurrency={async_concurrency}) ---"
    )
    await batch_process(
        image_dir=image_dir,
        csv_path=csv_path,
        output_csv=async_temp_csv,
        strategy=strategy,
        use_openrouter=use_openrouter,
        window_size=window_size,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        concurrency_limit=async_concurrency,
    )

    log.info(
        "--- Step 3: Comparing Results and Identifying Retries ---"
    )  # Use log.info
    try:  # Outer try for file loading and comparison logic
        # The following baseline_csv logic is deprecated and should be removed.
        # baseline_df = pd.read_csv(baseline_csv)
        async_df = pd.read_csv(async_temp_csv)
    except FileNotFoundError:
        log.error(
            "Baseline or async temp file not found. Aborting comparison."
        )  # Use log.error
        # No need to clean up here, finally will handle it
        return  # Exit if files aren't ready
    else:  # This block executes only if the try block succeeds (files loaded)
        # Ensure dataframes are not empty before merging
        # Remove all baseline_df logic, only keep async_df logic.
        if async_df.empty:
            log.warning("Async result dataframe is empty. Nothing to retry.")
            async_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            log.info(f"Saved empty async results to {output_csv}")
            try:
                if os.path.exists(async_temp_csv):
                    os.remove(async_temp_csv)
            except OSError as e:
                log.warning(f"Could not remove async temp file: {e}")
            return

        retry_timestamps = async_df.loc[
            async_df["action"] == "error", "trade_time"
        ].tolist()

        if not retry_timestamps:
            log.info("No failed timestamps in async results. Saving as final output.")
            async_df.sort_values("trade_time", inplace=True)
            async_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            try:
                if os.path.exists(async_temp_csv):
                    os.remove(async_temp_csv)
            except OSError as e:
                log.warning(f"Could not remove async temp file: {e}")
            log.info(f"Final results saved to {output_csv}")
            return

        log.info(
            f"Found {len(retry_timestamps)} timestamps requiring synchronous retry."
        )
        log.info("--- Step 3: Performing Synchronous Retries ---")

        _, img_map_full, md_windows_full = _prepare_batch_data(
            image_dir, csv_path, window_size, start_timestamp, end_timestamp
        )

        retry_pipeline = TradePipeline(use_openrouter=use_openrouter)
        retry_results = []

        with tqdm(
            total=len(retry_timestamps),
            desc="Retrying failed timestamps (Sync)",
        ) as pbar:
            for ts in retry_timestamps:
                img_path = img_map_full.get(ts)
                md_window = md_windows_full.get(ts)
                action = "error"
                reason = "Retry failed or skipped"
                advice = None

                try:
                    if img_path or md_window is not None:
                        md_text = (
                            md_window.to_markdown(index=False)
                            if md_window is not None
                            else None
                        )
                        advice = retry_pipeline.run_pipeline(
                            strategy=strategy,
                            image_path=img_path,
                            markdown_text=md_text,
                        )
                        if advice:
                            action = str(advice.action)
                            reason = str(advice.reason) + " (Retried)"
                        else:
                            reason = "Retry pipeline returned None"
                    else:
                        reason = "No data found for retry timestamp (unexpected)"
                except Exception as e:
                    log.error(f"Error during retry for timestamp {ts}", exc_info=True)
                    reason = f"Retry Exception: {e}"

                retry_results.append(
                    {
                        "trade_time": ts,
                        "action": action,
                        "reason": reason,
                    }
                )
                pbar.update(1)

        log.info("--- Step 4: Merging Async and Retry Results ---")
        retry_df = pd.DataFrame(retry_results)

        if not retry_df.empty:
            final_df = async_df.copy()
            final_df.set_index("trade_time", inplace=True)
            retry_df.set_index("trade_time", inplace=True)
            final_df.update(retry_df)
            final_df.reset_index(inplace=True)
            final_df.sort_values("trade_time", inplace=True)
            log.info(f"Saving final combined results to {output_csv}")
            final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        else:
            log.warning(
                "Retry process yielded no results. Saving original async results."
            )
            async_df.sort_values("trade_time", inplace=True)
            async_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        try:
            if os.path.exists(async_temp_csv):
                os.remove(async_temp_csv)
                log.info("Successfully removed async temp file.")
        except OSError as e:
            log.warning(f"Could not remove async temp file: {e}")

    finally:  # This finally corresponds to the outer try block starting at line 336 (Corrected Alignment)
        log.info("--- Step 6: Cleaning up temporary files ---")  # Use log.info
        cleaned_count = 0
        try:  # Inner try for cleanup actions
            # Remove baseline_csv cleanup, as baseline_csv is no longer used.
            if os.path.exists(async_temp_csv):
                os.remove(async_temp_csv)
                cleaned_count += 1
            log.info(
                f"Successfully removed {cleaned_count} temporary file(s)."
            )  # Use log.info
        except OSError as e:
            log.warning(
                f"Could not remove one or more temporary files: {e}"
            )  # Use log.warning


# --- Main execution block ---
if __name__ == "__main__":  # Corrected Alignment (No Indentation)
    # Define parameters
    image_dir = "data/btc_daily"
    csv_path = "data/BTC_USDT_1d_with_indicators.csv"
    output_file = "output/trade_advice_final_results_refactored.csv"
    strategy = "仅分析最后一天的K线数据。当日收盘价跌破布林带下轨时买入，当日收盘价超过布林带上轨时卖出，否则持有。"
    start_date = "2021-06-30"
    end_date = "2021-12-31"
    concurrency = 10
    w_size = 120
    use_router = False

    log.info("Starting validation process...")  # Use log.info
    log.info(
        f"Params: Image Dir='{image_dir}', CSV Path='{csv_path}', Output='{output_file}'"
    )  # Use log.info
    log.info(
        f"Strategy='{strategy[:50]}...', Dates={start_date}-{end_date}, Concurrency={concurrency}"
    )  # Use log.info

    batch_process_threaded(
        image_dir=image_dir,
        csv_path=csv_path,
        output_csv=output_file,
        strategy=strategy,
        window_size=w_size,
        use_openrouter=use_router,
        start_timestamp=start_date,
        end_timestamp=end_date,
        max_workers=concurrency,
    )
    print("Threaded batch process finished.")
