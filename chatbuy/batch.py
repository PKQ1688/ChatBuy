import asyncio
import glob
import os

import pandas as pd
from tqdm import tqdm

from chatbuy.und_img import TradePipeline


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
        advice = await pipeline.a_run_pipeline(
            strategy=strategy,
            image_path=img_path,
            markdown_text=md_text,
        )
        action = str(advice.action)
        reason = str(advice.reason)
        return {
            "trade_time": ts,
            "action": action,
            "reason": reason,
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
    # concurrency_limit = 10 # 根据需要调整并发限制
    # semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_with_progress_and_limit(task):
        # async with semaphore: # 应用并发限制
        result = await task
        pbar.update(1)
        return result

    # 辅助函数，用于在异步代码中更新 tqdm 进度条 (需要定义或传递 pbar)
    async def process_with_progress(task, pbar_instance):
        result = await task
        pbar_instance.update(1)
        return result

    with tqdm(total=len(tasks), desc="Processing unified batch") as pbar:
        # progress_tasks = [process_with_progress_and_limit(task) for task in tasks]
        # 使用原始的并发处理，如果怀疑是并发问题，再启用 Semaphore
        # 将 pbar 实例传递给 process_with_progress
        progress_tasks = [process_with_progress(task, pbar) for task in tasks]
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


if __name__ == "__main__":
    image_dir = "data/btc_daily"
    csv_path = "data/BTC_USDT_1d_with_indicators.csv"
    # strategy = "只分析最后一天的K线数据。当价格跌破布林线下轨时买入，当价格升至布林线上轨时卖出，否则持有"
    strategy = "只分析最后一天的K线数据。当天的收盘价格跌破布林线下轨时买入，当收盘价格超过布林线上轨时卖出，否则持有"

    # 异步批处理
    asyncio.run(
        batch_process(
            image_dir=image_dir,
            # csv_path=csv_path,
            output_csv="output/trade_advice_unified_results.csv",
            strategy=strategy,
            window_size=120,
            use_openrouter=False,
            start_timestamp="2021-06-30",
            end_timestamp="2021-12-31",
        )
    )

    # 同步批处理（如需同步执行，取消注释即可）
    # batch_process_sync(
    #     image_dir=image_dir,
    #     # csv_path=csv_path,
    #     output_csv="output/trade_advice_unified_results_one.csv",
    #     strategy=strategy,
    #     window_size=120,
    #     use_openrouter=False,
    #     start_timestamp="2021-06-30",
    #     end_timestamp="2021-12-31",
    # )
