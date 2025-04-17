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

    all_timestamps = sorted(img_timestamps | md_timestamps, key=lambda x: x)
    if start_timestamp:
        all_timestamps = [ts for ts in all_timestamps if ts >= start_timestamp]
    if end_timestamp:
        all_timestamps = [ts for ts in all_timestamps if ts <= end_timestamp]

    return all_timestamps, img_map, md_windows

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
    for ts in all_timestamps:
        img_path = img_map.get(ts)
        md_window = md_windows.get(ts)
        if img_path or md_window is not None:
            tasks.append(process_one(ts, img_path, md_window))

    with tqdm(total=len(tasks), desc="Processing unified batch") as pbar:
        async def process_with_progress(task):
            result = await task
            pbar.update(1)
            return result

        progress_tasks = [process_with_progress(task) for task in tasks]
        results = await asyncio.gather(*progress_tasks)

    out_df = pd.DataFrame(results)
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
    with tqdm(total=len(all_timestamps), desc="Processing unified batch (sync)") as pbar:
        for ts in all_timestamps:
            img_path = img_map.get(ts)
            md_window = md_windows.get(ts)
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
                results.append({
                    "trade_time": ts,
                    "action": action,
                    "reason": reason,
                })
            pbar.update(1)

    out_df = pd.DataFrame(results)
    out_df.sort_values("trade_time", inplace=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Processed {len(results)} timestamps, results saved to {output_csv}")

if __name__ == "__main__":
    image_dir = "data/btc_daily"
    output_csv = "output/trade_advice_unified_results_one.csv"
    csv_path = "data/BTC_USDT_1d_with_indicators.csv"
    strategy = "只分析最后一天的K线数据。当价格跌破布林线下轨时买入，当价格升至布林线上轨时卖出，否则持有。"

    # 异步批处理
    # asyncio.run(
    #     batch_process(
    #         image_dir=image_dir,
    #         # csv_path=csv_path,
    #         output_csv=output_csv,
    #         strategy=strategy,
    #         window_size=120,
    #         use_openrouter=True,
    #         start_timestamp="2021-06-30",
    #         end_timestamp="2021-12-31",
    #     )
    # )

    # 同步批处理（如需同步执行，取消注释即可）
    batch_process_sync(
        image_dir=image_dir,
        # csv_path=csv_path,
        output_csv=output_csv,
        strategy=strategy,
        window_size=120,
        use_openrouter=True,
        start_timestamp="2021-06-30",
        end_timestamp="2021-12-31",
    )
