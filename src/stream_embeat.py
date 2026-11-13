

import os
import sys
import io
import logging
from typing import Callable, Optional, List
import requests
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from src.preprocess import validate_and_clean_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_REPO = "GD-Studio/embeat_45m_spotify_tracks"
PARQUET_FILES: List[str] = [
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
]
#DEFAULT_LOCAL_SAMPLE = "data/processed/sample_tracks.csv"
DEFAULT_LOCAL_SAMPLE = "data/processed/embeat_sample_tracks.csv"


def test_token_access(hf_token: Optional[str]) -> bool:
    """Kiểm tra tính hợp lệ của HF_TOKEN và quyền truy cập vào repo Embeat 45M."""
    if not hf_token or hf_token.strip() == "" or "your_access_token" in hf_token:
        logging.warning("Chưa cấu hình HF_TOKEN hợp lệ.")
        return False

    url = f"https://huggingface.co/api/datasets/{HF_REPO}"
    headers = {"Authorization": f"Bearer {hf_token.strip()}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            logging.info("Xác thực HF_TOKEN thành công! Đã kết nối tới GD-Studio/embeat_45m_spotify_tracks.")
            return True
        elif resp.status_code == 403:
            logging.error(
                "Token hợp lệ nhưng tài khoản chưa bấm 'Accept License' tại: "
                "https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks"
            )
            return False
        else:
            logging.error(f"Không thể truy cập repository: HTTP {resp.status_code}")
            return False
    except Exception as exc:
        logging.error(f"Lỗi kết nối mạng tới Hugging Face Hub: {exc}")
        return False


def stream_from_local_sample(
    sample_csv_path: str = DEFAULT_LOCAL_SAMPLE,
    callback_fn: Optional[Callable[[pd.DataFrame], None]] = None,
    batch_size: int = 1000,
    min_popularity: float = 0.0,
) -> bool:
    """
    Luồng Fallback Offline: Đọc tập dữ liệu mẫu `data/processed/sample_tracks.csv`
    theo từng batch và truyền vào `callback_fn` với đúng giao ước Schema của Embeat 45M.
    """
    if not os.path.exists(sample_csv_path):
        logging.error(f"Không tìm thấy tệp dữ liệu mẫu offline tại: {sample_csv_path}")
        return False

    logging.info(f"Đang đọc dữ liệu theo chế độ Local Sample từ: {sample_csv_path}...")
    df_full = pd.read_csv(sample_csv_path)
    df_clean = validate_and_clean_batch(df_full, deduplicate=True, min_popularity=min_popularity)

    total_rows = len(df_clean)
    processed = 0
    for start_idx in range(0, total_rows, batch_size):
        batch_df = df_clean.iloc[start_idx : start_idx + batch_size].copy()
        processed += len(batch_df)
        if callback_fn:
            callback_fn(batch_df)
        logging.info(f"[Local Stream] Đã nạp batch: {processed:,}/{total_rows:,} bài hát.")

    logging.info(f"Hoàn tất nạp dữ liệu từ Local Sample: Tổng cộng {processed:,} bài hát sạch.")
    return True


def stream_and_train_50pct(
    hf_token: Optional[str] = None,
    callback_fn: Optional[Callable[[pd.DataFrame], None]] = None,
    sample_ratio: float = 0.5,
    max_row_groups_per_file: Optional[int] = None,
    max_total_tracks: Optional[int] = None,
    min_popularity: float = 0.0,
    clean_batches: bool = True,
    fallback_to_local: bool = True,
) -> bool:
    """
    Stream các phân vùng Parquet của Embeat 45M từ Hugging Face Hub theo từng Row Group.
    - Tự động lấy mẫu theo tỷ lệ `sample_ratio` (mặc định 50%).
    - Tự động đi qua chốt kiểm định chất lượng `validate_and_clean_batch` nếu `clean_batches=True`.
    """
    token = hf_token or os.environ.get("HF_TOKEN")

    if not test_token_access(token):
        if fallback_to_local and os.path.exists(DEFAULT_LOCAL_SAMPLE):
            logging.warning(
                "Không thể kết nối Cloud Stream. Tự động chuyển sang chế độ Fallback Offline "
                f"với '{DEFAULT_LOCAL_SAMPLE}'."
            )
            return stream_from_local_sample(
                sample_csv_path=DEFAULT_LOCAL_SAMPLE,
                callback_fn=callback_fn,
                min_popularity=min_popularity,
            )
        return False

    if pq is None:
        logging.error("Thư viện 'pyarrow' chưa được cài đặt. Hãy chạy: pip install pyarrow")
        return False

    headers = {"Authorization": f"Bearer {token.strip()}"}
    total_processed = 0

    for filename in PARQUET_FILES:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
        logging.info(f"Đang kết nối luồng stream phân vùng: {filename}...")

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        if resp.status_code != 200:
            logging.error(f"Không thể đọc phân vùng {filename}: HTTP {resp.status_code}")
            continue

        buffer = io.BytesIO(resp.content)
        parquet_file = pq.ParquetFile(buffer)
        num_row_groups = parquet_file.num_row_groups
        limit_rg = (
            min(num_row_groups, max_row_groups_per_file)
            if max_row_groups_per_file is not None
            else num_row_groups
        )
        logging.info(f"Đã mở {filename}: {num_row_groups} row groups (sẽ xử lý: {limit_rg}).")

        for rg_idx in range(limit_rg):
            table = parquet_file.read_row_group(rg_idx)
            df = table.to_pandas()

            if 0.0 < sample_ratio < 1.0:
                sampled_df = df.sample(frac=sample_ratio, random_state=42)
            else:
                sampled_df = df

            if clean_batches:
                sampled_df = validate_and_clean_batch(
                    sampled_df, deduplicate=True, min_popularity=min_popularity
                )

            if sampled_df.empty:
                continue

            if max_total_tracks is not None and (total_processed + len(sampled_df)) > max_total_tracks:
                needed = max_total_tracks - total_processed
                sampled_df = sampled_df.iloc[:needed].copy()

            total_processed += len(sampled_df)
            if callback_fn:
                callback_fn(sampled_df)

            if (rg_idx + 1) % 50 == 0 or (rg_idx + 1) == limit_rg:
                logging.info(
                    f"Đã xử lý row group {rg_idx + 1}/{limit_rg} | "
                    f"Tổng số bài hát đã nạp: {total_processed:,}"
                )

            if max_total_tracks is not None and total_processed >= max_total_tracks:
                logging.info(f"Đã đạt ngưỡng giới hạn yêu cầu: {total_processed:,} bài hát.")
                return True

    logging.info(f"Hoàn tất Cloud Streaming. Tổng số bài hát đã cấp phát: {total_processed:,}.")
    return total_processed > 0


if __name__ == "__main__":
    cli_token = os.environ.get("HF_TOKEN")
    if not cli_token and len(sys.argv) > 1:
        cli_token = sys.argv[1]

    stream_and_train_50pct(
        hf_token=cli_token,
        sample_ratio=0.5,
        max_total_tracks=5000,
    )

