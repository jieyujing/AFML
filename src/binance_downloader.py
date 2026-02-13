#!/usr/bin/env python3
"""
Binance Historical Data Downloader

从 Binance Vision 下载 BTCUSDT 聚合交易数据（aggTrades）。
支持日期范围下载、断点续传、增量回补。
"""

import argparse
import hashlib
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests


# 常量配置
BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT"
DEFAULT_DATA_DIR = "data/BTCUSDT/aggTrades"
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Binance 历史数据下载工具 - 下载 BTCUSDT aggTrades 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  下载指定日期范围:
    python src/binance_downloader.py --start 2026-02-01 --end 2026-02-10

  下载从某天到今天:
    python src/binance_downloader.py --start 2026-02-01

  启用增量回补模式:
    python src/binance_downloader.py --start 2026-02-01 --end 2026-02-10 --backfill

  自定义数据目录:
    python src/binance_downloader.py --start 2026-02-01 --end 2026-02-10 --dir my_data
""",
    )
    parser.add_argument("--start", required=True, help="开始日期 (格式: YYYY-MM-DD)")
    parser.add_argument("--end", help="结束日期 (默认: 今天)")
    parser.add_argument(
        "--backfill", action="store_true", help="增量回补模式: 只下载缺失的文件"
    )
    parser.add_argument(
        "--dir", default=DEFAULT_DATA_DIR, help=f"下载目录 (默认: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=MAX_RETRIES,
        help=f"下载失败重试次数 (默认: {MAX_RETRIES})",
    )

    return parser.parse_args()


def parse_date(date_str: str) -> date:
    """解析日期字符串。"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"无效日期格式: {date_str}，应使用 YYYY-MM-DD")


def generate_date_range(start_date: date, end_date: date) -> List[date]:
    """生成日期范围内的所有日期列表。"""
    if start_date > end_date:
        raise ValueError(f"开始日期 {start_date} 晚于结束日期 {end_date}")

    delta = end_date - start_date
    return [start_date + timedelta(days=i) for i in range(delta.days + 1)]


def get_filename(target_date: date) -> str:
    """生成目标日期的文件名。"""
    return f"BTCUSDT-aggTrades-{target_date.strftime('%Y-%m-%d')}.zip"


def get_url(target_date: date) -> str:
    """生成目标日期的下载 URL。"""
    return f"{BASE_URL}/{get_filename(target_date)}"


def get_checksum_filename(target_date: date) -> str:
    """生成目标日期的 checksum 文件名。"""
    return f"{get_filename(target_date)}.CHECKSUM"


def get_checksum_url(target_date: date) -> str:
    """生成目标日期的 checksum 下载 URL。"""
    return f"{BASE_URL}/{get_checksum_filename(target_date)}"


def download_checksum(url: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """下载 checksum 文件并返回内容。"""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text.strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
            else:
                print(f"    Checksum 下载失败: {e}")
                return None
    return None


def verify_checksum(filepath: Path, checksum_content: str) -> bool:
    """使用 SHA256 校验文件完整性。"""
    expected_hash = checksum_content.split()[0]
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)

    actual_hash = sha256_hash.hexdigest().lower()
    expected = expected_hash.lower()

    return actual_hash == expected


def ensure_directory(data_dir: str) -> Path:
    """确保数据目录存在。"""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_existing_files(data_dir: str) -> set:
    """获取目录下已存在的文件集合。"""
    path = Path(data_dir)
    if not path.exists():
        return set()

    return {f.name for f in path.glob("BTCUSDT-aggTrades-*.zip")}


def download_file(url: str, filepath: Path, max_retries: int = MAX_RETRIES) -> bool:
    """
    下载单个文件。

    Returns:
        True 表示下载成功，False 表示失败
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            return True

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"  下载失败 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  下载失败 (已重试 {max_retries} 次): {e}")
                return False

    return False


def download_range(
    dates: List[date],
    data_dir: str,
    backfill: bool = False,
    max_retries: int = MAX_RETRIES,
) -> tuple:
    """
    下载指定日期范围的数据。

    Returns:
        (成功下载数, 跳过数)
    """
    data_path = ensure_directory(data_dir)
    existing_files = get_existing_files(data_dir)

    downloaded = 0
    skipped = 0

    for target_date in dates:
        filename = get_filename(target_date)
        filepath = data_path / filename

        # 检测已存在文件
        if filename in existing_files:
            if backfill:
                # 回补模式下：跳过已存在的文件
                print(f"  跳过 (已存在): {filename}")
                skipped += 1
                continue
            else:
                # 普通模式：跳过已存在的文件（断点续传）
                print(f"  跳过 (已存在): {filename}")
                skipped += 1
                continue

        url = get_url(target_date)
        print(f"  下载: {filename} ...")

        if download_file(url, filepath, max_retries):
            checksum_url = get_checksum_url(target_date)
            checksum_content = download_checksum(checksum_url, max_retries)

            if checksum_content is None:
                print("    警告: 无法下载 Checksum 文件，跳过校验")
                downloaded += 1
                print(f"    完成: {filename} ({filepath.stat().st_size} bytes)")
            elif verify_checksum(filepath, checksum_content):
                downloaded += 1
                print(
                    f"    完成: {filename} ({filepath.stat().st_size} bytes) - 校验通过"
                )
            else:
                filepath.unlink()
                print(f"    校验失败，文件已删除: {filename}")
        else:
            # 下载失败，删除不完整的文件
            if filepath.exists():
                filepath.unlink()
            print(f"    失败: {filename}")

    return downloaded, skipped


def main():
    """主函数。"""
    args = parse_args()

    # 解析日期
    try:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end) if args.end else date.today()
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 验证日期
    if start_date > end_date:
        print(f"错误: 开始日期 {start_date} 晚于结束日期 {end_date}", file=sys.stderr)
        sys.exit(1)

    if end_date > date.today():
        print(
            f"警告: 结束日期 {end_date} 晚于今天 {date.today()}，将只下载到今天",
            file=sys.stderr,
        )
        end_date = date.today()

    # 生成日期列表
    dates = generate_date_range(start_date, end_date)
    print(f"\n目标日期范围: {start_date} -> {end_date} ({len(dates)} 天)")

    # 检查现有文件
    existing = get_existing_files(args.dir)
    print(f"已存在文件: {len(existing)} 个\n")

    # 开始下载
    print(f"开始下载到目录: {args.dir}")
    if args.backfill:
        print("模式: 增量回补 (只下载缺失文件)\n")
    else:
        print("模式: 完整下载\n")

    downloaded, skipped = download_range(
        dates, args.dir, backfill=args.backfill, max_retries=args.retry
    )

    print(f"\n下载完成: {downloaded} 个文件, {skipped} 个跳过")


if __name__ == "__main__":
    main()
