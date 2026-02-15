import polars as pl
from pathlib import Path
import zipfile
import concurrent.futures


def load_zip_as_df(f_zip: Path) -> pl.DataFrame:
    with zipfile.ZipFile(f_zip, "r") as z:
        with z.open(z.namelist()[0]) as f:
            df = pl.read_csv(f.read())
            
            # Select requests fields with correct types
            df = df.select([
                pl.col("agg_trade_id").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                pl.col("first_trade_id").cast(pl.Int64),
                pl.col("last_trade_id").cast(pl.Int64),
                pl.col("transact_time").cast(pl.Int64),
                pl.col("is_buyer_maker").cast(pl.Boolean),
            ])

            # Add timestamp and amount columns
            df = df.with_columns([
                pl.col("transact_time").cast(pl.Datetime("ms")).alias("timestamp"),
                (pl.col("price") * pl.col("quantity")).alias("amount")
            ])

    return df


def merge_to_monthly_parquet(input_dir, output_dir, incremental: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    files = list(input_path.glob("*.zip"))
    groups = {}
    for f in files:
        month_key = "-".join(f.name.split("-")[2:4])
        groups.setdefault(month_key, []).append(f)

    for month, fs in groups.items():
        target_file = output_path / f"BTCUSDT_{month}.parquet"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if incremental and target_file.exists():
                print(f"正在回补 {month} ...")
                existing_df = pl.read_parquet(target_file)
                existing_days = set(
                    str(d)
                    for d in existing_df["timestamp"].dt.date().unique().to_list()
                )

                new_dfs = []
                # Filter files first
                files_to_load = []
                for f_zip in sorted(fs):
                    parts = f_zip.name.replace(".zip", "").split("-")
                    zip_date = f"{parts[2]}-{parts[3]}-{parts[4]}"
                    if zip_date not in existing_days:
                        files_to_load.append(f_zip)

                if files_to_load:
                    new_dfs = list(executor.map(load_zip_as_df, files_to_load))

                if new_dfs:
                    new_df = pl.concat(new_dfs)
                    combined = (
                        pl.concat([existing_df, new_df])
                        .unique(subset=["timestamp"], keep="first")
                        .sort("timestamp")
                    )
                    combined.write_parquet(target_file, compression="zstd")
                    print(f"  新增 {len(new_dfs)} 天, 总计 {len(combined)} 行")
                else:
                    print(f"  无新数据，跳过")
            else:
                print(f"正在整合 {month} ...")
                month_dfs = list(executor.map(load_zip_as_df, sorted(fs)))
                if month_dfs:
                    monthly_df = pl.concat(month_dfs).sort("timestamp")
                    monthly_df.write_parquet(target_file, compression="zstd")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--incremental":
        merge_to_monthly_parquet(
            "data/BTCUSDT/aggTrades", "data/BTCUSDT/parquet_db", incremental=True
        )
    else:
        merge_to_monthly_parquet("data/BTCUSDT/aggTrades", "data/BTCUSDT/parquet_db")
    
    print("\nRunning data continuity check...")
    from check_data_continuity import visualize_continuity
    visualize_continuity("data/BTCUSDT/parquet_db")
