"""
Gold Layer — Sales Dataset Job (PySpark)
=========================================
Reads Silver layer and produces the `gold_sales` domain dataset.

Output columns (per spec):
  - order_id
  - order_date      (Order Date)
  - ship_date       (Shipment Date)
  - ship_mode       (Shipment Mode)
  - city

Why PySpark over Pandas?
  Gold jobs run aggregations over the full Silver history, not just today's
  slice. As the Silver layer grows across years of partitions, a Pandas job
  would need to load the entire dataset into memory. PySpark reads only the
  partitions needed, pushes predicates down to the Parquet layer, and
  distributes the deduplication across the cluster — no memory ceiling.

AWS Glue entry point: aws glue start-job-run --job-name hema-gold-sales
Job parameters: --input_path, --output_path, --glue_database, --glue_table
"""

import argparse
import os
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.spark_session import get_spark
from src.utils.glue_catalog import register_or_update_table, add_partitions

logger = get_logger(__name__)

SALES_COLS = ["order_id", "order_date", "ship_date", "ship_mode", "city",
              "year", "month", "day"]


def build_sales_dataset(df):
    missing = [c for c in SALES_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Silver columns for Gold Sales: {missing}")

    # Gold Sales grain: one row per order_id.
    # Silver contains one row per line item, so we collapse using groupBy.
    # order_date, ship_date, ship_mode and city are constant across all line
    # items of the same order — we take max() which is equivalent to first()
    # for constant columns but is a stable, deterministic aggregation.
    # year/month/day are derived from order_date so they are equally constant.
    sales = (
        df.groupBy("order_id")
        .agg(
            F.max("order_date").alias("order_date"),
            F.max("ship_date").alias("ship_date"),
            F.max("ship_mode").alias("ship_mode"),
            F.max("city").alias("city"),
            F.max("year").alias("year"),
            F.max("month").alias("month"),
            F.max("day").alias("day"),
        )
    )
    logger.info("Sales dataset built", extra={"rows": sales.count()})
    return sales


def write_gold(df, output_path: str) -> list[tuple[str, str, str]]:
    logger.info("Writing Gold Sales Parquet", extra={"output_path": output_path})
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "day")
        .parquet(output_path)
    )
    partitions = df.select("year", "month", "day").distinct().collect()
    result = [(r["year"], r["month"], r["day"]) for r in partitions]
    logger.info("Gold Sales write complete", extra={"partitions_written": len(result)})
    return result


def run(
    input_path: str,
    output_path: str,
    glue_database: str = "hema_retail",
    glue_table: str = "gold_sales",
) -> None:
    start = time.monotonic()
    logger.info("Gold Sales job started",
                extra={"input_path": input_path, "output_path": output_path})

    spark = get_spark("hema-gold-sales")
    df = spark.read.parquet(input_path)
    sales = build_sales_dataset(df)
    partitions = write_gold(sales, output_path)

    s3_loc = output_path if output_path.startswith("s3://") \
        else f"s3://hema-data-lake/{glue_table}/"
    register_or_update_table(
        database=glue_database, table_name=glue_table,
        s3_location=s3_loc,
        columns=[{"Name": "placeholder", "Type": "string"}],
        description="Sales domain dataset — Gold layer. Order and shipment attributes.",
    )
    for year, month, day in partitions:
        add_partitions(glue_database, glue_table, s3_loc, year, month, day)

    elapsed = time.monotonic() - start
    logger.info("Gold Sales job completed",
                extra={"duration_seconds": round(elapsed, 2)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEMA Gold Sales Job")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--glue_database", default="hema_retail")
    parser.add_argument("--glue_table", default="gold_sales")
    args = parser.parse_args()

    os.environ.setdefault("HEMA_LOCAL_MODE", "true")
    run(args.input_path, args.output_path, args.glue_database, args.glue_table)