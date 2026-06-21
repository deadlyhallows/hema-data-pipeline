"""
Gold Layer — Customer Dataset Job (PySpark)
============================================
Reads Silver layer and produces the `gold_customer` domain dataset.

Output columns (per spec):
  - customer_id
  - customer_first_name
  - customer_last_name
  - segment
  - country
  - orders_last_30_days   (distinct orders in 30 days before reference date)
  - orders_last_6_months  (distinct orders in 6 months before reference date)
  - orders_all_time       (all distinct orders ever)

Reference date: 2018-12-30 (latest date in the Superstore dataset).

Why PySpark over Pandas?
  The Customer Gold job aggregates order history across the entire Silver
  dataset — potentially years of data at millions of rows. Computing rolling
  window counts (last 30 days, last 6 months) in Pandas requires the full
  dataset in memory. PySpark executes these as distributed aggregations with
  predicate pushdown, so the job scales linearly with cluster size rather than
  being constrained by the memory of a single driver node. This also makes
  incremental daily runs cheap: only the new Silver partitions need to be
  scanned to refresh the counts.

AWS Glue entry point: aws glue start-job-run --job-name hema-gold-customer
Job parameters: --input_path, --output_path, --reference_date,
                --glue_database, --glue_table
"""

import argparse
import os
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.spark_session import get_spark
from src.utils.glue_catalog import register_or_update_table, add_partitions

logger = get_logger(__name__)

DEFAULT_REFERENCE_DATE = "2018-12-30"


def split_customer_name(df: DataFrame) -> DataFrame:
    """
    Split customer_name → customer_first_name + customer_last_name.
    First token = first name; remainder = last name.
    """
    return (
        df
        .withColumn("customer_first_name",
                    F.trim(F.element_at(F.split(F.col("customer_name"), " "), 1)))
        .withColumn("customer_last_name",
                    F.trim(
                        F.expr("substring(customer_name, length(split(customer_name,' ')[0]) + 2)")
                    ))
    )


def compute_order_aggregations(df: DataFrame, reference_date: str) -> DataFrame:
    """
    Per-customer order count aggregations anchored to reference_date.

    Orders are deduplicated at (customer_id, order_id) before counting
    because the source has one row per line item, not per order.

    Windows:
      last_30_days  : order_date in (ref - 30 days, ref]
      last_6_months : order_date in (ref - 6 months, ref]
      all_time      : any order_date
    """
    ref = F.to_date(F.lit(reference_date))
    cutoff_30d = F.date_sub(ref, 30)
    cutoff_6m  = F.add_months(ref, -6)

    logger.info("Computing order aggregations",
                extra={
                    "reference_date": reference_date,
                    "cutoff_30d": str(cutoff_30d),
                    "cutoff_6m": str(cutoff_6m),
                })

    # One row per (customer_id, order_id) — eliminate line-item duplication
    orders = df.select("customer_id", "order_id", "order_date").dropDuplicates()

    agg = (
        orders.groupBy("customer_id")
        .agg(
            F.sum(
                F.when(
                    (F.col("order_date") > cutoff_30d) & (F.col("order_date") <= ref),
                    1
                ).otherwise(0)
            ).cast(IntegerType()).alias("orders_last_30_days"),

            F.sum(
                F.when(
                    (F.col("order_date") > cutoff_6m) & (F.col("order_date") <= ref),
                    1
                ).otherwise(0)
            ).cast(IntegerType()).alias("orders_last_6_months"),

            F.countDistinct("order_id").cast(IntegerType()).alias("orders_all_time"),
        )
    )

    logger.info("Order aggregations computed",
                extra={"unique_customers": agg.count()})
    return agg


def build_customer_dataset(df: DataFrame, reference_date: str) -> DataFrame:
    """
    Build the Gold Customer dataset.

    Steps:
    1. Split full name → first / last
    2. Deduplicate customer attributes (latest record per customer_id)
    3. Compute order aggregations
    4. Join attributes with aggregations
    5. Add static partition columns from reference date
    """
    required = ["customer_id", "customer_name", "segment", "country",
                "order_id", "order_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Silver columns for Gold Customer: {missing}")

    df = split_customer_name(df)

    # Latest record per customer_id for dimension attributes
    from pyspark.sql.window import Window
    w = Window.partitionBy("customer_id").orderBy(F.col("order_date").desc())
    customer_attrs = (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select("customer_id", "customer_first_name", "customer_last_name",
                "segment", "country")
    )
    logger.info("Customer attributes deduplicated",
                extra={"unique_customers": customer_attrs.count()})

    agg = compute_order_aggregations(df, reference_date)

    customer = customer_attrs.join(agg, on="customer_id", how="left")

    # Static partition columns = reference date
    ref_ts = F.to_date(F.lit(reference_date))
    customer = (
        customer
        .withColumn("year",  F.date_format(ref_ts, "yyyy"))
        .withColumn("month", F.date_format(ref_ts, "MM"))
        .withColumn("day",   F.date_format(ref_ts, "dd"))
    )

    logger.info("Customer dataset built", extra={"rows": customer.count()})
    return customer


def write_gold_customer(df: DataFrame, output_path: str) -> list[tuple[str, str, str]]:
    logger.info("Writing Gold Customer Parquet", extra={"output_path": output_path})
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "day")
        .parquet(output_path)
    )
    partitions = df.select("year", "month", "day").distinct().collect()
    result = [(r["year"], r["month"], r["day"]) for r in partitions]
    logger.info("Gold Customer write complete",
                extra={"partitions_written": len(result)})
    return result


def run(
    input_path: str,
    output_path: str,
    reference_date: str = DEFAULT_REFERENCE_DATE,
    glue_database: str = "hema_retail",
    glue_table: str = "gold_customer",
) -> None:
    start = time.monotonic()
    logger.info("Gold Customer job started",
                extra={"input_path": input_path, "reference_date": reference_date})

    spark = get_spark("hema-gold-customer")
    df = spark.read.parquet(input_path)
    customer = build_customer_dataset(df, reference_date)
    partitions = write_gold_customer(customer, output_path)

    s3_loc = output_path if output_path.startswith("s3://") \
        else f"s3://hema-data-lake/{glue_table}/"
    register_or_update_table(
        database=glue_database, table_name=glue_table,
        s3_location=s3_loc,
        columns=[{"Name": "placeholder", "Type": "string"}],
        description=(
            f"Customer domain dataset — Gold layer. "
            f"Order aggregations anchored to {reference_date}."
        ),
    )
    for year, month, day in partitions:
        add_partitions(glue_database, glue_table, s3_loc, year, month, day)

    elapsed = time.monotonic() - start
    logger.info("Gold Customer job completed",
                extra={"duration_seconds": round(elapsed, 2)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEMA Gold Customer Job")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--reference_date", default=DEFAULT_REFERENCE_DATE)
    parser.add_argument("--glue_database", default="hema_retail")
    parser.add_argument("--glue_table", default="gold_customer")
    args = parser.parse_args()

    os.environ.setdefault("HEMA_LOCAL_MODE", "true")
    run(args.input_path, args.output_path, args.reference_date,
        args.glue_database, args.glue_table)
