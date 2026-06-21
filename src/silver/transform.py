"""
Silver Layer — Cleanse & Standardise Job (PySpark)
===================================================
Reads Bronze Parquet, applies:
  - Column renaming to snake_case
  - Type casting (dates, numerics)
  - Null enforcement on critical key columns
  - Deduplication
  - Pass-through of any new/unknown columns (schema evolution)

Why PySpark over Pandas?
  Silver sits between raw ingestion and business-logic Gold jobs. In production,
  the Bronze layer may accumulate billions of rows across many partitions.
  PySpark's distributed execution and predicate pushdown mean Silver can process
  only the new partitions (via Glue job bookmarks or partition filters) without
  ever loading the full dataset into memory on a single node.

AWS Glue entry point: aws glue start-job-run --job-name hema-silver-transform
Job parameters: --input_path, --output_path, --glue_database, --glue_table
"""

import argparse
import os
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.spark_session import get_spark
from src.utils.glue_catalog import register_or_update_table, add_partitions

logger = get_logger(__name__)

COLUMN_RENAME: dict[str, str] = {
    "Row ID":        "row_id",
    "Order ID":      "order_id",
    "Order Date":    "order_date",
    "Ship Date":     "ship_date",
    "Ship Mode":     "ship_mode",
    "Customer ID":   "customer_id",
    "Customer Name": "customer_name",
    "Segment":       "segment",
    "Country":       "country",
    "City":          "city",
    "State":         "state",
    "Postal Code":   "postal_code",
    "Region":        "region",
    "Product ID":    "product_id",
    "Category":      "category",
    "Sub-Category":  "sub_category",
    "Product Name":  "product_name",
    "Sales":         "sales",
    "Quantity":      "quantity",
    "Discount":      "discount",
    "Profit":        "profit",
}

CRITICAL_NOT_NULL = ["order_id", "customer_id", "order_date"]


def rename_columns(df: DataFrame) -> DataFrame:
    """
    Rename known columns to snake_case.
    Unknown columns (schema evolution) are auto-snake_cased and preserved.
    """
    known = {k for k in COLUMN_RENAME if k in df.columns}
    unknown = [c for c in df.columns if c not in known]

    for old, new in COLUMN_RENAME.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # Auto snake_case unknown columns
    for col in unknown:
        auto = col.lower().replace(" ", "_").replace("-", "_")
        if auto != col:
            logger.info("Auto-renaming unknown column (schema evolution)",
                        extra={"from": col, "to": auto})
            df = df.withColumnRenamed(col, auto)

    return df


def cast_types(df: DataFrame) -> DataFrame:
    """Cast columns to their canonical Silver types."""
    date_fmt = F.coalesce(
        F.to_date(F.col("order_date"), "M/d/yyyy"),
        F.to_date(F.col("order_date"), "MM/dd/yyyy"),
        F.to_date(F.col("order_date"), "yyyy-MM-dd"),
    )
    ship_fmt = F.coalesce(
        F.to_date(F.col("ship_date"), "M/d/yyyy"),
        F.to_date(F.col("ship_date"), "MM/dd/yyyy"),
        F.to_date(F.col("ship_date"), "yyyy-MM-dd"),
    )

    df = (
        df
        .withColumn("order_date", date_fmt)
        .withColumn("ship_date",  ship_fmt)
        .withColumn("row_id",     F.col("row_id").cast(IntegerType()))
        .withColumn("quantity",   F.col("quantity").cast(IntegerType()))
        .withColumn("sales",      F.col("sales").cast(DoubleType()))
        .withColumn("discount",   F.col("discount").cast(DoubleType()))
        .withColumn("profit",     F.col("profit").cast(DoubleType()))
    )

    # Log null counts for dates after cast
    for col in ("order_date", "ship_date"):
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count:
            logger.warning("Null values after date cast",
                           extra={"column": col, "null_count": null_count})
    return df


def deduplicate(df: DataFrame) -> DataFrame:
    """
    Deduplicate to the Silver grain: one row per (order_id, product_id).

    The Superstore source has one row per line item, so the same order_id
    appearing multiple times is expected and correct — each row is a different
    product on that order. A true duplicate is the same product on the same
    order appearing more than once (e.g. from a double-ingestion or reprocessing).

    We use row_number() over a window partitioned by (order_id, product_id),
    ordered by _ingested_at descending, so the most recently ingested copy wins.
    This is safer than dropDuplicates(), which picks a winner arbitrarily.
    """
    from pyspark.sql.window import Window

    dedup_key = ["order_id", "product_id"]
    missing_key_cols = [c for c in dedup_key if c not in df.columns]
    if missing_key_cols:
        logger.warning(
            "Dedup key columns missing — falling back to full-row dedup",
            extra={"missing": missing_key_cols},
        )
        before = df.count()
        df = df.dropDuplicates()
        logger.warning("Fallback dedup removed rows", extra={"count": before - df.count()})
        return df

    before = df.count()

    order_col = F.col("_ingested_at") if "_ingested_at" in df.columns else F.lit(1)
    w = Window.partitionBy(*dedup_key).orderBy(order_col.desc())

    df = (
        df.withColumn("_rn", F.row_number().over(w))
          .filter(F.col("_rn") == 1)
          .drop("_rn")
    )

    removed = before - df.count()
    if removed:
        logger.warning(
            "Duplicate line items removed (grain: order_id + product_id, kept latest ingested)",
            extra={"count": removed},
        )
    else:
        logger.info("No duplicates found at (order_id, product_id) grain")
    return df


def enforce_not_null(
    df: DataFrame,
    quarantine_path: str | None = None,
) -> DataFrame:
    """
    Quarantine rows where critical key columns are null rather than silently
    dropping them.

    Why quarantine instead of drop?
      Dropping invalid rows is an extreme and silent measure — in production,
      nobody knows which rows were lost, making debugging upstream issues very
      difficult. Quarantining routes invalid rows to a separate S3 path where
      they remain visible, auditable, and reprocessable once the upstream data
      quality issue is fixed.

    Invalid rows are:
      1. Written to `quarantine_path` as Parquet with an extra `_dq_failed_checks`
         column listing which key(s) were null.
      2. Logged with a count and a sample of the failing rows.
      3. Excluded from the Silver output (Silver stays a trusted layer).

    If `quarantine_path` is None the path defaults to a sibling `_quarantine/`
    directory next to the Silver output, or simply logs without writing when
    running in local mode with no path configured.
    """
    from datetime import datetime, timezone

    existing = [c for c in CRITICAL_NOT_NULL if c in df.columns]

    # Build a null-check condition: row is invalid if ANY critical key is null
    null_condition = F.lit(False)
    for col in existing:
        null_condition = null_condition | F.col(col).isNull()

    valid_df   = df.filter(~null_condition)
    invalid_df = df.filter(null_condition)

    invalid_count = invalid_df.count()

    if invalid_count == 0:
        logger.info("Data quality check passed — no null critical keys",
                    extra={"checked_columns": existing})
        return valid_df

    # Build a human-readable list of which checks failed per row
    failed_checks_col = F.concat_ws(
        ", ",
        *[
            F.when(F.col(c).isNull(), F.lit(f"{c} is null"))
            for c in existing
        ],
    )
    invalid_df = invalid_df.withColumn("_dq_failed_checks", failed_checks_col)
    invalid_df = invalid_df.withColumn(
        "_dq_quarantined_at",
        F.lit(datetime.now(timezone.utc).isoformat()),
    )

    # Log a sample so engineers can see the problem without opening S3
    sample = invalid_df.select("_dq_failed_checks", *existing).limit(5).collect()
    logger.warning(
        "Data quality: rows quarantined due to null critical keys",
        extra={
            "quarantine_count": invalid_count,
            "valid_count": valid_df.count(),
            "checked_columns": existing,
            "sample_failures": [row.asDict() for row in sample],
            "quarantine_path": quarantine_path or "not configured — rows counted but not persisted",
        },
    )

    # Persist to quarantine if a path is provided
    if quarantine_path:
        (
            invalid_df.write
            .mode("append")   # append so multiple runs accumulate — quarantine is a log
            .parquet(quarantine_path)
        )
        logger.info("Quarantined rows written", extra={"path": quarantine_path})

    return valid_df


def write_silver(df: DataFrame, output_path: str) -> list[tuple[str, str, str]]:
    """Write Silver Parquet partitioned by year/month/day."""
    # Re-derive partition cols from cast order_date if not present
    if "year" not in df.columns:
        df = (
            df
            .withColumn("year",  F.date_format("order_date", "yyyy"))
            .withColumn("month", F.date_format("order_date", "MM"))
            .withColumn("day",   F.date_format("order_date", "dd"))
        )

    logger.info("Writing Silver Parquet", extra={"output_path": output_path})
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "day")
        .parquet(output_path)
    )
    partitions = (
        df.select("year", "month", "day").distinct().collect()
    )
    result = [(r["year"], r["month"], r["day"]) for r in partitions]
    logger.info("Silver write complete", extra={"partitions_written": len(result)})
    return result


def run(
    input_path: str,
    output_path: str,
    quarantine_path: str | None = None,
    glue_database: str = "hema_retail",
    glue_table: str = "silver_retail_sales",
) -> None:
    start = time.monotonic()
    logger.info("Silver transform job started",
                extra={"input_path": input_path, "output_path": output_path,
                       "quarantine_path": quarantine_path or "not configured"})

    spark = get_spark("hema-silver-transform")

    df = spark.read.parquet(input_path)
    df = rename_columns(df)
    df = cast_types(df)
    df = deduplicate(df)
    df = enforce_not_null(df, quarantine_path=quarantine_path)

    row_count = df.count()
    partitions = write_silver(df, output_path)

    s3_loc = output_path if output_path.startswith("s3://") \
        else f"s3://hema-data-lake/{glue_table}/"
    register_or_update_table(
        database=glue_database,
        table_name=glue_table,
        s3_location=s3_loc,
        columns=[{"Name": "placeholder", "Type": "string"}],
        description="Cleansed and typed Superstore sales data — Silver layer.",
    )
    for year, month, day in partitions:
        add_partitions(glue_database, glue_table, s3_loc, year, month, day)

    elapsed = time.monotonic() - start
    logger.info("Silver transform job completed",
                extra={
                    "total_rows": row_count,
                    "partitions_written": len(partitions),
                    "duration_seconds": round(elapsed, 2),
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEMA Silver Transform Job")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--quarantine_path", default=None,
                        help="S3/local path to write quarantined invalid rows (optional)")
    parser.add_argument("--glue_database", default="hema_retail")
    parser.add_argument("--glue_table", default="silver_retail_sales")
    args = parser.parse_args()

    os.environ.setdefault("HEMA_LOCAL_MODE", "true")
    run(args.input_path, args.output_path, args.quarantine_path,
        args.glue_database, args.glue_table)