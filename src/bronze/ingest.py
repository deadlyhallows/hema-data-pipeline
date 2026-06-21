"""
Bronze Layer — Raw Ingestion Job (PySpark)
==========================================
Reads the raw Superstore CSV from S3 (or local path) using PySpark,
applies minimal schema validation, adds partition + audit columns, and
writes Parquet to the Bronze S3 prefix.

Why PySpark over Pandas?
  The pipeline currently processes a single CSV file, but in production we
  may receive many large files simultaneously (multiple daily drops, backfills,
  regional feeds). Pandas loads everything into a single JVM process on one
  machine — that becomes a hard wall as data volume grows. PySpark distributes
  the read and write across a cluster, so scaling from 1 file to 1,000 files
  requires no code change, only cluster sizing.

This job is intentionally lightweight: no business logic, no column drops.
Unknown/new columns are preserved to support schema evolution transparently.

AWS Glue entry point: aws glue start-job-run --job-name hema-bronze-ingest
Job parameters: --input_path, --output_path, --glue_database, --glue_table
"""

import argparse
import os
import sys
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.spark_session import get_spark
from src.utils.schema_validator import validate_schema
from src.utils.glue_catalog import register_or_update_table, add_partitions

logger = get_logger(__name__)


def read_raw(spark, input_path: str) -> DataFrame:
    """
    Read raw CSV — all columns as strings (schema-on-read).
    PySpark handles multiple files in a directory transparently.
    """
    logger.info("Reading raw source", extra={"input_path": input_path})
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")   # Keep everything string at Bronze
        .option("encoding", "iso-8859-1")
        .option("multiLine", "false")
        .csv(input_path)
    )
    logger.info("Raw data loaded",
                extra={"columns": df.columns, "partitions": df.rdd.getNumPartitions()})
    return df


def add_audit_columns(df: DataFrame, source_path: str) -> DataFrame:
    """Add _ingested_at and _source_file lineage columns."""
    from datetime import datetime, timezone
    ingested_at = datetime.now(timezone.utc).isoformat()
    return (
        df
        .withColumn("_ingested_at", F.lit(ingested_at))
        .withColumn("_source_file", F.lit(os.path.basename(source_path)))
    )


def add_partition_columns(df: DataFrame) -> DataFrame:
    """
    Derive year / month / day partition columns from Order Date.
    Tries MM/dd/yyyy first (Superstore format), falls back to yyyy-MM-dd.
    Rows where Order Date cannot be parsed get today's date as fallback.
    """
    parsed = F.coalesce(
        F.to_date(F.col("`Order Date`"), "M/d/yyyy"),
        F.to_date(F.col("`Order Date`"), "MM/dd/yyyy"),
        F.to_date(F.col("`Order Date`"), "yyyy-MM-dd"),
        F.current_date(),
    )
    return (
        df
        .withColumn("_parsed_order_date", parsed)
        .withColumn("year",  F.date_format("_parsed_order_date", "yyyy"))
        .withColumn("month", F.date_format("_parsed_order_date", "MM"))
        .withColumn("day",   F.date_format("_parsed_order_date", "dd"))
        .drop("_parsed_order_date")
    )


def write_bronze(df: DataFrame, output_path: str) -> list[tuple[str, str, str]]:
    """
    Write partitioned Parquet to the Bronze layer.
    Returns list of (year, month, day) partitions written.
    """
    logger.info("Writing Bronze Parquet", extra={"output_path": output_path})
    (
        df.write
        .mode("overwrite")
        .partitionBy("year", "month", "day")
        .parquet(output_path)
    )
    # Collect distinct partitions for catalog registration
    partitions = (
        df.select("year", "month", "day")
        .distinct()
        .collect()
    )
    result = [(r["year"], r["month"], r["day"]) for r in partitions]
    logger.info("Bronze write complete",
                extra={"partitions_written": len(result)})
    return result


def run(
    input_path: str,
    output_path: str,
    glue_database: str = "hema_retail",
    glue_table: str = "bronze_retail_sales",
) -> None:
    start = time.monotonic()
    logger.info("Bronze ingestion job started",
                extra={"input_path": input_path, "output_path": output_path})

    spark = get_spark("hema-bronze-ingest")

    df = read_raw(spark, input_path)
    validate_schema(df, job_name="bronze_ingest")
    df = add_audit_columns(df, source_path=input_path)
    df = add_partition_columns(df)

    row_count = df.count()
    partitions = write_bronze(df, output_path)

    s3_loc = output_path if output_path.startswith("s3://") \
        else f"s3://hema-data-lake/{glue_table}/"
    register_or_update_table(
        database=glue_database,
        table_name=glue_table,
        s3_location=s3_loc,
        columns=[{"Name": "placeholder", "Type": "string"}],
        description=(
            "Raw Superstore sales data — Bronze layer. "
            "All columns stored as strings. Schema evolution: new columns passed through."
        ),
    )
    for year, month, day in partitions:
        add_partitions(glue_database, glue_table, s3_loc, year, month, day)

    elapsed = time.monotonic() - start
    logger.info("Bronze ingestion job completed",
                extra={
                    "total_rows": row_count,
                    "partitions_written": len(partitions),
                    "duration_seconds": round(elapsed, 2),
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEMA Bronze Ingestion Job")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--glue_database", default="hema_retail")
    parser.add_argument("--glue_table", default="bronze_retail_sales")
    args = parser.parse_args()

    os.environ.setdefault("HEMA_LOCAL_MODE", "true")
    run(args.input_path, args.output_path, args.glue_database, args.glue_table)
