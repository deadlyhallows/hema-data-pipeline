"""Tests for Bronze ingestion job — PySpark."""

import os
from pyspark.sql import SparkSession


def _read_parquet(spark, path):
    return spark.read.parquet(path)


def test_bronze_creates_parquet(spark, tmp_bronze_path):
    df = _read_parquet(spark, tmp_bronze_path)
    assert df.count() > 0


def test_bronze_partition_columns_present(spark, tmp_bronze_path):
    df = _read_parquet(spark, tmp_bronze_path)
    for col in ("year", "month", "day"):
        assert col in df.columns, f"Missing partition column: {col}"


def test_bronze_preserves_source_columns(spark, tmp_bronze_path):
    df = _read_parquet(spark, tmp_bronze_path)
    expected = {"Order ID", "Order Date", "Ship Date", "Customer ID",
                "Customer Name", "Segment", "Country", "City"}
    for col in expected:
        assert col in df.columns, f"Source column dropped: {col}"


def test_bronze_audit_columns_added(spark, tmp_bronze_path):
    df = _read_parquet(spark, tmp_bronze_path)
    assert "_ingested_at" in df.columns
    assert "_source_file" in df.columns


def test_bronze_schema_evolution_new_column(spark, tmp_path, raw_csv_path):
    """New columns in source must pass through Bronze without being dropped."""
    from pyspark.sql import functions as F
    raw_df = spark.read.option("header","true").option("inferSchema","false").csv(raw_csv_path)
    evolved_df = raw_df.withColumn("New Column From Future", F.lit("some_value"))

    evolved_csv = str(tmp_path / "evolved.csv")
    evolved_df.coalesce(1).write.option("header","true").mode("overwrite").csv(evolved_csv)

    from src.bronze.ingest import run as bronze_run
    out = str(tmp_path / "bronze_evolved")
    bronze_run(input_path=evolved_csv, output_path=out)

    result = spark.read.parquet(out)
    assert "New Column From Future" in result.columns
