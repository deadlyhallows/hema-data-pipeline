"""Tests for Silver transformation job — PySpark."""

import os
from pyspark.sql import types as T


def _read_parquet(spark, path):
    return spark.read.parquet(path)


def test_silver_creates_parquet(spark, tmp_silver_path):
    df = _read_parquet(spark, tmp_silver_path)
    assert df.count() > 0


def test_silver_column_rename(spark, tmp_silver_path):
    df = _read_parquet(spark, tmp_silver_path)
    assert "order_id" in df.columns
    assert "Order ID" not in df.columns
    assert "customer_id" in df.columns
    assert "ship_mode" in df.columns


def test_silver_order_date_typed(spark, tmp_silver_path):
    df = _read_parquet(spark, tmp_silver_path)
    field = next(f for f in df.schema.fields if f.name == "order_date")
    assert isinstance(field.dataType, T.DateType), \
        f"Expected DateType for order_date, got {field.dataType}"


def test_silver_no_critical_nulls(spark, tmp_silver_path):
    """Silver output must contain no nulls in critical key columns."""
    df = _read_parquet(spark, tmp_silver_path)
    for col in ("order_id", "customer_id", "order_date"):
        null_count = df.filter(df[col].isNull()).count()
        assert null_count == 0, f"Unexpected nulls in {col}: {null_count}"


def test_silver_quarantine_invalid_rows(spark, tmp_path, raw_csv_path):
    """
    Rows with null critical keys must be written to the quarantine path,
    not silently dropped, and must NOT appear in the Silver output.
    """
    from pyspark.sql import functions as SparkF

    raw = spark.read.option("header", "true").option("inferSchema", "false").csv(raw_csv_path)
    # Inject one row with a null order_id
    bad_row = raw.limit(1).withColumn("Order ID", SparkF.lit(None))
    combined = raw.union(bad_row)

    combined_csv = str(tmp_path / "with_nulls.csv")
    combined.coalesce(1).write.option("header", "true").mode("overwrite").csv(combined_csv)

    from src.bronze.ingest import run as bronze_run
    bronze_out = str(tmp_path / "bronze_nulls")
    bronze_run(input_path=combined_csv, output_path=bronze_out)

    from src.silver.transform import run as silver_run
    silver_out = str(tmp_path / "silver_nulls")
    quarantine_out = str(tmp_path / "quarantine")
    silver_run(input_path=bronze_out, output_path=silver_out,
               quarantine_path=quarantine_out)

    # Silver must not contain the bad row
    silver_df = spark.read.parquet(silver_out)
    assert silver_df.filter(SparkF.col("order_id").isNull()).count() == 0

    # Quarantine must contain the bad row with _dq_failed_checks populated
    quarantine_df = spark.read.parquet(quarantine_out)
    assert quarantine_df.count() == 1
    assert "_dq_failed_checks" in quarantine_df.columns
    assert "_dq_quarantined_at" in quarantine_df.columns


def test_silver_deduplication(spark, tmp_path, raw_csv_path):
    """
    Silver dedup grain is (order_id, product_id).
    Doubling the source CSV produces exact duplicates per line item —
    Silver should collapse them back to the original row count.
    """
    raw = spark.read.option("header","true").option("inferSchema","false").csv(raw_csv_path)
    doubled = raw.union(raw)

    doubled_csv = str(tmp_path / "doubled.csv")
    doubled.coalesce(1).write.option("header","true").mode("overwrite").csv(doubled_csv)

    from src.bronze.ingest import run as bronze_run
    bronze_out = str(tmp_path / "bronze_dup")
    bronze_run(input_path=doubled_csv, output_path=bronze_out)

    from src.silver.transform import run as silver_run
    silver_out = str(tmp_path / "silver_dup")
    silver_run(input_path=bronze_out, output_path=silver_out)

    result = spark.read.parquet(silver_out)
    original_count = raw.count()
    assert result.count() == original_count, \
        f"Expected {original_count} rows after (order_id, product_id) dedup, got {result.count()}"