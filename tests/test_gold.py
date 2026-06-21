"""Tests for Gold layer jobs — PySpark."""

import os
from datetime import date
from pyspark.sql import functions as F


def _read_parquet(spark, path):
    return spark.read.parquet(path)


# ── Gold Sales ──────────────────────────────────────────────

def test_gold_sales_creates_parquet(spark, tmp_silver_path, tmp_path):
    from src.gold.sales import run
    out = str(tmp_path / "gold_sales")
    run(input_path=tmp_silver_path, output_path=out)
    assert _read_parquet(spark, out).count() > 0


def test_gold_sales_columns(spark, tmp_silver_path, tmp_path):
    from src.gold.sales import run
    out = str(tmp_path / "gold_sales_cols")
    run(input_path=tmp_silver_path, output_path=out)
    df = _read_parquet(spark, out)
    for col in ("order_id", "order_date", "ship_date", "ship_mode", "city"):
        assert col in df.columns, f"Missing Gold Sales column: {col}"


def test_gold_sales_no_customer_columns(spark, tmp_silver_path, tmp_path):
    from src.gold.sales import run
    out = str(tmp_path / "gold_sales_nocust")
    run(input_path=tmp_silver_path, output_path=out)
    df = _read_parquet(spark, out)
    for col in ("customer_id", "customer_name", "segment"):
        assert col not in df.columns, f"Customer column leaked into Gold Sales: {col}"


# ── Gold Customer ────────────────────────────────────────────

def test_gold_customer_creates_parquet(spark, tmp_silver_path, tmp_path):
    from src.gold.customer import run
    out = str(tmp_path / "gold_customer")
    run(input_path=tmp_silver_path, output_path=out)
    assert _read_parquet(spark, out).count() > 0


def test_gold_customer_columns(spark, tmp_silver_path, tmp_path):
    from src.gold.customer import run
    out = str(tmp_path / "gold_customer_cols")
    run(input_path=tmp_silver_path, output_path=out)
    df = _read_parquet(spark, out)
    expected = {
        "customer_id", "customer_first_name", "customer_last_name",
        "segment", "country",
        "orders_last_30_days", "orders_last_6_months", "orders_all_time",
    }
    for col in expected:
        assert col in df.columns, f"Missing Gold Customer column: {col}"


def test_gold_customer_name_split(silver_df):
    from src.gold.customer import split_customer_name
    df = split_customer_name(silver_df)
    row = df.filter(F.col("customer_id") == "CG-12520").first()
    assert row["customer_first_name"] == "Claire"
    assert row["customer_last_name"] == "Gute"

    row2 = df.filter(F.col("customer_id") == "DV-13045").first()
    assert row2["customer_first_name"] == "Darrin"
    assert row2["customer_last_name"] == "Van Huff"


def test_gold_customer_order_aggregations(silver_df):
    """
    Reference date: 2018-12-30
      CG-12520  order 2018-11-08 → in 6m (cutoff 2018-06-30), NOT in 30d (cutoff 2018-11-30)
      DV-13045  order 2018-06-12 → NOT in 6m (before cutoff), NOT in 30d
      SO-20335  order 2018-10-11 → in 6m, NOT in 30d
    """
    from src.gold.customer import compute_order_aggregations
    agg = compute_order_aggregations(silver_df, "2018-12-30")

    def get(customer_id, col):
        return agg.filter(F.col("customer_id") == customer_id).first()[col]

    assert get("CG-12520", "orders_last_30_days")  == 0
    assert get("CG-12520", "orders_last_6_months") == 1
    assert get("CG-12520", "orders_all_time")      == 1

    assert get("DV-13045", "orders_last_6_months") == 0
    assert get("DV-13045", "orders_all_time")      == 1

    assert get("SO-20335", "orders_last_6_months") == 1


def test_gold_customer_one_row_per_customer(spark, tmp_silver_path, tmp_path):
    from src.gold.customer import run
    out = str(tmp_path / "gold_customer_unique")
    run(input_path=tmp_silver_path, output_path=out)
    df = _read_parquet(spark, out)
    total = df.count()
    distinct = df.select("customer_id").distinct().count()
    assert total == distinct, "Gold Customer must have exactly one row per customer_id"


def test_gold_customer_no_order_columns(spark, tmp_silver_path, tmp_path):
    from src.gold.customer import run
    out = str(tmp_path / "gold_customer_noorder")
    run(input_path=tmp_silver_path, output_path=out)
    df = _read_parquet(spark, out)
    for col in ("order_id", "ship_date", "ship_mode", "city"):
        assert col not in df.columns, f"Order column leaked into Gold Customer: {col}"
