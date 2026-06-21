"""Shared pytest fixtures for HEMA pipeline PySpark tests."""

import os
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)
from datetime import date


@pytest.fixture(scope="session")
def spark():
    """Single SparkSession shared across all tests in the session."""
    os.environ["HEMA_LOCAL_MODE"] = "true"
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("hema-test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.setenv("HEMA_LOCAL_MODE", "true")


@pytest.fixture
def raw_csv_path(tmp_path):
    """Write a minimal Superstore CSV and return its path."""
    csv = tmp_path / "superstore.csv"
    csv.write_text(
        "Row ID,Order ID,Order Date,Ship Date,Ship Mode,"
        "Customer ID,Customer Name,Segment,Country,City,State,"
        "Postal Code,Region,Product ID,Category,Sub-Category,"
        "Product Name,Sales,Quantity,Discount,Profit\n"
        "1,CA-2018-152156,11/8/2018,11/11/2018,Second Class,"
        "CG-12520,Claire Gute,Consumer,United States,Henderson,Kentucky,"
        "42420,South,FUR-BO-10001798,Furniture,Bookcases,"
        "Bush Somerset,261.96,2,0,41.9136\n"
        "2,CA-2018-152156,11/8/2018,11/11/2018,Second Class,"
        "CG-12520,Claire Gute,Consumer,United States,Henderson,Kentucky,"
        "42420,South,FUR-CH-10000454,Furniture,Chairs,"
        "Hon Deluxe,731.94,3,0,219.582\n"
        "3,CA-2018-138688,6/12/2018,6/16/2018,Second Class,"
        "DV-13045,Darrin Van Huff,Corporate,United States,Los Angeles,California,"
        "90036,West,OFF-LA-10000240,Office Supplies,Labels,"
        "Self-Adhesive,14.62,2,0,6.8714\n"
        "4,US-2018-108966,10/11/2018,10/18/2018,Standard Class,"
        "SO-20335,Sean O'Donnell,Consumer,United States,Fort Lauderdale,Florida,"
        "33311,South,FUR-TA-10000577,Furniture,Tables,"
        "Bretford CR4500,957.58,5,0.45,-383.032\n"
        "5,US-2018-108966,10/11/2018,10/18/2018,Standard Class,"
        "SO-20335,Sean O'Donnell,Consumer,United States,Fort Lauderdale,Florida,"
        "33311,South,OFF-ST-10000760,Office Supplies,Storage,"
        "Eldon Fold,22.37,2,0.2,2.5164\n",
        encoding="utf-8",
    )
    return str(csv)


@pytest.fixture
def silver_df(spark):
    """Minimal Silver-layer DataFrame (typed, snake_case columns)."""
    schema = StructType([
        StructField("order_id",       StringType(),  True),
        StructField("order_date",     DateType(),    True),
        StructField("ship_date",      DateType(),    True),
        StructField("ship_mode",      StringType(),  True),
        StructField("customer_id",    StringType(),  True),
        StructField("customer_name",  StringType(),  True),
        StructField("segment",        StringType(),  True),
        StructField("country",        StringType(),  True),
        StructField("city",           StringType(),  True),
        StructField("year",           StringType(),  True),
        StructField("month",          StringType(),  True),
        StructField("day",            StringType(),  True),
    ])
    rows = [
        ("CA-2018-152156", date(2018,11,8),  date(2018,11,11), "Second Class",   "CG-12520", "Claire Gute",     "Consumer",  "United States", "Henderson",       "2018","11","08"),
        ("CA-2018-152156", date(2018,11,8),  date(2018,11,11), "Second Class",   "CG-12520", "Claire Gute",     "Consumer",  "United States", "Henderson",       "2018","11","08"),
        ("CA-2018-138688", date(2018,6,12),  date(2018,6,16),  "Second Class",   "DV-13045", "Darrin Van Huff", "Corporate", "United States", "Los Angeles",     "2018","06","12"),
        ("US-2018-108966", date(2018,10,11), date(2018,10,18), "Standard Class", "SO-20335", "Sean O'Donnell",  "Consumer",  "United States", "Fort Lauderdale", "2018","10","11"),
        ("US-2018-108966", date(2018,10,11), date(2018,10,18), "Standard Class", "SO-20335", "Sean O'Donnell",  "Consumer",  "United States", "Fort Lauderdale", "2018","10","11"),
    ]
    return spark.createDataFrame(rows, schema)


@pytest.fixture
def tmp_bronze_path(spark, raw_csv_path, tmp_path):
    """Run Bronze ingestion and return output path."""
    from src.bronze.ingest import run as bronze_run
    out = str(tmp_path / "bronze")
    bronze_run(input_path=raw_csv_path, output_path=out)
    return out


@pytest.fixture
def tmp_silver_path(spark, tmp_bronze_path, tmp_path):
    """Run Silver transform on Bronze output and return path."""
    from src.silver.transform import run as silver_run
    out = str(tmp_path / "silver")
    silver_run(input_path=tmp_bronze_path, output_path=out)
    return out
