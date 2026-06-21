"""
SparkSession factory for HEMA ETL pipeline.

In AWS Glue the session is created by the Glue runtime and passed in via
GlueContext — this module provides a local-dev fallback so every job can
call get_spark() without branching.
"""

import os
from pyspark.sql import SparkSession


def get_spark(app_name: str = "hema-etl") -> SparkSession:
    """
    Return a SparkSession.

    - On AWS Glue: the runtime already has an active session; getOrCreate()
      returns it with no overhead.
    - Locally (HEMA_LOCAL_MODE=true): builds a minimal local[*] session with
      Parquet / Hive support enabled.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if os.environ.get("HEMA_LOCAL_MODE", "false").lower() == "true":
        builder = builder.master("local[*]")

    return builder.getOrCreate()
