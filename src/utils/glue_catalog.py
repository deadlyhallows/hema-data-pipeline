"""
AWS Glue Data Catalog helpers.

Wraps boto3 Glue client calls for table creation, schema updates,
and partition registration. In local/test mode (no AWS credentials),
operations are logged but not executed.
"""

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.utils.logger import get_logger

logger = get_logger(__name__)

_LOCAL_MODE = os.environ.get("HEMA_LOCAL_MODE", "false").lower() == "true"


def _glue_client():
    return boto3.client("glue", region_name=os.environ.get("AWS_REGION", "eu-west-1"))


def register_or_update_table(
    database: str,
    table_name: str,
    s3_location: str,
    columns: list[dict[str, str]],
    partition_keys: list[dict[str, str]] | None = None,
    description: str = "",
) -> None:
    """
    Create or update a Glue Data Catalog table definition.

    Args:
        database: Glue database name
        table_name: Table name in catalog
        s3_location: S3 URI prefix (e.g. s3://bucket/prefix/)
        columns: List of {"Name": ..., "Type": ...} dicts
        partition_keys: Partition column definitions
        description: Human-readable table description
    """
    if _LOCAL_MODE:
        logger.info(
            "LOCAL MODE — skipping Glue catalog registration",
            extra={"database": database, "table": table_name, "location": s3_location},
        )
        return

    partition_keys = partition_keys or [
        {"Name": "year", "Type": "string"},
        {"Name": "month", "Type": "string"},
        {"Name": "day", "Type": "string"},
    ]

    table_input: dict[str, Any] = {
        "Name": table_name,
        "Description": description,
        "StorageDescriptor": {
            "Columns": columns,
            "Location": s3_location,
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            },
            "Compressed": True,
        },
        "PartitionKeys": partition_keys,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": "parquet", "compressionType": "snappy"},
    }

    glue = _glue_client()
    try:
        glue.create_table(DatabaseName=database, TableInput=table_input)
        logger.info(
            "Glue table created",
            extra={"database": database, "table": table_name},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            glue.update_table(DatabaseName=database, TableInput=table_input)
            logger.info(
                "Glue table updated",
                extra={"database": database, "table": table_name},
            )
        else:
            logger.error(
                "Failed to register Glue table",
                extra={"database": database, "table": table_name, "error": str(e)},
            )
            raise


def add_partitions(
    database: str,
    table_name: str,
    s3_location: str,
    year: str,
    month: str,
    day: str,
) -> None:
    """Register a new Hive-style partition in the Glue catalog."""
    if _LOCAL_MODE:
        logger.info(
            "LOCAL MODE — skipping partition registration",
            extra={"database": database, "table": table_name, "partition": f"{year}/{month}/{day}"},
        )
        return

    glue = _glue_client()
    partition_input = {
        "Values": [year, month, day],
        "StorageDescriptor": {
            "Location": f"{s3_location}year={year}/month={month}/day={day}/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            },
        },
    }
    try:
        glue.create_partition(
            DatabaseName=database,
            TableName=table_name,
            PartitionInput=partition_input,
        )
        logger.info(
            "Partition registered",
            extra={"database": database, "table": table_name, "partition": f"{year}/{month}/{day}"},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            logger.debug(
                "Partition already exists — skipping",
                extra={"partition": f"{year}/{month}/{day}"},
            )
        else:
            logger.error("Failed to register partition", extra={"error": str(e)})
            raise
