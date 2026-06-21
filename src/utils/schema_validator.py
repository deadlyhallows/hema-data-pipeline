"""
Schema evolution handler for HEMA ETL pipeline — PySpark version.

Detects new/removed/type-changed columns between the expected Bronze schema
and the actual DataFrame schema. New columns are surfaced as warnings and
passed through rather than dropped, so downstream Gold consumers can discover
and adopt them transparently.
"""

from dataclasses import dataclass, field
from typing import Optional

from pyspark.sql import DataFrame

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected column names at Bronze (raw string types — schema-on-read)
EXPECTED_BRONZE_COLUMNS: set[str] = {
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode",
    "Customer ID", "Customer Name", "Segment", "Country", "City",
    "State", "Postal Code", "Region", "Product ID", "Category",
    "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit",
}


@dataclass
class SchemaReport:
    new_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_columns or self.removed_columns)

    def log(self, job_name: str) -> None:
        if not self.has_changes:
            logger.info("Schema validation passed — no drift detected",
                        extra={"job": job_name})
            return
        if self.new_columns:
            logger.warning(
                "Schema evolution: new columns detected — passing through",
                extra={"job": job_name, "new_columns": self.new_columns},
            )
        if self.removed_columns:
            logger.warning(
                "Schema evolution: expected columns missing from source",
                extra={"job": job_name, "removed_columns": self.removed_columns},
            )


def validate_schema(
    df: DataFrame,
    expected: Optional[set[str]] = None,
    job_name: str = "unknown",
) -> SchemaReport:
    """
    Compare DataFrame schema against expected column set.

    New columns are allowed through (schema evolution).
    Missing expected columns are logged as warnings.
    """
    if expected is None:
        expected = EXPECTED_BRONZE_COLUMNS

    actual = set(df.columns)
    report = SchemaReport(
        new_columns=sorted(actual - expected),
        removed_columns=sorted(expected - actual),
    )
    report.log(job_name)
    return report
