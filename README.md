# HEMA Data Engineering Assignment

## Overview

End-to-end ETL pipeline for HEMA Retail Sales data, built on **AWS** using a **Medallion Architecture** (Bronze → Silver → Gold). The pipeline ingests raw Superstore sales data, applies transformations, and publishes two domain-split datasets (`Sales` and `Customer`) to the AWS Glue Data Catalog.

See `docs/hema_architecture.png` for the full AWS architecture diagram.

---

## Why PySpark (not Pandas)

The pipeline currently processes a single CSV file, but in production the ingestion pattern is expected to scale significantly — multiple daily drops, regional feeds, backfills, and historical reloads arriving as many files simultaneously.

Pandas loads data into memory on a single driver process. That works fine for one file but becomes a hard ceiling as data volume grows — there is no path to distribute the load without rewriting the code. PySpark distributes reads, transforms, and writes across a cluster; scaling from 1 file to 10,000 files is a cluster-sizing decision, not a code change. Additional benefits at scale:

- **Predicate pushdown** — Gold jobs that aggregate over years of Silver partitions only scan the partitions they need, not the full dataset.
- **Glue job bookmarks** — Glue's incremental processing features work natively with PySpark, so daily runs process only new partitions rather than the full history.
- **AQE (Adaptive Query Execution)** — enabled by default; Spark automatically coalesces small partitions and handles skew without manual tuning.

---

## Architecture

| Layer | Service | Purpose |
|---|---|---|
| Ingestion | S3 | Raw file drop zone (single file or multi-file) |
| Orchestration | AWS Glue Workflows + Triggers | Daily batch scheduling with SUCCEEDED conditions |
| Processing | AWS Glue ETL Jobs (PySpark) | Bronze → Silver → Gold transforms |
| Catalog | AWS Glue Data Catalog | Schema registration and discoverability |
| Schema Evolution | Glue Crawler | Runs after Bronze ingestion; detects new columns and updates the Data Catalog automatically |
| Governance | AWS Lake Formation | Column-level access control per data mesh consumer team |
| CI/CD | AWS CodePipeline + CodeBuild | Lint, test, deploy Glue jobs on merge to main |
| Monitoring | CloudWatch | Job logs, alarms, and metrics |

> **Note on Schema Registry:** The architecture does not use AWS Glue Schema Registry. That service solves a streaming problem — ensuring producers and consumers on Kafka/Kinesis share a compatible wire format. For this batch, S3-based pipeline, schema evolution is handled by the Glue Crawler (which updates the Data Catalog table definition after each Bronze ingestion) and by the pipeline's own pass-through logic for unknown columns.

---

## Repository Structure

```
hema-data-pipeline/
├── configs/
│   └── pipeline_config.yaml
├── docs/
│   ├── architecture_diagram.md
│   └── hema_architecture.png       # Miro-style AWS architecture diagram
├── src/
│   ├── utils/
│   │   ├── logger.py               # Structured JSON logger (CloudWatch-compatible)
│   │   ├── spark_session.py        # SparkSession factory (local[*] + Glue)
│   │   ├── glue_catalog.py         # Glue Data Catalog helpers
│   │   └── schema_validator.py     # Schema evolution detector (new cols logged, not dropped)
│   ├── bronze/
│   │   └── ingest.py               # Raw CSV → Parquet (schema-on-read, all cols preserved)
│   ├── silver/
│   │   └── transform.py            # Rename, type-cast, dedup, quarantine
│   └── gold/
│       ├── sales.py                # Sales domain dataset
│       └── customer.py             # Customer dataset + order aggregations
├── tests/
│   ├── conftest.py                 # Session-scoped SparkSession + fixtures
│   ├── test_bronze.py
│   ├── test_silver.py
│   └── test_gold.py
├── requirements.txt
└── README.md
```

---

## Data Flow

```
S3 (raw CSV / multi-file drop)
    │
    ▼
[Bronze Job]   Schema-on-read · all columns preserved · audit cols added
    │                │
    │                └──▶ Glue Crawler ──▶ Glue Data Catalog (schema updated)
    ▼
[Silver Job]   Rename · type-cast · dedup (order_id + product_id grain)
    │                │
    │                └──▶ S3 Quarantine (invalid rows + _dq_failed_checks col)
    ▼
[Gold Jobs — parallel]
    ├── gold_sales      order_id · order_date · ship_date · ship_mode · city
    └── gold_customer   customer dims + orders_last_30d / 6m / all-time
              │
              ▼
    Glue Data Catalog ──▶ Lake Formation ──▶ Athena / QuickSight / data mesh consumers
```

---

## Deduplication Strategy

| Layer | Grain | Method |
|---|---|---|
| Silver | One row per `(order_id, product_id)` | `row_number()` window ordered by `_ingested_at` desc — most recently ingested copy wins |
| Gold Sales | One row per `order_id` | `groupBy("order_id").agg(F.max(...))` — collapses line items into a single order record |
| Gold Customer | One row per `customer_id` | `row_number()` window ordered by `order_date` desc — latest known attributes win |

---

## Data Quality — Quarantine Pattern

Rather than silently dropping rows with null critical keys (`order_id`, `customer_id`, `order_date`), the Silver job routes them to a separate quarantine path:

- Invalid rows are written to `s3://hema-data-lake/quarantine/silver_retail_sales/` with `mode("append")` so every daily run accumulates there.
- Each quarantined row gets a `_dq_failed_checks` column (e.g. `"order_id is null"`) and a `_dq_quarantined_at` timestamp.
- A sample of failures is logged to CloudWatch so engineers can diagnose the upstream issue without opening S3.
- Silver and Gold layers only ever contain valid rows — the quarantine path is a separate audit log, not part of the main pipeline.

To configure locally:

```bash
HEMA_LOCAL_MODE=true python -m src.silver.transform \
  --input_path output/bronze \
  --output_path output/silver \
  --quarantine_path output/quarantine
```

---

## Partitioning

All layers partition by Order Date:

```
s3://hema-data-lake/{bronze|silver|gold_sales|gold_customer}/year=YYYY/month=MM/day=DD/
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Run all tests (session-scoped SparkSession, local[*])
python -m pytest tests/ -v

# Run jobs individually
HEMA_LOCAL_MODE=true python -m src.bronze.ingest \
  --input_path data/superstore.csv \
  --output_path output/bronze

HEMA_LOCAL_MODE=true python -m src.silver.transform \
  --input_path output/bronze \
  --output_path output/silver \
  --quarantine_path output/quarantine

HEMA_LOCAL_MODE=true python -m src.gold.sales \
  --input_path output/silver \
  --output_path output/gold/sales

HEMA_LOCAL_MODE=true python -m src.gold.customer \
  --input_path output/silver \
  --output_path output/gold/customer
```

---

## Dataset

Source: [Kaggle Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)  
Latest date in dataset: **30 Dec 2018** — used as the reference anchor for rolling order aggregations in the Customer Gold job.