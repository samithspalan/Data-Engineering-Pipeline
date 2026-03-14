import logging
import os
import shutil
import glob
from datetime import datetime
from typing import Dict, Any

from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ── 1. Configuration & Schema Definition ─────────────────────────────────────
# Using Static StructType to prevent misinterpretation and improve performance
TARGET_SCHEMA = T.StructType([
    T.StructField("order_id", T.StringType(), True),
    T.StructField("order_date", T.StringType(), True),
    T.StructField("product", T.StringType(), True),
    T.StructField("revenue", T.DoubleType(), True)
])

REQUIRED_FIELDS = ["order_id", "order_date"]
VALID_EXTENSIONS = [".csv", ".json"]
THRESHOLD_RATIO = 0.5  # Max 50% duplicates or nulls allowed

def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("validation_pipeline")

def create_spark_session(app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Ensure winutils compatibility on Windows
        .config("spark.hadoop.fs.permissions.umask-mode", "000")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

def ensure_directories(project_root: str) -> Dict[str, str]:
    paths = {
        "input":       os.path.join(project_root, "data", "input"),
        "unsupported": os.path.join(project_root, "data", "unsupported"),
        "rejected":    os.path.join(project_root, "data", "rejected"),
        "quarantine":  os.path.join(project_root, "data", "quarantine"),
        "silver":      os.path.join(project_root, "data", "silver", "clean_table"),
        "logs":        os.path.join(project_root, "data", "logs")
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

# ── 2. Validation Logic ───────────────────────────────────────────────────────

def route_unsupported_files(paths: Dict[str, str], logger: logging.Logger):
    files = glob.glob(os.path.join(paths["input"], "*"))
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in VALID_EXTENSIONS:
            logger.warning(f"Routing unsupported file to {paths['unsupported']}: {os.path.basename(f)}")
            shutil.move(f, os.path.join(paths["unsupported"], os.path.basename(f)))

def compute_quality_metrics(df: DataFrame, total_rows: int) -> Dict[str, float]:
    if total_rows == 0:
        return {"dup_ratio": 0.0, "null_ratio": 0.0}
    
    unique_rows = df.distinct().count()
    dup_ratio = (total_rows - unique_rows) / total_rows
    
    # Calculate nulls in required fields
    null_rows = df.filter(F.col("id").isNull() | F.col("name").isNull()).count()
    null_ratio = null_rows / total_rows
    
    return {"dup_ratio": dup_ratio, "null_ratio": null_ratio}

def process_file(spark: SparkSession, file_path: str, paths: Dict[str, str], logger: logging.Logger):
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    
    logger.info(f"Processing candidate: {filename}")
    
    try:
        # Load with Static Schema
        if ext == ".csv":
            df = spark.read.csv(file_path, header=True, schema=TARGET_SCHEMA)
        else:
            df = spark.read.json(file_path, schema=TARGET_SCHEMA)
            
        # Schema Validation: Ensure required columns actually entered the DataFrame
        actual_cols = df.columns
        missing = [c for c in REQUIRED_FIELDS if c not in actual_cols]
        
        if missing:
            logger.error(f"Missing required columns {missing}. Rejecting {filename}")
            shutil.move(file_path, os.path.join(paths["rejected"], filename))
            return

        total_rows = df.count()
        if total_rows == 0:
            logger.warning(f"File {filename} is empty. Rejecting.")
            shutil.move(file_path, os.path.join(paths["rejected"], filename))
            return

        # Metrics & Thresholds
        metrics = compute_quality_metrics(df, total_rows)
        logger.info(f"Metrics for {filename}: {metrics}")
        
        if metrics["dup_ratio"] > THRESHOLD_RATIO or metrics["null_ratio"] > THRESHOLD_RATIO:
            logger.error(f"Threshold exceeded for {filename}. Rejecting.")
            shutil.move(file_path, os.path.join(paths["rejected"], filename))
            return

        # Quarantine Bad Rows
        quarantine_rows = df.filter(F.col("id").isNull() | F.col("name").isNull())
        if quarantine_rows.count() > 0:
            logger.info(f"Quarantining {quarantine_rows.count()} rows from {filename}")
            quarantine_rows.write.format("delta").mode("append").save(paths["quarantine"])

        # Clean Dataset
        clean_df = df.filter(F.col("id").isNotNull() & F.col("name").isNotNull())
        clean_df = (
            clean_df.dropDuplicates(["id"])
            .fillna({"age": 0})
            .withColumn("processing_timestamp", F.current_timestamp())
        )

        # Write to Silver Clean Table
        logger.info(f"Writing {clean_df.count()} valid rows from {filename} to Silver layer.")
        clean_df.write.format("delta").mode("append").save(paths["silver"])
        
        # Archive processed file
        archive_path = os.path.join(paths["input"], "processed")
        os.makedirs(archive_path, exist_ok=True)
        shutil.move(file_path, os.path.join(archive_path, filename))

    except Exception as e:
        logger.error(f"Critical failure processing {filename}: {str(e)}")
        shutil.move(file_path, os.path.join(paths["rejected"], filename))

def main():
    logger = configure_logging()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = ensure_directories(project_root)
    
    spark = create_spark_session("ValidationPipeline")
    
    try:
        # Stage 1: Route unsupported files
        route_unsupported_files(paths, logger)
        
        # Stage 2: Process valid files
        candidates = glob.glob(os.path.join(paths["input"], "*"))
        for cand in candidates:
            if os.path.isdir(cand): continue # Skip 'processed' folder
            process_file(spark, cand, paths, logger)
            
        logger.info("Pipeline run completed.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
