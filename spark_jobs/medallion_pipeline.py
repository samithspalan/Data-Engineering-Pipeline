import logging
import os
import re
import glob
import sys
from typing import List, Tuple

from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("medallion_pipeline")


def _configure_windows_runtime() -> None:
    """Configure HADOOP_HOME on Windows using the local hadoop directory."""
    if os.name != "nt":
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    hadoop_home = os.path.join(project_root, "hadoop")
    
    if os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home
        hadoop_bin = os.path.join(hadoop_home, "bin")
        if hadoop_bin not in os.environ["PATH"]:
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")


def _configure_java_runtime() -> None:
    """Prefer a Spark-compatible Java runtime to avoid gateway startup failures."""
    candidate_paths = [
        os.environ.get("JAVA_HOME", ""),
        r"C:\Users\User\.jdk\jdk-17.0.16",
        r"C:\Program Files\Java\jdk-17",
    ]

    java_home = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not java_home:
        return

    os.environ["JAVA_HOME"] = java_home
    java_bin = os.path.join(java_home, "bin")
    current_path = os.environ.get("PATH", "")
    if java_bin not in current_path:
        os.environ["PATH"] = java_bin + os.pathsep + current_path

def create_spark_session(app_name: str) -> SparkSession:
    _configure_java_runtime()
    _configure_windows_runtime()

    # Ensure Spark workers use the same Python interpreter as this process.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def ensure_directories(paths: Tuple[str, ...]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _canonicalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")
    return normalized or "col"


def _normalize_batch_columns(df: DataFrame) -> DataFrame:
    """Normalize CSV headers so unionByName can reliably align evolving schemas."""
    seen: dict[str, int] = {}
    output_cols: list[str] = []
    for col_name in df.columns:
        base_name = _canonicalize_column_name(col_name)
        seen[base_name] = seen.get(base_name, 0) + 1
        suffix = seen[base_name]
        output_cols.append(base_name if suffix == 1 else f"{base_name}_{suffix}")

    normalized_df = df.toDF(*output_cols)

    # Keep downstream logic stable across common source header variants.
    if "product_name" in normalized_df.columns and "product" not in normalized_df.columns:
        normalized_df = normalized_df.withColumnRenamed("product_name", "product")

    return normalized_df


def _read_single_csv_batch(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Read one CSV batch with dynamic-width row support.

    Why: Schema Evolution belongs in Delta writes, but upstream files may contain
    rows with additional trailing values before the whole source header is updated.
    This parser now strictly discards extra columns not specified in the header to
    enforce clean schema evolution as per user request.
    """
    raw_df = (
        spark.read.text(file_path)
        .where(F.col("value").isNotNull() & (F.length(F.trim(F.col("value"))) > 0))
    )

    header_row = raw_df.first()
    if header_row is None:
        raise ValueError(f"CSV file is empty: {file_path}")

    header_tokens = [token.strip() for token in str(header_row["value"]).split(",")]
    normalized_first_row = [_canonicalize_column_name(token) for token in header_tokens]
    known_headers = {
        "order_id", "customer_id", "product", "product_name", "unit_price",
        "quantity", "order_date", "event_timestamp", "date", "revenue",
        "amount", "discount_code",
    }
    has_header = any(token in known_headers for token in normalized_first_row)

    data_df = raw_df.where(F.col("value") != F.lit(header_row["value"])) if has_header else raw_df
    if data_df.limit(1).count() == 0:
        fallback_cols = normalized_first_row if has_header else ["order_id", "customer_id", "product", "unit_price", "quantity", "order_date"]
        empty_exprs = [F.lit(None).cast("string").alias(col_name) for col_name in fallback_cols]
        return raw_df.limit(0).select(*empty_exprs)

    split_col = F.split(F.col("value"), ",")

    if has_header:
        header_cols = normalized_first_row
    else:
        header_cols = ["order_id", "customer_id", "product", "unit_price", "quantity", "order_date"]

    select_exprs = []
    for idx, col_name in enumerate(header_cols):
        select_exprs.append(F.trim(F.get(split_col, F.lit(idx))).alias(col_name))

    parsed_df = data_df.select(*select_exprs)
    return _normalize_batch_columns(parsed_df)


def write_delta(
    df: DataFrame,
    output_path: str,
    mode: str = "overwrite",
    partition_cols: list = None,
    evolve_schema: bool = True,
) -> None:
    writer = df.write.format("delta").mode(mode)

    # Schema Evolution happens here at Delta write-time.
    # mergeSchema expands the Delta table schema when new columns arrive in incoming batches.
    # Existing rows in older files remain valid and will read NULL for the newly added columns.
    if evolve_schema:
        writer = writer.option("mergeSchema", "true")
    else:
        writer = writer.option("overwriteSchema", "true")

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.save(output_path)


def _read_and_union_csv_batches(spark: SparkSession, input_path: str, logger: logging.Logger) -> DataFrame:
    csv_pattern = os.path.join(input_path, "*.csv")
    csv_files: List[str] = sorted(glob.glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at: {csv_pattern}")

    logger.info("Bronze layer: discovered %s input file(s)", len(csv_files))

    union_df = None
    for file_path in csv_files:
        batch_df = _read_single_csv_batch(spark, file_path)

        if union_df is None:
            union_df = batch_df
        else:
            # Allow new batch-specific columns (for example discount_code) to flow into Bronze.
            # Missing columns in older batches are automatically filled with NULL by Spark.
            union_df = union_df.unionByName(batch_df, allowMissingColumns=True)

    if union_df is None:
        raise ValueError("CSV read produced an empty DataFrame")

    return union_df


def run_bronze_layer(spark: SparkSession, input_path: str, bronze_path: str, logger: logging.Logger) -> DataFrame:
    logger.info("Bronze layer: syncing records from %s (full refresh to support removals)", input_path)

    # Read each batch and union by column names so new columns can evolve safely.
    bronze_df = _read_and_union_csv_batches(spark, input_path, logger)

    # Harmonize common column aliases to keep downstream logic stable.
    if "product_name" in bronze_df.columns and "product" not in bronze_df.columns:
        bronze_df = bronze_df.withColumnRenamed("product_name", "product")

    # Apply known type casts after dynamic parsing to preserve downstream logic.
    integer_like = ["order_id", "customer_id", "quantity"]
    for col_name in integer_like:
        if col_name in bronze_df.columns:
            bronze_df = bronze_df.withColumn(col_name, F.col(col_name).cast("long"))

    decimal_like = ["unit_price", "price", "revenue", "amount"]
    for col_name in decimal_like:
        if col_name in bronze_df.columns:
            bronze_df = bronze_df.withColumn(col_name, F.col(col_name).cast("double"))

    bronze_df = (
        bronze_df.withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("source_file", F.input_file_name())
    )

    # Overwrite mode ensures the Bronze table is a perfect mirror of the input folder
    logger.info("Bronze layer: updating Delta table at %s", bronze_path)
    write_delta(bronze_df, bronze_path, mode="overwrite")

    return bronze_df


def run_quarantine_layer(
    spark: SparkSession,
    bronze_path: str,
    quarantine_path: str,
    validated_path: str,
    logger: logging.Logger,
) -> None:
    """Split Bronze records into quarantined (invalid) and validated (clean) sets.

    Quarantined rows are written to a dedicated Delta table with a rejection_reason
    column. Only the clean rows are written to the validated Delta table, which
    Silver then reads from instead of Bronze.
    """
    logger.info("Quarantine layer: reading Bronze Delta table from %s", bronze_path)
    df = spark.read.format("delta").load(bronze_path)

    # Build a single rejection_reason expression — first matching rule wins.
    rejection_expr = F.lit(None).cast("string")

    if "order_id" in df.columns:
        rejection_expr = F.when(
            F.col("order_id").isNull(), F.lit("null_order_id")
        ).otherwise(rejection_expr)

    if "order_date" in df.columns:
        rejection_expr = F.when(
            F.col("order_date").isNull() | (F.trim(F.col("order_date").cast("string")) == ""),
            F.lit("null_order_date"),
        ).otherwise(rejection_expr)
        rejection_expr = F.when(
            F.col("order_date").isNotNull() & F.to_date(F.col("order_date")).isNull(),
            F.lit("invalid_order_date_format"),
        ).otherwise(rejection_expr)

    if "unit_price" in df.columns:
        rejection_expr = F.when(
            F.col("unit_price").cast("double") <= 0,
            F.lit("non_positive_unit_price"),
        ).otherwise(rejection_expr)

    if "quantity" in df.columns:
        rejection_expr = F.when(
            F.col("quantity").cast("double") <= 0,
            F.lit("non_positive_quantity"),
        ).otherwise(rejection_expr)

    df_tagged = df.withColumn("rejection_reason", rejection_expr)
    quarantine_df = df_tagged.filter(F.col("rejection_reason").isNotNull())
    valid_df = df_tagged.filter(F.col("rejection_reason").isNull()).drop("rejection_reason")

    quarantine_count = quarantine_df.count()
    valid_count = valid_df.count()
    logger.info(
        "Quarantine layer: %d valid | %d quarantined (total Bronze: %d)",
        valid_count, quarantine_count, valid_count + quarantine_count,
    )

    if quarantine_count > 0:
        logger.info("Quarantine layer: writing %d rejected rows to %s", quarantine_count, quarantine_path)
        write_delta(quarantine_df, quarantine_path, mode="overwrite", evolve_schema=False)

    logger.info("Quarantine layer: writing %d validated rows to %s", valid_count, validated_path)
    write_delta(valid_df, validated_path, mode="overwrite", evolve_schema=False)


def run_silver_layer(spark: SparkSession, source_path: str, silver_path: str, logger: logging.Logger) -> DataFrame:
    logger.info("Silver layer: reading validated Delta table from %s", source_path)
    df = spark.read.format("delta").load(source_path)
    metadata_columns = {"ingestion_timestamp", "source_file", "processing_timestamp"}
    business_columns = [c for c in df.columns if c not in metadata_columns]

    preferred_dedupe_keys = ["order_id", "order_date", "customer_id"]
    dedupe_columns = [c for c in preferred_dedupe_keys if c in df.columns]

    logger.info("Silver layer: removing duplicate records")
    if "order_id" in df.columns:
        logger.info("Silver layer: deduplicating by order_id (keeping latest record)")
        sort_exprs = []
        if "order_date" in df.columns:
            sort_exprs.append(F.to_timestamp(F.col("order_date")))
        if "event_timestamp" in df.columns:
            sort_exprs.append(F.to_timestamp(F.col("event_timestamp")))
        sort_exprs.append(F.col("ingestion_timestamp").cast("timestamp"))
        order_date_sort_col = F.coalesce(*sort_exprs)

        row_window = Window.partitionBy("order_id").orderBy(order_date_sort_col.desc_nulls_last())
        df = (
            df.withColumn("_row_num", F.row_number().over(row_window))
            .filter(F.col("_row_num") == 1)
            .drop("_row_num")
        )
    elif dedupe_columns:
        logger.info("Silver layer: deduplicating using key columns %s", dedupe_columns)
        df = df.dropDuplicates(dedupe_columns)
    elif business_columns:
        logger.info("Silver layer: deduplicating using all business columns")
        df = df.dropDuplicates(business_columns)
    else:
        df = df.dropDuplicates()

    logger.info("Silver layer: handling missing/null values")
    fill_map = {}
    for field in df.schema.fields:
        if field.name in metadata_columns:
            continue

        if isinstance(field.dataType, (T.IntegerType, T.LongType, T.ShortType, T.ByteType)):
            fill_map[field.name] = 0
        elif isinstance(field.dataType, (T.FloatType, T.DoubleType, T.DecimalType)):
            fill_map[field.name] = 0.0
        elif isinstance(field.dataType, T.StringType):
            fill_map[field.name] = "UNKNOWN"

    if fill_map:
        df = df.fillna(fill_map)

    if "age" in df.columns:
        avg_age = df.select(F.avg("age").alias("avg_age")).collect()[0]["avg_age"]
        if avg_age is not None:
            df = df.fillna({"age": float(avg_age)})

    df = df.withColumn("processing_timestamp", F.current_timestamp())

    logger.info("Silver layer: writing Delta table to %s", silver_path)
    write_delta(df, silver_path)

    return df


def resolve_order_date_column(df: DataFrame) -> F.Column:
    if "order_date" in df.columns:
        return F.to_date(F.col("order_date"))

    if "event_timestamp" in df.columns:
        return F.to_date(F.col("event_timestamp"))

    if "processing_timestamp" in df.columns:
        return F.to_date(F.col("processing_timestamp"))

    return F.to_date(F.col("ingestion_timestamp"))


def resolve_revenue_column(df: DataFrame) -> F.Column:
    if "revenue" in df.columns:
        return F.coalesce(F.col("revenue").cast("double"), F.lit(0.0))

    if "amount" in df.columns:
        return F.coalesce(F.col("amount").cast("double"), F.lit(0.0))

    if "unit_price" in df.columns and "quantity" in df.columns:
        return (
            F.coalesce(F.col("unit_price").cast("double"), F.lit(0.0))
            * F.coalesce(F.col("quantity").cast("double"), F.lit(0.0))
        )

    return F.lit(0.0)


def run_gold_layer(spark: SparkSession, silver_path: str, gold_path: str, logger: logging.Logger) -> DataFrame:
    logger.info("Gold layer: reading Silver Delta table from %s", silver_path)
    df = spark.read.format("delta").load(silver_path)

    logger.info("Gold layer: generating analytics (orders and revenue per day)")
    order_date_col = resolve_order_date_column(df)
    revenue_col = resolve_revenue_column(df)

    gold_df = (
        df.withColumn("order_date", order_date_col)
        .withColumn("revenue_value", revenue_col)
        .groupBy("order_date")
        .agg(
            F.count(F.lit(1)).alias("total_orders_per_day"),
            F.sum("revenue_value").alias("total_revenue_per_day"),
            F.concat_ws(", ", F.collect_list(F.col("order_id"))).alias("order_ids")
        )
        .select("order_date", "order_ids", "total_orders_per_day", "total_revenue_per_day")
        .orderBy("order_date")
    )

    logger.info("Gold layer: writing Delta table to %s (partitioned by order_date)", gold_path)
    write_delta(gold_df, gold_path, partition_cols=["order_date"])

    return gold_df


def main() -> None:
    logger = configure_logging()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_path = os.path.join(project_root, "data", "input")
    bronze_path = os.path.join(project_root, "data", "bronze")
    quarantine_path = os.path.join(project_root, "data", "quarantine")
    validated_path = os.path.join(project_root, "data", "validated")
    silver_path = os.path.join(project_root, "data", "silver")
    gold_path = os.path.join(project_root, "data", "gold")

    ensure_directories((bronze_path, quarantine_path, validated_path, silver_path, gold_path))

    logger.info("Starting Medallion pipeline")
    spark = create_spark_session("MedallionPipeline")

    try:
        bronze_df = run_bronze_layer(spark, input_path, bronze_path, logger)
        logger.info("Bronze row count: %s", bronze_df.count())

        run_quarantine_layer(spark, bronze_path, quarantine_path, validated_path, logger)

        silver_df = run_silver_layer(spark, validated_path, silver_path, logger)
        logger.info("Silver row count: %s", silver_df.count())

        gold_df = run_gold_layer(spark, silver_path, gold_path, logger)
        logger.info("Gold row count: %s", gold_df.count())

        logger.info("Medallion pipeline completed successfully")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
