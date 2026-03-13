import logging
import os
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


DELTA_COORDINATE = "io.delta:delta-spark_2.12:3.2.0"


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

def create_spark_session(app_name: str) -> SparkSession:
    _configure_windows_runtime()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", DELTA_COORDINATE)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .getOrCreate()
    )


def ensure_directories(paths: Tuple[str, ...]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def write_delta(df: DataFrame, output_path: str, mode: str = "overwrite") -> None:
    df.write.format("delta").mode(mode).option("overwriteSchema", "true").save(output_path)


def run_bronze_layer(spark: SparkSession, input_path: str, bronze_path: str, logger: logging.Logger) -> DataFrame:
    logger.info("Bronze layer: reading raw files from %s", input_path)

    bronze_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(os.path.join(input_path, "*.csv"))
        .withColumn("ingestion_timestamp", F.current_timestamp())
        .withColumn("source_file", F.input_file_name())
    )

    logger.info("Bronze layer: writing Delta table to %s", bronze_path)
    write_delta(bronze_df, bronze_path)

    return bronze_df


def run_silver_layer(spark: SparkSession, bronze_path: str, silver_path: str, logger: logging.Logger) -> DataFrame:
    logger.info("Silver layer: reading Bronze Delta table from %s", bronze_path)
    df = spark.read.format("delta").load(bronze_path)

    metadata_columns = {"ingestion_timestamp", "source_file", "processing_timestamp"}
    business_columns = [c for c in df.columns if c not in metadata_columns]

    logger.info("Silver layer: removing duplicate records")
    if business_columns:
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
        )
        .orderBy("order_date")
    )

    logger.info("Gold layer: writing Delta table to %s", gold_path)
    write_delta(gold_df, gold_path)

    return gold_df


def main() -> None:
    logger = configure_logging()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_path = os.path.join(project_root, "data", "input")
    bronze_path = os.path.join(project_root, "data", "bronze")
    silver_path = os.path.join(project_root, "data", "silver")
    gold_path = os.path.join(project_root, "data", "gold")

    ensure_directories((bronze_path, silver_path, gold_path))

    logger.info("Starting Medallion pipeline")
    spark = create_spark_session("MedallionPipeline")

    try:
        bronze_df = run_bronze_layer(spark, input_path, bronze_path, logger)
        logger.info("Bronze row count: %s", bronze_df.count())

        silver_df = run_silver_layer(spark, bronze_path, silver_path, logger)
        logger.info("Silver row count: %s", silver_df.count())

        gold_df = run_gold_layer(spark, silver_path, gold_path, logger)
        logger.info("Gold row count: %s", gold_df.count())

        logger.info("Medallion pipeline completed successfully")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
