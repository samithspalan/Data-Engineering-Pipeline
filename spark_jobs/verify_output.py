import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Bootstrap HADOOP_HOME before importing PySpark
# ─────────────────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_file     = os.path.join(project_root, ".hadoop_env")

if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if not os.path.isfile(os.path.join(hadoop_home, "bin", "winutils.exe")):
        print("  HADOOP_HOME is set but winutils.exe not found.")
        sys.exit(1)
        
    hadoop_bin = os.path.join(hadoop_home, "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
else:
    print("[WARN] .hadoop_env not found - PySpark may fail on Windows.")
    sys.exit(1)

from pyspark.sql import SparkSession

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Create Spark Session
# ─────────────────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("VerifyOutput")
    .config(
        "spark.driver.extraJavaOptions",
        " ".join([
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
            "--add-opens=java.base/java.io=ALL-UNNAMED",
            "--add-opens=java.base/java.net=ALL-UNNAMED",
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/java.util=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
            "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        ])
    )
    .config("spark.hadoop.home.dir", hadoop_home)
    .config("spark.hadoop.mapreduce.job.user.name", os.environ.get("USERNAME", "spark"))
    .config("spark.hadoop.mapreduce.app-submission.cross-platform", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Read and Verify the Parquet Output
# ─────────────────────────────────────────────────────────────────────────────
output_path = os.path.join(project_root, "data", "output", "cleaned_data")
output_uri = "file:///" + output_path.replace("\\", "/")

print(f"\n--- Reading Parquet Output from: {output_path} ---")

try:
    df_cleaned = spark.read.parquet(output_uri)
    
    print("\n[OK] Successfully loaded cleaned data!")
    print("\n--- Schema ---")
    df_cleaned.printSchema()
    
    print("\n--- Data ---")
    df_cleaned.show(truncate=False)
    
    row_count = df_cleaned.count()
    print(f"\nTotal rows in parquet file: {row_count}")
    
except Exception as e:
    print(f"\n[ERROR] Failed to read parquet data: {e}")

finally:
    spark.stop()
