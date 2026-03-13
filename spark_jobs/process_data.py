"""
process_data.py  —  PySpark Data Engineering Pipeline
======================================================
Reads raw CSV data, cleans it, and saves it as Parquet using
PySpark's native writer (no pandas/pyarrow workaround needed).

Prerequisites on Windows:
  1. Run `python setup_winutils.py` once to download winutils.exe
  2. Then run this script normally: `python spark_jobs/process_data.py`
"""

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Bootstrap HADOOP_HOME before importing PySpark
#
# winutils.exe must be discoverable BEFORE the JVM starts.
# We read the path written by setup_winutils.py so no manual env-var
# configuration is needed on the machine.
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
        print(f"   Expected: {os.path.join(hadoop_home, 'bin', 'winutils.exe')}")
        print("   Run: python setup_winutils.py")
        sys.exit(1)
        
    # CRITICAL: add hadoop/bin to PATH so JVM can load hadoop.dll
    hadoop_bin = os.path.join(hadoop_home, "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
        
    print(f"[OK] HADOOP_HOME = {hadoop_home}")
else:
    print("[WARN] .hadoop_env not found - PySpark Parquet write may fail on Windows.")
    print("   Run: python setup_winutils.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Now it's safe to import PySpark
# ─────────────────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, avg

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Create Spark Session
#
# Key configs for Windows:
#   spark.hadoop.home.dir          → points Hadoop to winutils location
#   mapreduce.job.user.name        → satisfies Hadoop's user check
#   mapreduce.app-submission.cross-platform → skips POSIX-only code paths
# ─────────────────────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("DataEngineeringPipeline")

    # ── Java 17+ module access (required by Spark 4.x) ───────────────────────
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

    # ── Windows / winutils fixes ─────────────────────────────────────────────
    # Tell Hadoop's Shell where HADOOP_HOME is (redundant with env-var but safe)
    .config("spark.hadoop.home.dir", hadoop_home)

    # Satisfy Hadoop's job user check on Windows
    .config("spark.hadoop.mapreduce.job.user.name",
            os.environ.get("USERNAME", "spark"))

    # Skip the POSIX chmod / chown call that crashes without a real POSIX shell
    .config("spark.hadoop.mapreduce.app-submission.cross-platform", "true")

    # Use single partition for small data (avoids multiple part files)
    .config("spark.sql.shuffle.partitions", "1")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
input_path  = os.path.join(project_root, "data", "input",  "raw_data.csv")
output_path = os.path.join(project_root, "data", "output", "cleaned_data")

# Convert Windows backslashes → forward slashes for Hadoop URI compatibility
output_uri = "file:///" + output_path.replace("\\", "/")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Read the Input Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step 5: Reading Input Dataset ---")
df = spark.read.csv(input_path, header=True, inferSchema=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — View Raw Data
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step 6: Raw Data ---")
df.show()

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Remove Duplicate Records
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step 7: After Removing Duplicates ---")
df = df.dropDuplicates()
df.show()

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step 8: Handling Missing Values ---")
average_age_row = df.select(avg("age")).collect()[0][0]
average_age     = float(average_age_row) if average_age_row is not None else 0.0
print(f"  Computed Average Age = {average_age}")

df = df.fillna({"age": int(round(average_age))})
df.show()

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Add Processing Timestamp
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Step 9: Adding Processing Timestamp ---")
df = df.withColumn("processed_time", current_timestamp())
df.show()

# ─────────────────────────────────────────────────────────────────────────────
# Final Step — Save as Parquet using native PySpark writer
#
# We write exactly 1 partition (coalesce) so only one .parquet file is created.
# The file:/// URI prevents Hadoop from treating the path as HDFS.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Final Step: Saving Cleaned Data to Parquet at {output_path} ---")

(
    df.coalesce(1)
      .write
      .mode("overwrite")
      .parquet(output_uri)          # ← native PySpark Parquet writer
)

print("\n[OK] Pipeline execution completed successfully!")
print(f"   Output saved to: {output_path}")

spark.stop()
