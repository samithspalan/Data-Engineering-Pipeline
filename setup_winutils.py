"""
setup_winutils.py
-----------------
Downloads winutils.exe + hadoop.dll for Hadoop 3.x into this project's
local hadoop/bin directory so PySpark can write Parquet/Delta files on
Windows without manual system-level installation.

Run once before running the pipeline:
    python setup_winutils.py
"""

import os
import sys
import urllib.request
import shutil

# ── Configuration ─────────────────────────────────────────────────────────────
# Closest available winutils build to Hadoop 3.4.2
HADOOP_VERSION = "hadoop-3.3.6"

BASE_URL = (
    f"https://github.com/cdarlint/winutils/raw/master/{HADOOP_VERSION}/bin"
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
HADOOP_HOME = os.path.join(SCRIPT_DIR, "hadoop")
BIN_DIR     = os.path.join(HADOOP_HOME, "bin")

FILES = ["winutils.exe", "hadoop.dll"]
# ──────────────────────────────────────────────────────────────────────────────

def download(filename: str) -> None:
    dest = os.path.join(BIN_DIR, filename)
    if os.path.exists(dest):
        print(f"  ✓ {filename} already present — skipping download.")
        return
    url = f"{BASE_URL}/{filename}"
    print(f"  ↓ Downloading {filename} from GitHub ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, \
             open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        size_kb = os.path.getsize(dest) // 1024
        print(f"  ✓ {filename} saved ({size_kb} KB)")
    except Exception as exc:
        print(f"  ✗ Failed to download {filename}: {exc}")
        sys.exit(1)

def write_env_config() -> None:
    """Write HADOOP_HOME to a .env file so the pipeline script can read it."""
    env_path = os.path.join(SCRIPT_DIR, ".hadoop_env")
    with open(env_path, "w") as f:
        f.write(f"HADOOP_HOME={HADOOP_HOME}\n")
    print(f"\n  ✓ Hadoop home path written to: {env_path}")

def main() -> None:
    print("=" * 60)
    print(" Winutils Setup for PySpark on Windows")
    print(f" Hadoop build : {HADOOP_VERSION}")
    print(f" Install path : {BIN_DIR}")
    print("=" * 60)

    os.makedirs(BIN_DIR, exist_ok=True)

    print("\n[1/2] Downloading binaries …")
    for f in FILES:
        download(f)

    print("\n[2/2] Writing environment config …")
    write_env_config()

    print("\n" + "=" * 60)
    print(" ✅  Setup complete!")
    print(f"    HADOOP_HOME = {HADOOP_HOME}")
    print("    You can now run:  python spark_jobs/process_data.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
