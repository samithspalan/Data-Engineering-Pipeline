# 🚀 Medallion Architecture Data Pipeline with Delta Lake & Streamlit

An enterprise-grade, event-driven data engineering pipeline implementing the **Medallion Architecture (Bronze → Silver → Gold)**. This project leverages **Apache Spark** for processing, **Delta Lake** for ACID transactions and Time-Travel, **Apache Airflow** for orchestration, and **Streamlit** for real-time monitoring and Root-Cause Analysis (RCA).

---

## 🏗️ Architecture Overview

The pipeline processes raw CSV and JSON data through three distinct layers:
1.  **Bronze (Raw)**: Captures incoming raw data in its original form.
2.  **Silver (Validated)**: Cleans, deduplicates, and enforces data quality checks. Rejected records are redirected to a **Quarantine** table.
3.  **Gold (Aggregated)**: Stores business-level aggregates (Daily orders, growth metrics) optimized for ultra-fast dashboarding.

---

## ✨ Key Features built

### 📊 Real-Time Analytics Dashboard
- **Live Heartbeat**: 1-second auto-refresh engine for zero-latency monitoring.
- **Dynamic Funnel**: Visualizes the flow from Bronze through Rejections to Silver using interactive donut charts.
- **Business Intelligence**: Month-on-Month (MoM) order growth tracking with automated percentage calculations.
- **Partition Pruning Simulation**: Interactive UI to demonstrate Spark's performance optimization by selectively scanning partition folders.

### 🕒 Delta Lake Time-Travel & Audit
- **Pipeline Update History**: A full-width stacked bar chart tracking the last 5 updates.
- **Quality Evolution**: Side-by-side comparison of **Silver (Clean)** vs **Quarantine (Rejected)** records across versions.
- **Rollback Demonstration**: Explained workflows for using `versionAsOf` to debug data drift and perform zero-copy restores.

### 🛠️ Advanced Root-Cause Analysis (RCA) [Diagnostic Tab]
- **Trend Visualization**: High-level line charts for Row Counts, DQ Scores, and Rejection Ratios.
- **Column-Level Diagnostics**: Automated null-checks, uniqueness analysis, and record sampling for any historical version.
- **Record Inspection**: Side-by-side data frames showing Valid records vs Rejected records (with specific rejection reasons) for auditability.

### 🤖 Intelligent Orchestration & Alerts
- **Hybrid Scheduling**: Supports both Daily Batch (Scheduled) and Event-Driven (polling `data/input`) triggers using Airflow.
- **Automated Notifications**: Integrated **Resend API** for email alerts on pipeline Success/Failure.
- **Error Handling**: Dedicated callbacks and quality logs stored in `quality_report.json`.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Java JDK 17** (Required for Apache Spark)
- **Hadoop Winutils** (If running on Windows)

### 2. Environment Setup
Clone the repository and set up a virtual environment:
```powershell
# Create and activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (refer to `.env.example`):
```env
RESEND_API_KEY=your_api_key_here
OWNER_EMAIL=your_email@example.com
```

### 4. Running the Project

#### 🚀 Launch the Dashboard
```powershell
streamlit run dashboard.py
```

#### ⚙️ Execute Spark Job (Local)
```powershell
spark-submit spark_jobs/medallion_pipeline.py
```

#### 🐳 Docker Execution (Spark-only)
```powershell
docker compose run --rm spark /opt/spark/bin/spark-submit /spark_jobs/medallion_pipeline.py
```

---

## 📁 Project Structure
- `dashboard.py`: Main Streamlit application with live analytics and RCA.
- `spark_jobs/`: PySpark implementation of the Medallion architecture.
- `dags/`: Airflow DAG definitions for scheduled and event-driven runs.
- `data/`: Local storage for Delta tables (Bronze, Silver, Gold, Quarantine).
- `utils/`: Helper modules for notifications and pipeline status.

---

## 🔍 Auditing & Debugging
To inspect a specific historical state of the Silver table via Spark:
```python
df = spark.read.format("delta").option("versionAsOf", 5).load("data/silver")
```
Rejections can be found in `data/quarantine` with the `rejection_reason` column explaining why the record failed validation.
