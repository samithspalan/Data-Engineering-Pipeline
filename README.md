# Data-Engineering-Pipeline
This project is about an End to End Data Engineering Pipeline using Apache Spark and Apache Airflow where raw data is processed, transformed, and stored in Delta format.

## Run Dashboard

1. Activate the project environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start Streamlit:

```powershell
streamlit run dashboard.py
```

## Windows Spark Notes

For local PySpark on Windows, both Java and Hadoop WinUtils are required.

1. Use JDK 17 and set `JAVA_HOME`.
2. Ensure `HADOOP_HOME` points to a folder that contains `bin\winutils.exe`.
3. Optional project-local layout supported by the dashboard: `.hadoop\bin\winutils.exe`.

If `winutils.exe` is missing, the dashboard falls back to a native Python Delta reader (`deltalake`) when dependencies are installed.

If local Spark setup is not available, run Spark jobs with Docker:

```powershell
docker compose run --rm spark /opt/spark/bin/spark-submit /spark_jobs/medallion_pipeline.py
```

## Airflow DAGs

Airflow DAG definitions are in `dags/pyspark_pipeline_dags.py`:

1. `daily_pyspark_pipeline`
- Runs every day at 5AM using cron: `0 5 * * *`
- Flow: `log_pipeline_start` -> `run_process_data` -> `log_pipeline_finish`

2. `event_driven_pyspark_pipeline`
- Polls for new files with `FileSensor`
- Flow: `detect_new_input_file` -> `log_pipeline_start` -> `run_process_data` -> `log_pipeline_finish`

Notes:
- Spark task retries are enabled (`retries=2`, `retry_delay=5 minutes`).
- Failures are logged by a dedicated `on_failure_callback`.
- `run_process_data` uses `BashOperator` and executes `process_data.py` via `spark-submit`.
- Configure Airflow connection `fs_default` so the FileSensor can resolve `/data/input/*`.
