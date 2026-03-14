import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

# Ensure project root is on sys.path so 'utils' package is always importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

st.set_page_config(page_title="Data Pipeline Analytics Dashboard", layout="wide")


class SparkRuntimePrecheckError(RuntimeError):
    """Raised when local Spark prerequisites are not available."""


def _configure_java_runtime() -> None:
    """Force a Spark-compatible Java runtime when available (Windows-safe)."""
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


def _configure_hadoop_runtime() -> None:
    """Configure HADOOP_HOME on Windows when winutils.exe is available."""
    if os.name != "nt":
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_hadoop_homes = [
        os.path.join(script_dir, "hadoop"),
        os.path.join(script_dir, ".hadoop"),
        os.environ.get("HADOOP_HOME", ""),
        os.environ.get("hadoop.home.dir", ""),
        r"C:\hadoop",
        r"C:\tools\hadoop",
        os.path.join(os.path.expanduser("~"), "hadoop"),
    ]

    selected_home = None
    for home in candidate_hadoop_homes:
        if not home:
            continue

        normalized_home = os.path.abspath(home)
        winutils_path = os.path.join(normalized_home, "bin", "winutils.exe")
        if os.path.exists(winutils_path):
            selected_home = normalized_home
            break

    if selected_home is None:
        raise SparkRuntimePrecheckError(
            "Failed to load Gold Delta table: Spark on Windows requires winutils.exe.\n\n"
            "Fix options:\n"
            "1) Install Hadoop WinUtils and set HADOOP_HOME to that folder (must contain bin\\winutils.exe).\n"
            "2) Place winutils.exe at .hadoop\\bin\\winutils.exe under this project.\n"
            "3) Run Spark jobs in Docker and use the generated Delta output."
        )

    os.environ["HADOOP_HOME"] = selected_home
    os.environ["hadoop.home.dir"] = selected_home

    hadoop_bin = os.path.join(selected_home, "bin")
    current_path = os.environ.get("PATH", "")
    if hadoop_bin not in current_path:
        os.environ["PATH"] = hadoop_bin + os.pathsep + current_path


def create_spark_session() -> SparkSession:
    _configure_java_runtime()
    _configure_hadoop_runtime()

    builder = (
        SparkSession.builder.appName("StreamlitGoldDashboard")
        .config("spark.master", "local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.ivy", os.path.join(os.getcwd(), ".ivy2"))
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
    )

    # Ensure Spark picks the same Python interpreter as Streamlit.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return configure_spark_with_delta_pip(builder).getOrCreate()


def _compute_medallion_realtime(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes real-time Bronze, Silver, and File Issues views from raw input files."""
    import glob
    all_files = glob.glob(os.path.join(input_path, "*"))
    if not all_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dfs = []
    file_issues = []
    
    for f in all_files:
        filename = os.path.basename(f)
        extension = Path(f).suffix.lower()
        
        # ── 1. Check for Supported File Extension (.csv or .json) ──
        if extension not in ['.csv', '.json']:
            file_issues.append({
                "order_id": "N/A",
                "source_file": filename,
                "Reason": f"Unsupported File Format: File extension '{extension}' is not supported. Use .csv or .json only.",
                "order_date": "N/A",
                "product": "N/A"
            })
            continue

        try:
            temp_df = pd.DataFrame()
            
            # ── 2. Handle JSON Format ──
            if extension == '.json':
                try:
                    temp_df = pd.read_json(f)
                    # If it's a single object, wrap it in a list
                    if isinstance(temp_df, pd.Series):
                        temp_df = temp_df.to_frame().T
                except Exception as e:
                    file_issues.append({
                        "order_id": "CRITICAL",
                        "source_file": filename,
                        "Reason": f"Corrupted JSON: File could not be parsed. Error: {str(e)}",
                        "order_date": "N/A",
                        "product": "N/A"
                    })
                    continue

            # ── 3. Handle CSV Format ──
            else:
                try:
                    import csv
                    with open(f, 'r', encoding='utf-8-sig') as file:
                        reader = csv.reader(file)
                        data = list(reader)
                        
                    if not data: 
                        file_issues.append({
                            "order_id": "EMPTY",
                            "source_file": filename,
                            "Reason": "Corrupted Data: File is empty or has no rows.",
                            "order_date": "N/A",
                            "product": "N/A"
                        })
                        continue
                    
                    first_row = data[0]
                    header_keywords = ["id", "order", "date", "revenue", "product", "amount"]
                    has_header = any(any(key in str(col).lower() for key in header_keywords) for col in first_row)
                    
                    temp_df = pd.DataFrame(data)
                    if has_header:
                        num_cols = len(first_row)
                        temp_df = temp_df.iloc[:, :num_cols]
                        temp_df.columns = first_row
                        temp_df = temp_df.iloc[1:].reset_index(drop=True)
                    else:
                        canonical = ["order_id", "order_date", "product", "revenue"]
                        if len(first_row) < 2:
                            raise ValueError(f"Inconsistent columns: Expected at least 2 (order_id, order_date), found {len(first_row)}")
                        num_cols = len(canonical)
                        temp_df = temp_df.iloc[:, :num_cols]
                        actual_cols = temp_df.shape[1]
                        temp_df.columns = canonical[:actual_cols]
                except Exception as e:
                    file_issues.append({
                        "order_id": "PARSE_ERROR",
                        "source_file": filename,
                        "Reason": f"Corrupted CSV: {str(e)}",
                        "order_date": "N/A",
                        "product": "N/A"
                    })
                    continue

            # ── 4. Structural Validation & Normalization ──
            if temp_df.empty:
                continue

            # Normalize and deduplicate headers
            temp_df.columns = [str(c).strip().lower() for c in temp_df.columns]
            temp_df = temp_df.loc[:, ~temp_df.columns.duplicated(keep="first")]

            rename_map = {
                "product_name": "product",
                "date": "order_date",
                "event_timestamp": "order_date",
                "id": "order_id",
                "name": "product"
            }
            temp_df.rename(columns=rename_map, inplace=True)

            if "order_id" not in temp_df.columns or "order_date" not in temp_df.columns:
                file_issues.append({
                    "order_id": "SCHEMA_ERROR",
                    "source_file": filename,
                    "Reason": "Validation Failed: Missing required columns (order_id, order_date).",
                    "order_date": "N/A",
                    "product": "N/A"
                })
                continue
            
            temp_df["ingestion_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            temp_df["source_file"] = filename
            dfs.append(temp_df)
        except Exception as e:
            file_issues.append({
                "order_id": "IO_ERROR",
                "source_file": filename,
                "Reason": f"Access Error: File could not be read. Error: {str(e)}",
                "order_date": "N/A",
                "product": "N/A"
            })

    file_issues_df = pd.DataFrame(file_issues)
    if not dfs:
        return pd.DataFrame(), pd.DataFrame(), file_issues_df

    bronze_rt = pd.concat(dfs, ignore_index=True, sort=False)

    # Silver is the Cleaned version: Deduplicated on order_id
    dedupe_cols = [c for c in ["order_id", "order_date"] if c in bronze_rt.columns]
    silver_rt = bronze_rt.drop_duplicates(subset=dedupe_cols or None, keep="first").copy()
    silver_rt["processing_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return bronze_rt, silver_rt, file_issues_df


def _compute_gold_from_raw(input_path: str) -> pd.DataFrame:
    """Replicates Spark Gold logic in-memory for instant feedback, supporting CSV and JSON."""
    import glob
    all_files = glob.glob(os.path.join(input_path, "*"))
    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        extension = Path(f).suffix.lower()
        if extension not in ['.csv', '.json']:
            continue
            
        try:
            temp_df = pd.DataFrame()
            if extension == '.json':
                temp_df = pd.read_json(f)
                if isinstance(temp_df, pd.Series):
                    temp_df = temp_df.to_frame().T
            else:
                import csv
                with open(f, 'r', encoding='utf-8-sig') as file:
                    reader = csv.reader(file)
                    data = list(reader)
                if not data: continue
                
                first_row = data[0]
                header_keywords = ["id", "order", "date", "revenue", "product", "amount"]
                has_header = any(any(key in str(col).lower() for key in header_keywords) for col in first_row)
                
                temp_df = pd.DataFrame(data)
                if has_header:
                    num_cols = len(first_row)
                    temp_df = temp_df.iloc[:, :num_cols]
                    temp_df.columns = first_row
                    temp_df = temp_df.iloc[1:].reset_index(drop=True)
                else:
                    canonical = ["order_id", "customer_id", "product", "unit_price", "quantity", "order_date"]
                    num_cols = len(canonical)
                    temp_df = temp_df.iloc[:, :num_cols]
                    actual_cols = temp_df.shape[1]
                    temp_df.columns = canonical[:actual_cols]
            
            if temp_df.empty:
                continue

            # normalize columns for mapping
            temp_df.columns = [str(c).strip().lower() for c in temp_df.columns]
            
            # ID column mapping
            if "order_id" not in temp_df.columns:
                potential_id_cols = ["id", "order id", "orderid"]
                found_id = next((c for c in potential_id_cols if c in temp_df.columns), None)
                if found_id:
                    temp_df.rename(columns={found_id: "order_id"}, inplace=True)
                else:
                    temp_df.rename(columns={temp_df.columns[0]: "order_id"}, inplace=True)

            # Revenue mapping
            if "revenue" not in temp_df.columns:
                potential_rev_cols = ["amount", "price", "total", "revenue_value"]
                found_rev = next((c for c in potential_rev_cols if c in temp_df.columns), None)
                if found_rev:
                    temp_df.rename(columns={found_rev: "revenue"}, inplace=True)
            
            # Date mapping
            if "order_date" not in temp_df.columns:
                potential_date_cols = ["date", "event_timestamp", "timestamp"]
                found_date = next((c for c in potential_date_cols if c in temp_df.columns), None)
                if found_date:
                    temp_df.rename(columns={found_date: "order_date"}, inplace=True)
                else:
                    date_col = None
                    for col in temp_df.columns:
                        if temp_df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any():
                            date_col = col
                            break
                    if date_col:
                        temp_df.rename(columns={date_col: "order_date"}, inplace=True)

            cols_to_keep = ["order_id", "order_date"]
            if "revenue" in temp_df.columns: cols_to_keep.append("revenue")
            if "unit_price" in temp_df.columns: cols_to_keep.append("unit_price")
            if "quantity" in temp_df.columns: cols_to_keep.append("quantity")
            
            dfs.append(temp_df[[c for c in cols_to_keep if c in temp_df.columns]])
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
        
    raw_combined = pd.concat(dfs, ignore_index=True)
    raw_combined["timestamp"] = pd.to_datetime(raw_combined.get("order_date", pd.Timestamp.now()), errors="coerce")
    raw_combined.dropna(subset=["timestamp"], inplace=True)
    raw_combined["date"] = raw_combined["timestamp"].dt.date
    raw_combined = raw_combined.sort_values("timestamp", ascending=False).drop_duplicates(subset=["order_id"], keep="first")

    gold_style = (
        raw_combined.groupby("date")
        .agg(
            total_orders=("date", "count"),
            last_processed=("timestamp", "max"),
            order_ids=("order_id", lambda x: ", ".join(x.dropna().astype(str).unique()))
        )
        .reset_index()
    )
    
    gold_style = gold_style[["date", "order_ids", "total_orders", "last_processed"]]
    gold_style["last_processed"] = gold_style["last_processed"].dt.strftime('%Y-%m-%d %H:%M:%S')
    return gold_style.sort_values("date")

    # 3. Deduplicate (Crucial: Prevents double-counting in metrics)
    raw_combined = raw_combined.sort_values("timestamp", ascending=False).drop_duplicates(subset=["order_id"], keep="first")

    # 4. Aggregate (Match Gold Schema + New Order IDs column + Timestamp)
    gold_style = (
        raw_combined.groupby("date")
        .agg(
            total_orders=("date", "count"),
            last_processed=("timestamp", "max"),
            order_ids=("order_id", lambda x: ", ".join(x.dropna().astype(str).unique()))
        )
        .reset_index()
    )
    
    # 4. Reorder columns: replace revenue with timestamp
    cols = ["date", "order_ids", "total_orders", "last_processed"]
    gold_style = gold_style[cols]
    
    # Format timestamp for better display
    gold_style["last_processed"] = gold_style["last_processed"].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Cast for performance and consistency
    gold_style["total_orders"] = gold_style["total_orders"].astype("int32")
    
    return gold_style.sort_values("date")


def _normalize_gold_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Memory-efficient normalization for large datasets."""
    if raw_df.empty:
        return raw_df

    # 1. Column Mapping (In-place where possible to save memory)
    mapping = {
        "order_date": "date",
        "total_orders_per_day": "total_orders",
        "total_revenue_per_day": "total_revenue",
    }
    
    for old_col, new_col in mapping.items():
        if old_col in raw_df.columns:
            raw_df.rename(columns={old_col: new_col}, inplace=True)
        elif new_col not in raw_df.columns:
            # Add missing columns with optimized defaults
            raw_df[new_col] = 0.0 if "revenue" in new_col else 0

    # 2. Select only needed columns (this creates a slice, then we make it a standalone df)
    final_cols = ["date", "total_orders", "total_revenue"]
    df = raw_df[final_cols].copy() 

    # 3. Optimized Type Casting
    # Convert date efficiently
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    
    # Use smaller dtypes to save 50% memory on large numeric scales
    df["total_orders"] = pd.to_numeric(df["total_orders"], errors="coerce").fillna(0).astype("int32")
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0.0).astype("float32")

    return df.sort_values("date")


def _load_gold_with_spark(gold_path: str) -> pd.DataFrame:
    spark = create_spark_session()

    spark_df = spark.read.format("delta").load(gold_path)
    if spark_df.rdd.isEmpty():
        return pd.DataFrame()

    return _normalize_gold_dataframe(spark_df.toPandas())


def _load_gold_without_spark(gold_path: str) -> pd.DataFrame:
    try:
        from deltalake import DeltaTable  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise SparkRuntimePrecheckError(
            "Spark is unavailable and Python Delta fallback is not installed. "
            "Install it with: pip install deltalake"
        ) from exc

    delta_table = DeltaTable(gold_path)
    arrow_table: Any = delta_table.to_pyarrow_table()
    pandas_df = arrow_table.to_pandas()
    
    if pandas_df.empty:
        st.warning("Native reader returned an empty dataframe.")

    return _normalize_gold_dataframe(pandas_df)


@st.cache_data(ttl=1, show_spinner=False)
def load_gold_data(gold_path: str) -> pd.DataFrame:
    """
    Load data from the Gold Delta table. 
    Prioritizes the native 'deltalake' reader for speed and stability in Streamlit.
    """
    try:
        # Try the native Delta reader first (recommended for Streamlit/Local dashboard)
        return _load_gold_without_spark(gold_path)
    except Exception as native_exc:
        # If native reader fails, try Spark as a secondary option
        try:
            return _load_gold_with_spark(gold_path)
        except Exception as spark_exc:
            # If both fail, raise the original native error but mention Spark also failed
            raise SparkRuntimePrecheckError(
                f"Failed to load data via native reader ({str(native_exc)[:100]}) "
                f"and Spark fallback also failed ({str(spark_exc)[:100]})."
            )


def inject_shared_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #e2e8f0; }
        [data-testid="stHeader"] { background-color: rgba(15, 23, 42, 0.9); }
        h1, h2, h3, h4, p, span, div { color: #e2e8f0 !important; }
        .top-nav-wrap {
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(14, 165, 233, 0.25);
            background: linear-gradient(90deg, rgba(14,165,233,0.12), rgba(34,197,94,0.12));
        }

        .flow-card {
            position: relative;
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin: 0.7rem 0 1rem 0;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.35);
            overflow: hidden;
            isolation: isolate;
        }

        .flow-card::before {
            content: "";
            position: absolute;
            inset: -2px;
            border-radius: 18px;
            background: conic-gradient(
                from var(--angle),
                transparent 0deg,
                transparent 220deg,
                var(--line-color) 300deg,
                transparent 360deg
            );
            animation: edge-run 3.2s linear infinite;
            z-index: -1;
        }

        .flow-card::after {
            content: "";
            position: absolute;
            inset: 1px;
            border-radius: 16px;
            background: rgba(30, 41, 59, 0.85);
            z-index: -1;
        }

        .bronze { --line-color: #c27b43; }
        .validation { --line-color: #ef4444; }
        .silver { --line-color: #94a3b8; }

        @property --angle {
            syntax: '<angle>';
            inherits: false;
            initial-value: 0deg;
        }

        @keyframes edge-run {
            from { --angle: 0deg; }
            to { --angle: 360deg; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav() -> str:
    st.markdown('<div class="top-nav-wrap"><strong>Navigation</strong></div>', unsafe_allow_html=True)
    return st.radio(
        "Select view",
        ["Dashboard", "Bronze -> Validation -> Silver", "Quality Report"],
        horizontal=True,
        label_visibility="collapsed",
    )


@st.cache_data(ttl=20, show_spinner=False)
def load_delta_df(path_str: str) -> pd.DataFrame:
    from deltalake import DeltaTable  # pyright: ignore[reportMissingImports]

    return DeltaTable(path_str).to_pyarrow_table().to_pandas()


@st.cache_data(ttl=20, show_spinner=False)
def load_delta_history(path_str: str) -> pd.DataFrame:
    from deltalake import DeltaTable  # pyright: ignore[reportMissingImports]
    try:
        history_rows = DeltaTable(path_str).history()
        if not history_rows:
            return pd.DataFrame()
        history_df = pd.DataFrame(history_rows)
        
        # Select and order key columns for a "Proper" display
        important_cols = [
            "version", "timestamp", "operation", "operationParameters", 
            "userName", "isBlindAppend", "engineInfo"
        ]
        available_cols = [c for c in important_cols if c in history_df.columns]
        history_df = history_df[available_cols]
        
        if "timestamp" in history_df.columns:
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], unit='ms', errors="coerce")
            history_df["timestamp"] = history_df["timestamp"].dt.strftime('%Y-%m-%d %H:%M:%S')

        if "version" in history_df.columns:
            history_df["version"] = history_df["version"].astype(str)

        return history_df
    except Exception:
        return pd.DataFrame()


def build_validation_frame(bronze_df: pd.DataFrame, silver_df: pd.DataFrame, file_issues_df: pd.DataFrame = None) -> pd.DataFrame:
    if bronze_df.empty and (file_issues_df is None or file_issues_df.empty):
        return pd.DataFrame()

    checks: list[pd.DataFrame] = []
    
    # 0. Add File-Level Issues
    if file_issues_df is not None and not file_issues_df.empty:
        file_issues_df["Occurrences"] = 1
        checks.append(file_issues_df)

    # 1. Check for Duplicate IDs (with occurrence counting)
    id_col = "order_id" if "order_id" in bronze_df.columns else "id"
    if not bronze_df.empty and id_col in bronze_df.columns:
        # Group by ID to see how many times each exists
        counts = bronze_df[id_col].value_counts().reset_index()
        counts.columns = [id_col, "Occurrences"]
        
        # Only take IDs that appear more than once
        dups_ref = counts[counts["Occurrences"] > 1]
        
        if not dups_ref.empty:
            # Join back to get the sample row data
            dup_rows = bronze_df[bronze_df[id_col].isin(dups_ref[id_col])].drop_duplicates(subset=[id_col])
            dup_merged = dup_rows.merge(dups_ref, on=id_col)
            dup_merged["Reason"] = dup_merged.apply(
                lambda x: f"Duplicate Error: Found {x['Occurrences']} entries for this ID. {x['Occurrences']-1} will be filtered.", axis=1
            )
            checks.append(dup_merged)

    # 2. Check for Missing/Incomplete Data
    if not bronze_df.empty:
        critical_cols = ["id", "name", "order_id", "order_date"]
        for col in critical_cols:
            if col in bronze_df.columns:
                null_mask = bronze_df[col].isna() | (bronze_df[col].astype(str) == "nan") | (bronze_df[col].astype(str) == "")
                null_rows = bronze_df[null_mask].copy()
                if not null_rows.empty:
                    null_rows["Occurrences"] = 1
                    null_rows["Reason"] = f"Validation Failed: {col} is missing or null"
                    checks.append(null_rows)

    # 3. Check for Negative Values (Quality Check)
    if not bronze_df.empty:
        numeric_cols = ["quantity", "unit_price", "revenue", "price"]
        for col in numeric_cols:
            if col in bronze_df.columns:
                try:
                    neg_mask = pd.to_numeric(bronze_df[col], errors="coerce") < 0
                    neg_rows = bronze_df[neg_mask].copy()
                    if not neg_rows.empty:
                        neg_rows["Occurrences"] = 1
                        neg_rows["Reason"] = f"Quality Alert: Negative value detected in {col}"
                        checks.append(neg_rows)
                except: pass

    # 4. Check for Future Dates
    if not bronze_df.empty and "order_date" in bronze_df.columns:
        try:
            dates = pd.to_datetime(bronze_df["order_date"], errors="coerce")
            future_mask = dates > pd.Timestamp.now() + pd.Timedelta(days=1)
            future_rows = bronze_df[future_mask].copy()
            if not future_rows.empty:
                future_rows["Occurrences"] = 1
                future_rows["Reason"] = "Temporal Error: Record date is set in the future"
                checks.append(future_rows)
        except: pass

    if not checks:
        return pd.DataFrame()

    # Combine all identified issues
    result = pd.concat(checks, ignore_index=True)
    
    preferred_cols = [
        c for c in ["order_id", "Occurrences", "source_file", "product", "order_date", "Reason"]
        if c in result.columns
    ]
    return result[preferred_cols] if preferred_cols else result


@st.dialog("Data Rejection Report")
def show_rejection_dialog(row_data: pd.Series) -> None:
    st.markdown(f"### 🔍 Analysis for Order ID: `{row_data.get('order_id', 'N/A')}`")
    st.divider()
    
    st.error(f"**Status:** REJECTED / MODIFIED")
    st.info(f"**Reason:** {row_data['Reason']}")
    
    st.markdown("#### Full Record Details")
    # Clean up for json display
    clean_row = row_data.to_dict()
    st.json(clean_row)
    
    if st.button("Close Report", use_container_width=True):
        st.rerun()


def open_flow_card(step_title: str, subtitle: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="flow-card {tone}">
            <h3 style="margin:0; color:#e2e8f0;">{step_title}</h3>
            <p style="margin:0.25rem 0 0.5rem 0; color:#94a3b8;">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def close_flow_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_medallion_section() -> None:
    script_dir = Path(__file__).resolve().parent
    bronze_path = script_dir / "data" / "bronze"
    silver_path = script_dir / "data" / "silver"
    quarantine_path = script_dir / "data" / "quarantine"
    input_path = script_dir / "data" / "input"

    # HYBRID ENGINE: Load real-time data first to avoid "Empty" errors
    rt_bronze, rt_silver, file_issues_df = _compute_medallion_realtime(str(input_path))

    # Try loading Delta Histories if they exist (for the ACID part)
    bronze_history = pd.DataFrame()
    silver_history = pd.DataFrame()
    
    if bronze_path.exists():
        try: bronze_history = load_delta_history(str(bronze_path))
        except: pass
    if silver_path.exists():
        try: silver_history = load_delta_history(str(silver_path))
        except: pass

    # --- STEP 1: BRONZE ---
    open_flow_card(
        "Step 1: Bronze Layer (Raw Ingested)",
        "Untouched source records. This layer captures everything, including mistakes and duplicates.",
        "bronze",
    )
    display_bronze = rt_bronze if not rt_bronze.empty else pd.DataFrame()
    
    if not display_bronze.empty:
        st.metric("Total Raw Records", f"{len(display_bronze)}")
        pending_row = pd.DataFrame([{
            "version": "Pending",
            "timestamp": "Real-time",
            "operation": "INGESTION",
            "operationParameters": "Scanning raw CSVs",
            "userName": "System",
            "isBlindAppend": True,
            "engineInfo": "Hybrid RT"
        }])
        bronze_history = pd.concat([pending_row, bronze_history], ignore_index=True)
        if len(display_bronze) > 1000:
            st.info(f"Showing top 1000 of {len(display_bronze):,} records.")
            st.dataframe(display_bronze.head(1000), use_container_width=True)
        else:
            st.dataframe(display_bronze, use_container_width=True)
    else:
        st.warning("Bronze table is empty. Add data to `data/input/`.")

    with st.expander("📝 View Full Bronze Transaction History"):
        if not bronze_history.empty:
            st.dataframe(bronze_history, use_container_width=True)
        else:
            st.info("No history available.")
    close_flow_card()

    # --- STEP 2: VALIDATION ---
    open_flow_card(
        "Step 2: Data Quality & Validation",
        "The logic gate that identifies duplicates and errors before they reach your charts.",
        "validation",
    )
    
    if display_bronze.empty and (file_issues_df is None or file_issues_df.empty):
        st.info("Waiting for data...")
    else:
        validation_df = build_validation_frame(display_bronze, rt_silver, file_issues_df)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.metric("Unique Issue IDs", len(validation_df))
        with col_v2:
            # TRUE DUPLICATE COUNT: Difference between Bronze Row count and Silver Row count
            actual_dups = len(display_bronze) - len(rt_silver) if not display_bronze.empty else 0
            st.metric("Duplicate Records Filtered", actual_dups, delta="Filtered", delta_color="inverse")

        if validation_df.empty:
            st.success("✅ Clean Sweep: No duplicates or errors detected.")
        else:
            st.markdown("### 🛠️ Quality Rejection Log")
            st.markdown("These records will **not** be counted in your Silver/Gold layers:")
            event = st.dataframe(
                validation_df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True
            )

            if event.selection.rows:
                show_rejection_dialog(validation_df.iloc[event.selection.rows[0]])
    close_flow_card()

    # --- STEP 2.5: QUARANTINE ---
    open_flow_card(
        "Step 2.5: Quarantine (Pipeline-Enforced Rejections)",
        "Records blocked by the Spark pipeline before reaching Silver. Written to a Delta table with rejection reasons.",
        "validation",
    )
    quarantine_df = pd.DataFrame()
    if quarantine_path.exists():
        try:
            quarantine_df = load_delta_df(str(quarantine_path))
        except Exception:
            pass

    if quarantine_df.empty:
        st.info("No quarantined records yet. Run the pipeline from Airflow to enforce hard validation.")
    else:
        reason_col = "rejection_reason"
        cols_q = st.columns(2)
        with cols_q[0]:
            st.metric("Quarantined Records", len(quarantine_df))
        with cols_q[1]:
            if reason_col in quarantine_df.columns:
                breakdown = quarantine_df[reason_col].value_counts().to_dict()
                st.metric("Distinct Rejection Reasons", len(breakdown))

        if reason_col in quarantine_df.columns:
            st.markdown("**Rejection Breakdown**")
            breakdown_df = (
                quarantine_df[reason_col]
                .value_counts()
                .rename_axis("reason")
                .reset_index(name="count")
            )
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown("**Rejected Records (sample)**")
        st.dataframe(quarantine_df.head(50), use_container_width=True)
    close_flow_card()

    # --- STEP 3: SILVER ---
    open_flow_card(
        "Step 3: Silver Layer (Cleaned Data)",
        "The final source of truth. Duplicates have been removed and data is ready for use.",
        "silver",
    )
    
    display_silver = rt_silver if not rt_silver.empty else pd.DataFrame()

    if not display_silver.empty:
        st.metric("Unique Records", len(display_silver), help="Duplicates are filtered out here.")
        pending_row_silver = pd.DataFrame([{
            "version": "Pending",
            "timestamp": "Real-time",
            "operation": "CLEAN/DEDUP",
            "operationParameters": "Applying Silver rules",
            "userName": "System",
            "isBlindAppend": True,
            "engineInfo": "Hybrid RT"
        }])
        silver_history = pd.concat([pending_row_silver, silver_history], ignore_index=True)
        if len(display_silver) > 1000:
            st.info(f"Showing top 1000 of {len(display_silver):,} records.")
            st.dataframe(display_silver.head(1000), use_container_width=True)
        else:
            st.dataframe(display_silver, use_container_width=True)
    else:
        st.warning("Silver table is empty. Records transition here after validation.")

    with st.expander("📝 View Full Silver Transaction History"):
        if not silver_history.empty:
            st.dataframe(silver_history, use_container_width=True)
        else:
            st.info("No history available.")
    close_flow_card()


def render_time_travel_demo() -> None:
    st.markdown("### 🕒 Delta Lake Time-Travel Audit")
    st.markdown("Track how your data quality evolves over time. This chart shows the **Silver (Accepted)** vs **Quarantine (Rejected)** records for the last 5 updates.")
    
    script_dir = Path(__file__).resolve().parent
    silver_path = str(script_dir / "data" / "silver")
    quarantine_path = str(script_dir / "data" / "quarantine")
    
    try:
        from deltalake import DeltaTable
        if not os.path.exists(silver_path):
            st.info("Silver table not found. Run the pipeline to see history.")
            return

        silver_dt = DeltaTable(silver_path)
        silver_hist = silver_dt.history()
        
        # Last 5 versions
        recent_hist = silver_hist[:5]
        
        quar_dt = None
        quar_hist_df = pd.DataFrame()
        try:
            if os.path.exists(quarantine_path):
                quar_dt = DeltaTable(quarantine_path)
                quar_hist_df = pd.DataFrame(quar_dt.history())
        except: pass
        
        plot_data = []
        for h in recent_hist:
            v = h["version"]
            ts = h["timestamp"]
            
            # Load Silver count
            silver_dt.load_as_version(int(v))
            s_count = len(silver_dt.to_pyarrow_table())
            
            # Load Quarantine count at that time
            q_count = 0
            if quar_dt is not None and not quar_hist_df.empty:
                # Sync with quarantine version at that timestamp (with 3-minute grace period for multi-stage jobs)
                valid_q = quar_hist_df[quar_hist_df['timestamp'] <= ts + 180000]
                if not valid_q.empty:
                    q_v = int(valid_q.iloc[0]['version'])
                    quar_dt.load_as_version(q_v)
                    q_count = len(quar_dt.to_pyarrow_table())
            
            plot_data.append({
                "Version": f"v{v}",
                "Time": pd.to_datetime(ts, unit='ms').strftime('%H:%M:%S'),
                "Silver (Clean)": s_count,
                "Rejected": q_count
            })
            
        if not plot_data:
            st.info("No version history available.")
            return

        df_plot = pd.DataFrame(plot_data).iloc[::-1] # Oldest to Newest
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_plot["Version"] + "<br>" + df_plot["Time"],
            y=df_plot["Silver (Clean)"],
            name="Silver (Clean)",
            marker_color="#2ecc71" # Green
        ))
        fig.add_trace(go.Bar(
            x=df_plot["Version"] + "<br>" + df_plot["Time"],
            y=df_plot["Rejected"],
            name="Rejected",
            marker_color="#e74c3c" # Red
        ))
        
        chart_config = {'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#94a3b8'}}
        fig.update_layout(
            **chart_config,
            barmode='stack',
            margin=dict(l=0, r=0, b=0, t=30),
            height=350,
            xaxis_title="Pipeline Update Version",
            yaxis_title="Record Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.caption(f"Waiting for history logs... ({str(e)[:50]})")


def render_dashboard_home() -> None:
    st.title("Data Pipeline Analytics Dashboard")

    top_col1, top_col2 = st.columns([6, 1])
    with top_col2:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            if "engine_notified" in st.session_state:
                del st.session_state["engine_notified"]
            st.rerun()

    # Ultra-fast auto-refresh: every 1 second
    st.markdown(
        """
        <script>
        if (!window.refreshIntervalSet) {
            window.refreshIntervalSet = true;
            setInterval(function() {
                window.location.reload();
            }, 1000);
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(script_dir, "data", "gold")

    raw_input_path = os.path.join(script_dir, "data", "input")
    
    # PERFORMANCE BOOST: Use Hybrid Real-time View
    # We calculate the Gold dataset directly from raw files in memory for 0-latency feedback
    gold_df = _compute_gold_from_raw(raw_input_path)
    
    if gold_df.empty:
        st.warning("No data found in input folder. Please add CSV files to `data/input/`.")
        return

    # --- Partition Pruning Simulation (Date Filter) ---
    st.subheader("Simulate Partition Pruning")
    st.markdown("Use this filter to simulate how **Partition Pruning** works. By selecting specific dates, the query only scans the relevant partitions (`order_date=YYYY-MM-DD`) instead of running a full table scan.")
    
    unique_dates = sorted(gold_df["date"].unique())
    if not unique_dates:
        st.warning("No dates found in data.")
        return
        
    min_date = unique_dates[0]
    max_date = unique_dates[-1]
    
    # Let user select a date range
    selected_dates = st.date_input(
        "Select Date Range (Simulates reading specific partition folders):",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle single date vs date range selection gracefully
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif isinstance(selected_dates, tuple) and len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date = end_date = selected_dates
        
    filtered_df = gold_df[(gold_df["date"] >= start_date) & (gold_df["date"] <= end_date)]

    if "engine_notified" not in st.session_state:
        st.toast("⚡ **Ultra-Fast Real-Time Engine Active**", icon="🚀")
        st.session_state["engine_notified"] = True

    total_orders = int(filtered_df["total_orders"].sum())
    
    # Calculate partition metrics
    partitions_scanned = sum(1 for d in unique_dates if start_date <= d <= end_date)
    total_partitions = len(unique_dates)
    scan_percentage = (partitions_scanned / total_partitions) * 100 if total_partitions > 0 else 0

    st.markdown("---")
    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Total Living Orders (Filtered)", f"{total_orders:,}")
    with colB:
        # Data Partitioning Div (Visual representation of pruning effect)
        st.markdown(f"""
            <div style='background: rgba(15, 23, 42, 0.4); padding: 15px; border-radius: 10px; border: 1px solid #1e293b; display: flex; align-items: center; gap: 20px;'>
                <div style='flex: 1;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span style='color: #64748b; font-size: 0.8rem;'>Partition Scans</span>
                        <span style='color: #38bdf8; font-size: 0.8rem; font-weight: bold;'>{100-scan_percentage:.0f}% Savings</span>
                    </div>
                    <div style='background: #334155; height: 6px; border-radius: 3px; overflow: hidden;'>
                        <div style='width: {scan_percentage}%; background: #38bdf8; height: 100%; box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);'></div>
                    </div>
                    <div style='margin-top: 8px; color: #94a3b8; font-size: 0.75rem;'>
                        Querying <b>{partitions_scanned}</b> / {total_partitions} partition folders
                    </div>
                </div>
                <div style='text-align: center; border-left: 1px solid #334155; padding-left: 20px;'>
                    <div style='color: #22c55e; font-size: 0.7rem; font-weight: bold; margin-bottom: 2px;'>PRUNING STATUS</div>
                    <div style='color: #e2e8f0; font-size: 1.1rem; font-weight: 800;'>{"PASS" if scan_percentage < 100 else "ACTIVE"}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Real-Time Daily Orders")
    if not filtered_df.empty:
        import plotly.express as px
        chart_config = {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#94a3b8'}
        }
        fig_orders = px.line(filtered_df, x='date', y='total_orders', markers=True)
        fig_orders.update_traces(line_color='#38bdf8', line_width=3, marker=dict(size=8, color='#38bdf8'))
        fig_orders.update_layout(
            **chart_config, 
            margin=dict(l=0, r=0, b=0, t=10), 
            xaxis_title='', 
            yaxis_title='', 
            yaxis_gridcolor='#1e293b', 
            xaxis_gridcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_orders, use_container_width=True)
        
        # --- Adding Donut and Bar Chart sections matching user request ---
        st.markdown("<br>", unsafe_allow_html=True)
        cA, cB = st.columns(2)
        
        with cA:
            st.markdown("### Processed Flow")
            # Retrieve real-time metrics for Gold, Silver, and Rejected/Bronze
            silver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "silver")
            try:
                # To simulate the Medallion funnel: we'll get real-time raw counts vs accepted silver counts
                raw_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input")
                rt_bronze, rt_silver, _ = _compute_medallion_realtime(raw_input)
                
                bronze_count = len(rt_bronze)
                silver_count = len(rt_silver)
                rejected_count = max(0, bronze_count - silver_count)
                
                # If we have no data, fallback to dummy
                if bronze_count == 0:
                    status_data = pd.DataFrame({'status': ['Empty'], 'count': [1]})
                else:
                    status_data = pd.DataFrame({
                        'status': ['Bronze', 'Rejected', 'Silver'],
                        'count': [bronze_count, rejected_count, silver_count]
                    })
                    
                fig_donut = px.pie(status_data, values='count', names='status', hole=0.6, 
                                 color_discrete_sequence=['#475569', '#ef4444', '#3b82f6'])
                fig_donut.update_layout(**chart_config, margin=dict(l=0, r=0, b=0, t=10), showlegend=False)
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_donut, use_container_width=True)
            except Exception:
                st.info("Funnel data unavailable")
                
        with cB:
            st.markdown("### Month-on-Month Order Growth")
            # Always aggregate from the FULL dataset and take the last 3 months
            mom_src = gold_df.copy()
            mom_src['_ym'] = pd.to_datetime(mom_src['date']).dt.to_period('M')
            monthly_all = (
                mom_src.groupby('_ym', as_index=False)['total_orders'].sum()
                .sort_values('_ym')
                .reset_index(drop=True)
            )

            # Exactly 3 months — always (pad with available data if fewer than 3 exist)
            three_months = monthly_all.tail(3).reset_index(drop=True)

            # Prepend prior month just to compute first bar's growth %
            if not three_months.empty:
                first_ym = three_months['_ym'].iloc[0]
                prior = monthly_all[monthly_all['_ym'] < first_ym].tail(1)
                calc_df = pd.concat([prior, three_months], ignore_index=True)
            else:
                calc_df = three_months.copy()

            calc_df['growth_pct'] = calc_df['total_orders'].pct_change() * 100
            calc_df['month_label'] = calc_df['_ym'].dt.strftime('%b %Y')

            # Keep only the 3 display months (drop the prepended prior row)
            display_months = calc_df.iloc[-3:].reset_index(drop=True)

            def _growth_label(v):
                if pd.isna(v):
                    return ""
                return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

            display_months = display_months.copy()
            display_months['growth_text'] = display_months['growth_pct'].apply(_growth_label)
            # Show orders count inside bar + growth % outside
            display_months['bar_text'] = (
                display_months['total_orders'].astype(str)
                + "<br><span style='font-size:11px'>"
                + display_months['growth_text']
                + "</span>"
            )

            bar_colors = [
                '#22c55e' if (pd.notna(v) and v >= 0) else '#ef4444'
                if pd.notna(v) else '#3b82f6'
                for v in display_months['growth_pct']
            ]

            import plotly.graph_objects as go
            fig_bar = go.Figure()
            for i, row in display_months.iterrows():
                growth_v = row['growth_pct']
                color = '#22c55e' if (pd.notna(growth_v) and growth_v >= 0) else ('#ef4444' if pd.notna(growth_v) else '#3b82f6')
                fig_bar.add_trace(go.Bar(
                    x=[row['month_label']],
                    y=[row['total_orders']],
                    name=row['month_label'],
                    marker_color=color,
                    marker_line_color='rgba(0,0,0,0)',
                    opacity=0.88,
                    width=0.45,
                    text=[f"{int(row['total_orders']):,}"],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='#ffffff', size=14, family='Arial Black'),
                    showlegend=False,
                ))
                # Growth % annotation above each bar
                if row['growth_text']:
                    fig_bar.add_annotation(
                        x=row['month_label'],
                        y=row['total_orders'],
                        text=row['growth_text'],
                        showarrow=False,
                        yanchor='bottom',
                        yshift=8,
                        font=dict(color='#e2e8f0', size=13, family='Arial'),
                    )

            fig_bar.update_layout(
                **chart_config,
                margin=dict(l=10, r=10, b=10, t=40),
                xaxis=dict(title='', tickfont=dict(color='#94a3b8', size=13), gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(title='', gridcolor='#1e293b', tickfont=dict(color='#94a3b8')),
                bargroupgap=0.3,
                height=260,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # --- NEW: Delta Lake Time-Travel Demo Section ---
        st.divider()
        render_time_travel_demo()
        
        with st.expander("🔍 Auditing & Rollback Demonstration"):
            st.markdown("""
            **Technical Capabilities Demonstrated Above:**
            - **Historical Auditing**: Every bar represents a distinct transaction version in Delta Lake.
            - **Data Drift Detection**: Sudden spikes in the <span style='color:#e74c3c'>Red (Rejected)</span> bars indicate schema changes or upstream data quality issues.
            - **Zero-Copy Rollback**: You can restore the table to any previous version instantly using:
              ```sql
              RESTORE TABLE silver_table TO VERSION AS OF <version_id>
              ```
            - **Debug Pattern**: By querying `versionAsOf(v)`, we can compare record-level differences between runs to identify specific corrupted rows that caused a pipeline failure.
            """, unsafe_allow_html=True)
        st.divider()
            
    else:
        st.info("No orders found in the selected date range.")

    st.subheader("Gold Layer Dataset (Live Preview)")
    if len(filtered_df) > 1000:
        st.info(f"Showing top 1000 of {len(filtered_df):,} rows.")
        st.dataframe(filtered_df.head(1000), use_container_width=True)
    else:
        st.dataframe(filtered_df, use_container_width=True)

    st.caption(f"Last heartbeat at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _run_live_quality_checks(bronze_df: pd.DataFrame, silver_df: pd.DataFrame, file_issues_df: pd.DataFrame = None) -> dict:
    """
    Run all data quality checks live against the current Bronze DataFrame.
    Returns a report dict structured identically to what quality_report.json used to provide,
    so the rendering logic below is reusable for both sources.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layers_out = []

    # ── 0. Input Folder Integrity ───────────────────────────────────────────
    file_checks = []
    file_count = 0
    if file_issues_df is not None:
        file_count = len(file_issues_df)
        unsupported = file_issues_df[file_issues_df["Reason"].str.contains("Unsupported", na=False)]
        corrupted = file_issues_df[file_issues_df["Reason"].str.contains("Corrupted", na=False)]
        
        file_checks.append({
            "name": "File Formats",
            "status": "PASS" if unsupported.empty else "WARNING",
            "message": "All files use supported .csv format." if unsupported.empty 
                       else f"{len(unsupported)} file(s) follow unsupported formats."
        })
        file_checks.append({
            "name": "Data Integrity",
            "status": "PASS" if corrupted.empty else "CRITICAL",
            "message": "No corrupted CSVs detected." if corrupted.empty 
                       else f"{len(corrupted)} corrupted file(s) found in input folder."
        })

    layers_out.append({
        "layer": "input",
        "row_count": file_count,
        "columns": ["file_name", "reason"],
        "checks": file_checks,
        "critical_failures": sum(1 for c in file_checks if c["status"] == "CRITICAL"),
        "warnings": sum(1 for c in file_checks if c["status"] == "WARNING"),
        "status": "FAIL" if any(c["status"] == "CRITICAL" for c in file_checks) else ("WARN" if any(c["status"] == "WARNING" for c in file_checks) else "PASS")
    })

    for layer_name, df in [("bronze", bronze_df), ("silver", silver_df)]:
        checks = []
        row_count = len(df)
        cols_present = list(df.columns)

        # ── 1. Row Count check ─────────────────────────────────────────────
        checks.append({
            "name":    "Row Count",
            "status":  "PASS" if row_count > 0 else "CRITICAL",
            "message": f"{row_count:,} rows found." if row_count > 0 else "Table is empty — no data loaded.",
        })

        # ── 2. Schema / required columns ──────────────────────────────────
        required = ["order_id", "order_date"]
        missing_cols = [c for c in required if c not in df.columns]
        checks.append({
            "name":    "Required Columns Present",
            "status":  "PASS" if not missing_cols else "CRITICAL",
            "message": "All required columns found." if not missing_cols
                       else f"Missing columns: {', '.join(missing_cols)}",
        })

        if df.empty:
            layers_out.append({
                "layer": layer_name, "row_count": 0, "columns": cols_present,
                "checks": checks, "critical_failures": 1, "warnings": 0, "status": "FAIL",
            })
            continue

        # ── 3. Duplicate IDs ───────────────────────────────────────────────
        if "order_id" in df.columns:
            dup_count = int(df.duplicated(subset=["order_id"], keep=False).sum())
            dup_ids   = df.loc[df.duplicated(subset=["order_id"], keep=False), "order_id"].unique().tolist()
            sample    = ", ".join(str(x) for x in dup_ids[:5])
            suffix    = f" … and {len(dup_ids)-5} more" if len(dup_ids) > 5 else ""
            checks.append({
                "name":    "No Duplicate Order IDs",
                "status":  "PASS" if dup_count == 0 else "WARNING",
                "message": f"No duplicates found." if dup_count == 0
                           else f"{dup_count} duplicate rows for IDs: {sample}{suffix}",
            })

        # ── 4. Null / empty critical fields ───────────────────────────────
        null_report = []
        for col in required:
            if col in df.columns:
                n = int(df[col].isna().sum()) + int((df[col].astype(str).str.strip() == "").sum())
                if n:
                    null_report.append(f"{col}: {n} null/empty")
        checks.append({
            "name":    "No Null Critical Fields",
            "status":  "PASS" if not null_report else "CRITICAL",
            "message": "All critical fields populated." if not null_report
                       else " | ".join(null_report),
        })

        # ── 5. Negative numeric values ────────────────────────────────────
        neg_issues = []
        for col in ["unit_price", "quantity", "revenue"]:
            if col in df.columns:
                neg = int((pd.to_numeric(df[col], errors="coerce").fillna(0) < 0).sum())
                if neg:
                    neg_issues.append(f"{col}: {neg} negative value(s)")
        checks.append({
            "name":    "No Negative Numeric Values",
            "status":  "PASS" if not neg_issues else "WARNING",
            "message": "All numeric values are non-negative." if not neg_issues
                       else " | ".join(neg_issues),
        })

        # ── 6. Date parse errors ──────────────────────────────────────────
        if "order_date" in df.columns:
            parsed = pd.to_datetime(df["order_date"], errors="coerce")
            bad_dates = int(parsed.isna().sum())
            checks.append({
                "name":    "Valid Date Format",
                "status":  "PASS" if bad_dates == 0 else "WARNING",
                "message": "All dates parse correctly." if bad_dates == 0
                           else f"{bad_dates} row(s) have unparsable dates.",
            })

            # ── 7. Future dates ───────────────────────────────────────────
            future = int((parsed > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
            checks.append({
                "name":    "No Future Dates",
                "status":  "PASS" if future == 0 else "WARNING",
                "message": "No future dates detected." if future == 0
                           else f"{future} row(s) have dates set in the future.",
            })

        # ── 8. Silver dedup effectiveness (only for silver) ───────────────
        if layer_name == "silver" and not bronze_df.empty and "order_id" in df.columns:
            removed = len(bronze_df) - len(df)
            pct     = (removed / len(bronze_df) * 100) if len(bronze_df) else 0
            checks.append({
                "name":    "Deduplication Applied",
                "status":  "PASS",
                "message": f"{removed} duplicate row(s) removed ({pct:.1f}% of Bronze). "
                           f"{len(df):,} unique records remain.",
            })

        criticals = sum(1 for c in checks if c["status"] == "CRITICAL")
        warnings  = sum(1 for c in checks if c["status"] == "WARNING")
        layer_status = "FAIL" if criticals else ("WARN" if warnings else "PASS")

        layers_out.append({
            "layer":             layer_name,
            "row_count":         row_count,
            "columns":           cols_present,
            "checks":            checks,
            "critical_failures": criticals,
            "warnings":          warnings,
            "status":            layer_status,
        })

    # Overall status
    all_statuses = [l["status"] for l in layers_out]
    overall = "FAIL" if "FAIL" in all_statuses else ("WARN" if "WARN" in all_statuses else "PASS")

    return {
        "overall_status": overall,
        "generated_at":   now_str,
        "source":         "live",
        "layers":         layers_out,
    }


@st.cache_data(ttl=2, show_spinner=False)
def _cached_quality_report(input_path: str) -> dict:
    """Cached wrapper so the report recalculates at most every 2 seconds."""
    bronze_df, silver_rt, file_issues_df = _compute_medallion_realtime(input_path)
    return _run_live_quality_checks(bronze_df, silver_rt, file_issues_df)


def render_quality_report() -> None:
    st.title("📋 Data Quality & Validation Report")
    st.caption("Live report — recalculates automatically as your CSV data changes.")

    script_dir = Path(__file__).resolve().parent
    input_path = str(script_dir / "data" / "input")

    # ── Run live checks ───────────────────────────────────────────────────────
    report = _cached_quality_report(input_path)

    overall      = report["overall_status"]
    generated_at = report["generated_at"]
    layers       = report["layers"]

    # ── Overall status banner ─────────────────────────────────────────────────
    _oc  = {"PASS": "#22c55e", "WARN": "#f97316", "FAIL": "#ef4444"}.get(overall, "#94a3b8")
    _oi  = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(overall, "❓")
    _obg = {"PASS": "rgba(34,197,94,0.12)", "WARN": "rgba(249,115,22,0.12)",
            "FAIL": "rgba(239,68,68,0.12)"}.get(overall, "rgba(148,163,184,0.1)")

    st.markdown(
        f"""
        <div style="border-radius:12px; padding:1rem 1.5rem; margin-bottom:1.2rem;
                    background:{_obg}; border:1.5px solid {_oc};">
            <span style="font-size:2rem; font-weight:800; color:{_oc};">
                {_oi} {overall}
            </span>
            <span style="margin-left:1.5rem; color:#94a3b8; font-size:0.9rem;">
                Last computed: {generated_at}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not layers:
        st.warning("No data found in `data/input/`. Add CSV files to see the report.")
        return

    # ── Summary layer cards ───────────────────────────────────────────────────
    cols = st.columns(len(layers))
    for col, layer in zip(cols, layers):
        s     = layer["status"]
        color = {"PASS": "#22c55e", "WARN": "#f97316", "FAIL": "#ef4444"}.get(s, "#94a3b8")
        icon  = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(s, "❓")
        with col:
            st.markdown(
                f"""
                <div style="border-radius:12px; padding:1rem; text-align:center;
                            border:1.5px solid {color}; background:rgba(0,0,0,0.05);">
                    <div style="font-size:1.6rem; font-weight:800; color:{color};">{icon}</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin:4px 0;">
                        {layer['layer'].upper()}
                    </div>
                    <div style="color:#94a3b8; font-size:0.85rem;">{layer['row_count']:,} rows</div>
                    <div style="color:#ef4444; font-size:0.8rem; margin-top:4px;">
                        {layer['critical_failures']} critical
                    </div>
                    <div style="color:#f97316; font-size:0.8rem;">{layer['warnings']} warnings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Aggregate metrics row ─────────────────────────────────────────────────
    total_rows    = sum(l["row_count"] for l in layers)
    total_crit    = sum(l["critical_failures"] for l in layers)
    total_warn    = sum(l["warnings"] for l in layers)
    total_checks  = sum(len(l["checks"]) for l in layers)
    total_pass    = sum(
        sum(1 for c in l["checks"] if c["status"] == "PASS") for l in layers
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows Checked",   f"{total_rows:,}")
    m2.metric("Checks Run",           f"{total_checks}")
    m3.metric("✅ Passing",            f"{total_pass}")
    m4.metric("Issues Found",         f"{total_crit + total_warn}",
              delta=f"{total_crit} critical" if total_crit else None,
              delta_color="inverse")

    st.divider()

    # ── Per-layer expandable check tables ─────────────────────────────────────
    def _color_status(val: str) -> str:
        return {
            "PASS":     "color: #22c55e; font-weight:600",
            "WARNING":  "color: #f97316; font-weight:600",
            "CRITICAL": "color: #ef4444; font-weight:700",
            "WARN":     "color: #f97316; font-weight:600",
        }.get(val, "")

    for layer in layers:
        lname = layer["layer"].upper()
        lstatus = layer["status"]
        lrows   = layer["row_count"]

        with st.expander(
            f"{lname} — {lrows:,} rows — {lstatus}",
            expanded=True,
        ):
            # Columns present
            st.caption(f"**Columns:** {', '.join(layer.get('columns', []))}")

            checks = layer.get("checks", [])
            if not checks:
                st.info("No checks available.")
                continue

            check_rows = []
            for c in checks:
                icon = {"PASS": "✅", "WARNING": "⚠️", "CRITICAL": "❌", "WARN": "⚠️"}.get(
                    c["status"], "❓"
                )
                check_rows.append({
                    "":       icon,
                    "Check":  c["name"],
                    "Status": c["status"],
                    "Detail": c["message"],
                })

            check_df = pd.DataFrame(check_rows)
            st.dataframe(
                check_df.style.map(_color_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ── Duplicate detail table ────────────────────────────────────────────────
    bronze_df, silver_df, file_issues_df = _compute_medallion_realtime(input_path)
    if not bronze_df.empty or (file_issues_df is not None and not file_issues_df.empty):
        st.subheader("🔍 Duplicate & Invalid File Detail")
        dup_df = build_validation_frame(bronze_df, silver_df, file_issues_df)
        if dup_df.empty:
            st.success("✅ No duplicate or invalid records found in the current dataset.")
        else:
            dup_count = len(dup_df[dup_df["Reason"].str.contains("Duplicate", na=False)])
            st.markdown(
                f"**{len(dup_df)} records flagged** — "
                f"**{dup_count} duplicates**, "
                f"**{len(dup_df) - dup_count} other issues**"
            )
            st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.caption(f"⚡ Report auto-refreshes. Last generated: {generated_at}")




def render_email_sidebar() -> None:
    """Sidebar: live pipeline status banner + Resend email controls."""
    with st.sidebar:

        # ── 1. LIVE PIPELINE STATUS BANNER ───────────────────────────────────
        st.markdown("## 🔴 Pipeline Status")
        st.caption("Live status from last Airflow run")

        pipeline_status: dict | None = None
        try:
            from utils.pipeline_status import read_status
            pipeline_status = read_status()
        except Exception:
            pass

        if pipeline_status is None:
            st.info("ℹ️ No pipeline run recorded yet.")
        else:
            s        = pipeline_status.get("status", "UNKNOWN")
            dag_id   = pipeline_status.get("dag_id", "—")
            task_id  = pipeline_status.get("task_id", "—")
            ts       = pipeline_status.get("timestamp", "—")
            msg      = pipeline_status.get("message", "")
            exc_text = pipeline_status.get("exception", "")
            attempt  = pipeline_status.get("try_number", "—")

            _color = {"SUCCESS": "#22c55e", "ERROR": "#ef4444",
                      "RUNNING": "#3b82f6"}.get(s, "#94a3b8")
            _icon  = {"SUCCESS": "✅", "ERROR": "❌", "RUNNING": "⏳"}.get(s, "❓")

            st.markdown(
                f"""
                <div style="border-radius:10px; padding:10px 14px;
                            background:{'rgba(34,197,94,0.12)'  if s=='SUCCESS' else
                                        'rgba(239,68,68,0.12)'  if s=='ERROR'   else
                                        'rgba(59,130,246,0.12)' if s=='RUNNING' else
                                        'rgba(148,163,184,0.12)'};
                            border:1.5px solid {_color}; margin-bottom:6px;">
                    <span style="font-size:1.1rem;font-weight:700;color:{_color};">
                        {_icon} {s}
                    </span><br/>
                    <span style="font-size:0.78rem;color:#64748b;">
                        DAG: {dag_id}<br/>
                        Task: {task_id}<br/>
                        {'Attempt: ' + str(attempt) + '<br/>' if s == 'ERROR' else ''}
                        At: {ts}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if msg:
                st.caption(f"📝 {msg}")

            # Show full traceback when ERROR
            if s == "ERROR" and exc_text:
                with st.expander("🔍 View Full Error Traceback", expanded=True):
                    st.code(exc_text, language="python")

                # One-click send error report
                api_key_q     = os.environ.get("RESEND_API_KEY", "").strip()
                owner_email_q = os.environ.get("OWNER_EMAIL", "").strip()
                if st.button("📨 Re-send Error Report", use_container_width=True,
                             disabled=not (api_key_q and owner_email_q)):
                    with st.spinner("Sending error report…"):
                        try:
                            from utils.notifications import send_pipeline_notification
                            details_full = (
                                f"DAG    : {dag_id}\n"
                                f"Task   : {task_id}\n"
                                f"Attempt: {attempt}\n"
                                f"At     : {ts}\n\n"
                                f"─── Traceback ───\n{exc_text}"
                            )
                            ok = send_pipeline_notification(
                                status  = "ERROR",
                                message = msg or f"Task '{task_id}' failed.",
                                details = details_full,
                            )
                            st.success("✅ Error report sent!") if ok else st.error("❌ Failed.")
                        except Exception as exc:
                            st.error(f"❌ {exc}")

        st.divider()

        # ── 2. EMAIL CONFIG STATUS ────────────────────────────────────────────
        st.markdown("## 📧 Email Notifications")
        st.caption("Powered by Resend API")

        try:
            from utils.notifications import _load_env
            _load_env()
        except Exception:
            pass

        api_key     = os.environ.get("RESEND_API_KEY", "").strip()
        owner_email = os.environ.get("OWNER_EMAIL", "").strip()

        if api_key:
            st.success("✅ API Key configured")
        else:
            st.error("❌ RESEND_API_KEY not set")

        if owner_email:
            st.info(f"📬 Alerts → `{owner_email}`")
        else:
            st.warning("⚠️ OWNER_EMAIL not set")

        st.divider()

        # ── 3. MANUAL EMAIL BUTTONS ───────────────────────────────────────────
        _ready = bool(api_key and owner_email)

        if st.button("🧪 Send Test Email", use_container_width=True, disabled=not _ready):
            with st.spinner("Sending test email…"):
                try:
                    from utils.notifications import send_pipeline_notification
                    ok = send_pipeline_notification(
                        status  = "TEST",
                        message = "Test alert from the Data Pipeline Dashboard.",
                        details = (
                            f"Dashboard is running correctly.\n"
                            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Owner: {owner_email}"
                        ),
                    )
                    st.success("✅ Test email sent!") if ok else st.error("❌ Failed.")
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

        st.markdown("**Send Pipeline Summary**")
        if st.button("📊 Email Current Summary", use_container_width=True, disabled=not _ready):
            with st.spinner("Building summary and sending…"):
                try:
                    import glob
                    from utils.notifications import send_pipeline_notification

                    script_dir   = Path(__file__).resolve().parent
                    input_path   = str(script_dir / "data" / "input")
                    csv_files    = glob.glob(os.path.join(input_path, "*.csv"))
                    raw_df       = _compute_gold_from_raw(input_path)
                    total_orders = int(raw_df["total_orders"].sum()) if not raw_df.empty else 0

                    bronze_rt, silver_rt, _ = _compute_medallion_realtime(input_path)
                    duplicates = max(0, len(bronze_rt) - len(silver_rt)) if not bronze_rt.empty else 0

                    details = (
                        f"Total CSV files in input  : {len(csv_files)}\n"
                        f"Total raw records          : {len(bronze_rt) if not bronze_rt.empty else 0}\n"
                        f"Unique (Silver) records    : {len(silver_rt) if not silver_rt.empty else 0}\n"
                        f"Duplicate records filtered : {duplicates}\n"
                        f"Total unique daily orders  : {total_orders}\n"
                        f"Report generated at        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    ok = send_pipeline_notification(
                        status  = "SUCCESS",
                        message = "Pipeline is healthy — see summary below.",
                        details = details,
                    )
                    st.success("✅ Summary email sent!") if ok else st.error("❌ Failed.")
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

        st.divider()
        st.caption("Emails auto-send from Airflow on SUCCESS / ERROR.")


def render_dashboard() -> None:
    inject_shared_styles()
    render_email_sidebar()
    current_view = render_top_nav()

    if current_view == "Dashboard":
        render_dashboard_home()
        # Diagnostics omitted per user request. Use render_diagnostics_workflow() if you want to restore it later.
    elif current_view == "Bronze -> Validation -> Silver":
        st.title("Medallion Flow and ACID")
        render_medallion_section()
    else:
        render_quality_report()


if __name__ == "__main__":
    render_dashboard()
