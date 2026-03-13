import pandas as pd
import glob
import os

input_path = r"g:\Data-Engineering-Pipeline\data\input"
csv_files = glob.glob(os.path.join(input_path, "*.csv"))

dfs = []
for f in csv_files:
    first_row = pd.read_csv(f, nrows=1)
    header_keywords = ["id", "order", "date", "revenue", "product", "amount"]
    has_header = any(any(key in str(col).lower() for key in header_keywords) for col in first_row.columns)
    
    # Simulate on_bad_lines='skip' because the dashboard doesn't have it and might fail
    try:
        temp_df = pd.read_csv(f) if has_header else pd.read_csv(f, header=None)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        try:
            temp_df = pd.read_csv(f, on_bad_lines='skip') if has_header else pd.read_csv(f, header=None, on_bad_lines='skip')
            print(f"Successfully read {f} with on_bad_lines='skip'")
        except Exception as e2:
            print(f"Failed again {f}: {e2}")
            continue

    if not has_header:
        canonical = ["order_id", "customer_id", "product", "unit_price", "quantity", "order_date"]
        temp_df.columns = [
            canonical[i] if i < len(canonical) else f"extra_col_{i - len(canonical) + 1}"
            for i in range(len(temp_df.columns))
        ]

    temp_df.columns = [str(c).strip().lower() for c in temp_df.columns]
    temp_df = temp_df.loc[:, ~temp_df.columns.duplicated(keep="first")]
    dfs.append(temp_df)
    print(f"File {f} columns: {list(temp_df.columns)}")

if dfs:
    bronze_rt = pd.concat(dfs, ignore_index=True, sort=False)
    print("Bronze RT columns:", list(bronze_rt.columns))
