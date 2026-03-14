import json
import random

def generate_validation_json(filename, total_rows=1000, unique_records=200):
    products = ["Cloud Service", "Quantum Core", "Smart Widget", "Bio-Sensor", "Neural Bridge"]
    
    # Generate unique base records
    base_records = []
    for i in range(unique_records):
        base_records.append({
            "order_id": f"J{1000 + i}",
            "order_date": "2024-03-14",
            "product": random.choice(products),
            "revenue": round(random.uniform(50.0, 500.0), 2)
        })
    
    # Fill up to 1000 total rows with duplicates
    data = []
    while len(data) < total_rows:
        data.append(random.choice(base_records))
        
    # Shuffle
    random.shuffle(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    import os
    target_path = r"c:\Users\nithi\OneDrive\Documents\Data-Engineering-Pipeline\data\input\validation_data_1k.json"
    generate_validation_json(target_path)
    print(f"Generated {target_path}")
