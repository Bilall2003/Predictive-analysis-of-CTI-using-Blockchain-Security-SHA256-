import numpy as np
import pandas as pd
from create_engine import ENGINE
from sqlalchemy import text

file_list = [
    r"C:\Users\UMAR.TECH\Desktop\cyber dataset\optimizeddatasets\Final_Tuesday.csv",
    r"C:\Users\UMAR.TECH\Desktop\cyber dataset\optimizeddatasets\Final_Wednesday.csv",
    r"C:\Users\UMAR.TECH\Desktop\cyber dataset\optimizeddatasets\Final_Friday.csv"
]

with ENGINE.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS raw_sentiment_data"))

print("✅ Old table dropped. Starting fresh...\n")

#  STEP 2: LOAD DATA IN CHUNKS
for file in file_list:
    print(f"Processing: {file}")

    reader = pd.read_csv(file, chunksize=50000, low_memory=False)

    for chunk in reader:

        # Clean data
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        try:
            # Insert into MySQL
            chunk.to_sql(
                name='IDS_DATA',
                con=ENGINE,
                if_exists='append',
                index=False
            )

        except Exception as e:
            print(f"❌ Chunk failed: {e}")
            continue

    print(f"✅ Finished file: {file}")

print("\n🎯 ALL DATA SUCCESSFULLY LOADED")