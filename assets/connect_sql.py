
import numpy as np
import pandas as pd
from sqlalchemy import text

from create_engine import ENGINE

file_list = [
    r"D:\downloads\Tuesday-WorkingHours_reduced.csv",
    r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\optimizeddatasets\Final_again_We.csv',
    r"C:\Users\UMAR.TECH\Downloads\Th_merged_shuffled (1).csv",
    r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\optimizeddatasets\Final_again_Fr.csv'
]

TABLE_NAME = 'IDS_DATA_BEFORE_PREPROCESSING'
CHUNK_SIZE = 50000

with ENGINE.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))

print(f"✅ Old table '{TABLE_NAME}' dropped. Starting fresh...\n")

all_chunks = []

for file in file_list:
    print(f"Reading: {file}")

    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        all_chunks.append(chunk)

    print(f"✅ Finished reading: {file}")

print("\n🔀 Combining and shuffling all datasets...")
full_df = pd.concat(all_chunks, ignore_index=True)
del all_chunks 

full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total rows after combining: {len(full_df)}")

for i in range(0, len(full_df), CHUNK_SIZE):
    chunk = full_df.iloc[i:i + CHUNK_SIZE]

    try:
        chunk.to_sql(
            name=TABLE_NAME,
            con=ENGINE,
            if_exists='append',
            index=False
        )
        print(f"✅ Inserted rows {i} to {i + len(chunk)}")

    except Exception as e:
        print(f"❌ Chunk starting at row {i} failed: {e}")
        continue

print("\n🎯 ALL DATA SUCCESSFULLY LOADED (SHUFFLED)")
