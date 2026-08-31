# import numpy as np
# import pandas as pd
# from create_engine import ENGINE
# from sqlalchemy import text

# file_list = [
#     r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\dataset_again\Tuesday-WorkingHours_reduced.csv',
#     r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\dataset_again\Final_again_Wednesday.csv',
#     r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\dataset_again\Final_again_Thursday.csv',
#     r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\dataset_again\Final_again_Friday.csv',
# ]

# with ENGINE.begin() as conn:
#     conn.execute(text("DROP TABLE IF EXISTS raw_sentiment_data"))

# print("✅ Old table dropped. Starting fresh...\n")

# # 🔥 STEP 2: LOAD DATA IN CHUNKS
# for file in file_list:
#     print(f"Processing: {file}")

#     reader = pd.read_csv(file, chunksize=50000, low_memory=False)

#     for chunk in reader:

#         # Clean data
#         chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
#         chunk.columns = [c.strip().lower() for c in chunk.columns]

#         try:
#             # Insert into MySQL
#             chunk.to_sql(
#                 name='IDS_DATA_2',
#                 con=ENGINE,
#                 if_exists='append',
#                 index=False
#             )

#         except Exception as e:
#             print(f"❌ Chunk failed: {e}")
#             continue

#     print(f"✅ Finished file: {file}")

# print("\n🎯 ALL DATA SUCCESSFULLY LOADED")
import numpy as np
import pandas as pd
from create_engine import ENGINE
from sqlalchemy import text

file_list = [
    r"D:\downloads\Tuesday-WorkingHours_reduced.csv",
    r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\optimizeddatasets\Final_again_We.csv',
    r"C:\Users\UMAR.TECH\Downloads\Th_merged_shuffled (1).csv",
    r'C:\Users\UMAR.TECH\Desktop\cyber\3rd milestone\optimizeddatasets\Final_again_Fr.csv'
]

TABLE_NAME = 'IDS_DATA_BEFORE_PREPROCESSING'

with ENGINE.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))

print(f"✅ Old table '{TABLE_NAME}' dropped. Starting fresh...\n")

# STEP 1: LOAD + CLEAN ALL FILES INTO ONE DATAFRAME
all_chunks = []

for file in file_list:
    print(f"Reading: {file}")

    for chunk in pd.read_csv(file, chunksize=50000, low_memory=False):
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        all_chunks.append(chunk)

    print(f"✅ Finished reading: {file}")

print("\n🔀 Combining and shuffling all datasets...")
full_df = pd.concat(all_chunks, ignore_index=True)
del all_chunks  # free memory

full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total rows after combining: {len(full_df)}")

# STEP 2: INSERT SHUFFLED DATA IN CHUNKS
chunk_size = 50000
total_chunks = (len(full_df) // chunk_size) + 1

for i in range(0, len(full_df), chunk_size):
    chunk = full_df.iloc[i:i + chunk_size]

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