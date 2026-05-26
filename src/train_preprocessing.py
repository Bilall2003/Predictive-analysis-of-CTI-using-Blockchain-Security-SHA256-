
# PREPROCESSING

import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
from config.create_engine import ENGINE

query = """
SELECT *
FROM IDS_DATA
"""

df = pd.read_sql(query, ENGINE)

print(df.tail())

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df.isnull().sum().sum()

# %%
df.duplicated().sum()

# %%
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%
df.dropna(inplace=True)

# %%
df.drop_duplicates(keep='first', inplace=True)

# %%


# %%
df.info()


# %%
df=df.reset_index(drop=True)

# %%
drop_cols=["flow id", "src ip","dst ip", "src port", "dst port", "timestamp"]

# %%
df.columns

# %%
# drop irrelevant cols
df_new = df.drop(drop_cols, axis=1)

# remove benign duplicates only
feature_cols = [c for c in df_new.columns if c != 'label']

benign = (
    df_new[df_new['label'] == 'BENIGN']
    .drop_duplicates(subset=feature_cols)
)

attack = df_new[df_new['label'] != 'BENIGN']

# combine
df_new = pd.concat([benign, attack], axis=0)

# reset index
df_new.reset_index(drop=True, inplace=True)

# %%
df_new.columns

# %%
df_new.duplicated().sum()

# %%
# df_new.select_dtypes(include="object")
df_new["label"] = df_new["label"].str.replace('- Attempted', '', regex=False).str.strip()

# %%
df_new["label"].unique()

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()
df_new['label'] = le.fit_transform(df_new['label'])

# %%
df_new["label"].value_counts()

# %%
df_new.isnull().sum().sum()

# %%
df_new['label'].tail()

# %%
df_new.describe()

# %%
iat_columns = [
        'flow iat min',
        'flow iat max', 
        'flow iat mean',
        'fwd iat min',
        'bwd iat min',
        'flow duration',
        'flow packets/s' 
    ]
    
for col in iat_columns:
    if col in df_new.columns:
            # Count negatives
        neg_count = (df_new[col] < 0).sum()
            
        if neg_count > 0:
                print(f"⚠️ Found {neg_count} negative values in {col}")
                
                # Check severity
                severe = (df_new[col] < -1).sum()
                
                if severe > 0:
                    print(f"  🔴 {severe} values < -1 (serious corruption)")
                
                # Fix: Replace negative with 0
                    df_new.loc[df_new[col] < 0, col] = 0
                    print(f"  ✅ Replaced with 0")

# %%
num_cols = df_new.select_dtypes(include=[np.number]).columns
neg_cols = (df_new[num_cols] > 0).sum()

# %%
neg_cols

# %%
df_new.columns

# %%
print((df_new == -np.inf).sum().sum())

# %%
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

df_new.to_sql(
    name="IDS_DATA_FINALIZED",
    con=ENGINE,
    if_exists="fail",
    index=False,
    chunksize=10000
)
logging.info(f"Data stored in mysql successfully............")

# %%
df = pd.read_sql("SELECT * FROM IDS_DATA_FINALIZED", ENGINE)
print(df.head())


