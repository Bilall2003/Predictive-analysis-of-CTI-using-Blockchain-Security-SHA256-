# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlalchemy import create_engine
from your_db_module import ENGINE

# Load data from database
query = "SELECT * FROM IDS_DATA_FINALIZED"
df = pd.read_sql(query, ENGINE)

# Load saved label encoder and decode label column
model_load = joblib.load('encoder_Saved.joblib')
print(dict(enumerate(model_load.classes_)))
df['label_name'] = model_load.inverse_transform(df['label'].astype(int))

# Bar plot: distribution of attack types
plt.figure(figsize=(12, 6))
label_counts = df['label_name'].value_counts()
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.xticks(rotation=45)
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.tight_layout()
plt.show()

# Pie chart: benign vs attack split
benign = (df['label'] == 0).sum()
attack = (df['label'] != 0).sum()
plt.pie([benign, attack], labels=['BENIGN', 'ATTACK'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
plt.title('BENIGN vs ATTACK')
plt.show()

# Detailed pie chart: all traffic classes with percentages and counts
counts = df['label_name'].value_counts()
percentages = df['label_name'].value_counts(normalize=True) * 100
plt.figure(figsize=(12, 6))
colors = sns.color_palette('tab20', len(counts))
explode = [0.05] * len(counts)
wedges, _ = plt.pie(counts, labels=None, startangle=140, colors=colors, explode=explode)
legend_labels = [
    f'{label} ({pct:.2f}%) [{count:,}]'
    for label, pct, count in zip(counts.index, percentages, counts.values)
]
plt.legend(wedges, legend_labels, title="Traffic Classes", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Bar chart: median flow duration per label
plt.figure(figsize=(12, 6))
df.groupby('label_name')['flow duration'].median().sort_values().plot(kind='barh')
plt.xlabel('Median Flow Duration')
plt.ylabel('Label')
plt.show()

# Bar chart: median down/up ratio per label
plt.figure(figsize=(12, 6))
order = df.groupby('label_name')['down/up ratio'].median().sort_values().index
sns.barplot(data=df, x='label_name', y='down/up ratio', order=order)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Heatmap: feature correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

# Pairplot: time-based features
time_features = ['flow duration', 'flow iat mean', 'fwd iat mean', 'bwd iat mean', 'label']
sns.pairplot(df.sample(1000, random_state=42)[time_features], hue='label')
plt.show()

# Pairplot: volume/header features
volume_features = ['total fwd packet', 'total bwd packets', 'fwd header length', 'bwd header length', 'label']
sns.pairplot(df.sample(1000, random_state=42)[volume_features], hue='label')
plt.show()

# Pairplot: packet length features
packet_cols = ['packet length min', 'packet length max', 'packet length mean', 'packet length std']
sns.pairplot(df.sample(1000, random_state=42)[packet_cols])
plt.show()
