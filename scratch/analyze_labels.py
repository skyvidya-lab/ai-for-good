import pandas as pd
import numpy as np

# Load the data
file_path = r'C:\Users\eluzq\workspace\ai-for-good\data\points_train_label.csv'
df = pd.read_csv(file_path)

# Convert phenophase_date to datetime
df['phenophase_date'] = pd.to_datetime(df['phenophase_date'], format='mixed')

# (1) Check if the time series is ordered by point_id
# Let's check point_id 1 specifically
p1 = df[df['point_id'] == 1].copy()
print("Point ID 1 (Original Order):")
print(p1[['point_id', 'phenophase_date', 'phenophase_name']])

# Check if it is sorted
is_sorted = p1['phenophase_date'].is_monotonic_increasing
print(f"\nIs Point 1 sorted? {is_sorted}")

# (2) Calculate phenophase_interval
# Sort by point_id and phenophase_date
df_sorted = df.sort_values(['point_id', 'phenophase_date'])

# Calculate interval in days
df_sorted['phenophase_interval'] = df_sorted.groupby('point_id')['phenophase_date'].diff().dt.days.fillna(0)

print("\nPoint ID 1 (Sorted with Interval):")
print(df_sorted[df_sorted['point_id'] == 1][['point_id', 'phenophase_date', 'phenophase_name', 'phenophase_interval']])

# (3) Exploratory Analysis Summary
print("\nPhenophase Interval Statistics:")
print(df_sorted['phenophase_interval'].describe())

# Average interval per transition
# To do this, we need to know the 'from' and 'to' phenophases
df_sorted['prev_phenophase'] = df_sorted.groupby('point_id')['phenophase_name'].shift(1)
df_sorted['transition'] = df_sorted['prev_phenophase'] + " -> " + df_sorted['phenophase_name']

transition_stats = df_sorted[df_sorted['phenophase_interval'] > 0].groupby('transition')['phenophase_interval'].agg(['mean', 'std', 'count']).sort_values('mean')
print("\nTransition Statistics (Interval in Days):")
print(transition_stats)

# Check for duplicates or multiple readings for the same phenophase per point
duplicates = df.groupby(['point_id', 'phenophase_name']).size().reset_index(name='counts')
print(f"\nAny point with more than one reading for the same phenophase? {any(duplicates['counts'] > 1)}")
if any(duplicates['counts'] > 1):
    print(duplicates[duplicates['counts'] > 1].head())
