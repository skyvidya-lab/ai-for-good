import pandas as pd

# Load the data
file_path = r'C:\Users\eluzq\workspace\ai-for-good\data\points_train_label.csv'
df = pd.read_csv(file_path)

# Convert phenophase_date to datetime
df['phenophase_date'] = pd.to_datetime(df['phenophase_date'], format='mixed')

# Sort by point_id and date
df_sorted = df.sort_values(['point_id', 'phenophase_date'])

# Get the sequence of phenophases for each point
point_sequences = df_sorted.groupby('point_id')['phenophase_name'].apply(list).reset_index()

# Convert list to tuple for counting
point_sequences['sequence_tuple'] = point_sequences['phenophase_name'].apply(tuple)

# Count unique sequences
sequence_counts = point_sequences['sequence_tuple'].value_counts()

print("Common Phenophase Sequences:")
for seq, count in sequence_counts.items():
    print(f"Count: {count} | Sequence: {seq}")

# Specifically check Maturity vs Peak position
print("\nInvestigating Maturity vs Peak position:")
def check_maturity_peak(seq):
    if 'Maturity' in seq and 'Peak' in seq:
        m_idx = seq.index('Maturity')
        p_idx = seq.index('Peak')
        if m_idx < p_idx:
            return "Maturity BEFORE Peak"
        else:
            return "Peak BEFORE Maturity"
    return "Missing one or both"

point_sequences['rel_pos'] = point_sequences['sequence_tuple'].apply(check_maturity_peak)
rel_pos_counts = point_sequences['rel_pos'].value_counts()
print(rel_pos_counts)

# Check if it depends on crop_type
point_crops = df[['point_id', 'crop_type']].drop_duplicates()
point_sequences = point_sequences.merge(point_crops, on='point_id')

print("\nMaturity vs Peak position by Crop Type:")
pivot = point_sequences.pivot_table(index='crop_type', columns='rel_pos', aggfunc='size', fill_value=0)
print(pivot)
