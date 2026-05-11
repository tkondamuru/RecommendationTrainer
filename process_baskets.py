import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RAW_FILE = 'DataExtractor/sales_history_raw_sample.csv'
INTERCHANGE_FILE = 'DataExtractor/interchanges_sample.csv'

print(f"Loading data files...")
df = pd.read_csv(RAW_FILE, low_memory=False)
inter_df = pd.read_csv(INTERCHANGE_FILE)

# 1. Region Mapping and Normalization
df['REGION'] = df['LOC_NO'].apply(lambda x: 'CA' if x > 900 else 'US')

print("Normalizing parts...")
inter_map = {}
for _, row in inter_df.iterrows():
    p1, p2 = str(row['ITEM_UID_NO']), str(row['RELATED_ITEM_UID_NO'])
    master = min(p1, p2)
    inter_map[p1] = master
    inter_map[p2] = master

df['MASTER_ITEM_ID'] = df['ITEM_UID_NO'].astype(str).apply(lambda x: inter_map.get(x, x))
df['ORDER_KEY'] = df['LOC_NO'].astype(str) + "-" + df['SHIPPER_NO'].astype(str)
df['CATGRY_CD'] = pd.to_numeric(df['CATGRY_CD'], errors='coerce').fillna(0)
df['is_sundry'] = df['CATGRY_CD'] >= 60
df['is_glass'] = df['CATGRY_CD'] < 60

# 2. Regional Negative Sampling (Aggressive 10:1 Ratio)
for region in ['US', 'CA']:
    print(f"\n--- Generating High-Precision Training Data: {region} ---")
    reg_df = df[df['REGION'] == region].copy()
    if reg_df.empty: continue

    # Identify valid cross-sell orders
    order_types = reg_df.groupby('ORDER_KEY').agg({'is_glass': 'any', 'is_sundry': 'any'})
    valid_orders = order_types[order_types['is_glass'] & order_types['is_sundry']].index
    reg_gold = reg_df[reg_df['ORDER_KEY'].isin(valid_orders)].copy()

    # Get Positive Pairs
    glass_parts = reg_gold[reg_gold['is_glass']][['ORDER_KEY', 'MASTER_ITEM_ID']].rename(columns={'MASTER_ITEM_ID': 'glass_id'})
    sundry_parts = reg_gold[reg_gold['is_sundry']][['ORDER_KEY', 'MASTER_ITEM_ID']].rename(columns={'MASTER_ITEM_ID': 'sundry_id'})
    
    positives = pd.merge(glass_parts, sundry_parts, on='ORDER_KEY')[['glass_id', 'sundry_id']].drop_duplicates()
    positives['LABEL'] = 1.0

    # --- AGGRESSIVE NEGATIVE SAMPLING ---
    print(f"Creating 10x negative samples (Contrast Learning)...")
    all_glasses = positives['glass_id'].unique()
    sundry_popularity = sundry_parts['sundry_id'].value_counts()
    all_sundries = sundry_popularity.index.tolist()
    weights = sundry_popularity.values / sundry_popularity.sum()
    
    pos_set = set(zip(positives['glass_id'], positives['sundry_id']))
    
    neg_data = []
    # Use a faster sampling method for large volume
    for glass in all_glasses:
        # Generate 10 negatives per glass
        found = 0
        attempts = 0
        while found < 10 and attempts < 20:
            random_sundries = np.random.choice(all_sundries, size=10, p=weights)
            for s in random_sundries:
                if (glass, s) not in pos_set:
                    neg_data.append([glass, s, 0.0])
                    found += 1
                if found >= 10: break
            attempts += 1
                
    negatives = pd.DataFrame(neg_data, columns=['glass_id', 'sundry_id', 'LABEL'])
    
    # Combine and Shuffle
    final_training = pd.concat([positives, negatives]).sample(frac=1).reset_index(drop=True)
    
    ml_file = f'recommendation_training_{region}.csv'
    final_training.to_csv(ml_file, index=False)
    print(f"Saved {len(final_training):,} rows to {ml_file} ({len(positives)} Pos / {len(negatives)} Neg)")

print("\n--- High-Precision Processing Complete ---")
