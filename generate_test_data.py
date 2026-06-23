import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Load real dataset
df_real = pd.read_csv('data/creditcard.csv')

# Separate frauds and legitimate
frauds = df_real[df_real['Class'] == 1]
legit = df_real[df_real['Class'] == 0]

# Sample 50 frauds and 950 legitimate transactions
sampled_frauds = frauds.sample(n=min(50, len(frauds)), replace=False)
sampled_legit = legit.sample(n=950, replace=False)

# Combine and shuffle
test_df = pd.concat([sampled_frauds, sampled_legit]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = 'test_dataset.csv'
test_df.to_csv(output_path, index=False)
print(f"Realistic test dataset with {len(sampled_frauds)} frauds generated and saved to {output_path}")
