import pandas as pd

# Read data
df = pd.read_csv('token_nationality.csv')

# Assuming 'df' is your original DataFrame
subset_df = df.sample(frac=0.10, random_state=42)  # Adjust 'frac' to 0.15 for 15%

# Save subset
subset_df.to_csv('token_nationality_subset.csv', index=False)

