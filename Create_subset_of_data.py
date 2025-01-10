import pandas as pd

# ======================================================================================================================
# INSTRUCTIONS
#
# change input csv in line 12
# change fraction of the original data you want to get as subset in line 16
# change name of output csv in line 18
# ======================================================================================================================

# Read data
df = pd.read_csv('token_nationality.csv')

# Assuming 'df' is your original DataFrame
subset_df = df.sample(frac=0.10, random_state=42)  # Adjust 'frac' to 0.15 for 15%

# Save subset
subset_df.to_csv('token_nationality_subset.csv', index=False)

