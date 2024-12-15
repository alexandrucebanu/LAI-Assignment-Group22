import pandas as pd
import re

# Read DataFrame that needs to be tokenized
df = pd.read_csv('nationality.csv')

# Function to tokenize text
def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text)

# Apply the function to the 'text' column and create a new column 'tokens'
df['tokens'] = df['post'].apply(tokenize)

df.to_csv('tokens.csv')
