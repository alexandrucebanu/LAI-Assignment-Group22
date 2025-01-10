import pandas as pd
import re

# ======================================================================================================================
# INSTRUCTIONS
#
# change input csv in line 13
# change column name to apply tokenization to in line 26 and 27
# change name output csv in line 30
# ======================================================================================================================

# Read DataFrame that needs to be tokenized
df = pd.read_csv('cleaned_train_split_2_Ambra_exp4.csv')

def remove_non_latin_words(text):
    text = re.sub(r'\b\S*[^\x00-\x7F]\S*\b', '', text)

    return text

# Function to tokenize text
def tokenize(text):
    # Match acronyms, contractions, words, and punctuation as separate tokens
    return re.findall(r'\b(?:[A-Za-z]\.)+[A-Za-z]\b|(?:\w+\'\w+)|\w+|[^\w\s]', text)

# Apply preprocessing and tokenization
df['tokens'] = df['post'].apply(tokenize)  # Apply tokenize to 'cleaned_text' column
df.drop(columns=['post'], inplace=True)

# Save the resulting DataFrame to a new CSV
df.to_csv('tokenized_cleaned_train_split_2_Ambra_exp4.csv', index=False)

print("Processing complete. Cleaned text and tokens have been saved.")