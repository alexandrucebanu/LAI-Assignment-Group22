import pandas as pd
import re

# Read DataFrame that needs to be tokenized
df = pd.read_csv('cleaned_no_non_latin_words_split_2_Ambra.csv')

# Function to preprocess text
def preprocess(text):
    # Make all text lowercase
    text = text.lower()
    # Remove list indices like (a), (b), etc.
    text = re.sub(r'\(\w\)', '', text)
    # Remove single letters except for "I", "A", "U" (and lowercase variants)
    text = re.sub(r'\b(?!(I|A|U|i|a|u)\b)\w\b', '', text)
    # Remove lone letters surrounded by punctuation, except i, o, u, a (case-insensitive)
    text = re.sub(r'\b(?!(i|o|u|a|I|O|U|A)\b)[a-zA-Z]\b(?=\W|\s|$)', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to tokenize text
def tokenize(text):
    # Match acronyms, contractions, words, and punctuation as separate tokens
    return re.findall(r'\b(?:[A-Za-z]\.)+[A-Za-z]\b|(?:\w+\'\w+)|\w+|[^\w\s]', text)

# Apply preprocessing and tokenization
df['cleaned_text'] = df['cleaned_text'].apply(preprocess)  # Apply preprocess to 'text' column
df['tokens'] = df['cleaned_text'].apply(tokenize)  # Apply tokenize to 'cleaned_text' column
df.drop(columns=['post'], inplace=True)

# Save the resulting DataFrame to a new CSV
df.to_csv('tokens_non_latin_words_split_2_Ambra.csv', index=False)

print("Processing complete. Cleaned text and tokens have been saved.")
