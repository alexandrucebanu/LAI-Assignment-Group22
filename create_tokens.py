import pandas as pd
import re

# Read DataFrame that needs to be preprocessed
df = pd.read_csv('/Users/alexandrucebanu/Desktop/BACHELOR/YEAR 3/Q2/Language and AI/AAAA/LAI-Assignment-Group22/cleaned_train_split_2_Ambra.csv')

def preprocess(text):
    # Make all text lowercase
    text = text.lower()
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed Characters
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    # Remove list indices like (a), (b), etc.
    text = re.sub(r'\(\w\)', '', text)
    # Remove single letters except for "I", "A", "U" (and lowercase variants)
    text = re.sub(r'\b(?!(I|A|U|i|a|u)\b)\w\b', '', text)
    # Remove lone letters surrounded by punctuation, except i, o, u, a (case-insensitive)
    text = re.sub(r'\b(?!(i|o|u|a|I|O|U|A)\b)[a-zA-Z]\b(?=\W|\s|$)', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing to the 'cleaned_text' column
df['cleaned_text'] = df['cleaned_text'].apply(preprocess)  # Apply preprocess to 'cleaned_text' column

# Save the resulting DataFrame to a new CSV without tokens
df.to_csv('new_cleaned_for_lore.csv', index=False)

print("Processing complete. Cleaned text has been saved to 'new_cleaned_Ambra.csv'.")
