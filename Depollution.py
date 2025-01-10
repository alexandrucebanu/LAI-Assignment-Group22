from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

# Read data
df = pd.read_csv('../raw_data/train_data.csv')


def remove_nationalities_cities_countries_and_urls(df_text, text_column, df_nationalities, nationality_column,
                                                   df_countries, country_column, df_city, city_column):
    """
    Removes all instances of nationalities, countries, cities, markdown, formatted URLs, emojis, and hyperlinks from a specified
    text column in a dataframe.

    Parameters:
    - df_text (pd.DataFrame): The dataframe containing the text to clean.
    - text_column (str): The name of the column in df_text containing the text to process.
    - df_nationalities (pd.DataFrame): The dataframe containing a list of nationalities.
    - nationality_column (str): The name of the column in df_nationalities with nationality names.
    - df_countries (pd.DataFrame): The dataframe containing a list of countries.
    - country_column (str): The name of the column in df_countries with country names.
    - df_city (pd.DataFrame): The dataframe containing a list of cities.
    - city_column (str): The name of the column in df_city with city names.

    Returns:
    - pd.DataFrame: A copy of the input dataframe with new columns:
        - `cleaned_text`: The processed text with terms and patterns removed.
        - `original_word_count`: Word count of the original text.
        - `cleaned_word_count`: Word count of the cleaned text.
        - `word_count_difference`: Difference in word counts.
    """
    import re
    import pandas as pd

    # Combine nationalities, countries, and cities into one list
    terms_to_remove = pd.concat([
        df_nationalities[nationality_column],
        df_countries[country_column],
        df_city[city_column]
    ]).unique()
    terms_pattern = r'\b(' + '|'.join(map(re.escape, terms_to_remove)) + r')\b'

    # Define regex patterns as individual components
    patterns = [
        r'&[a-zA-Z]+(?:;[a-zA-Z]+;)?',  # HTML entities like &nbsp; or &amp;nbsp;
        r'\burl\b',  # Standalone "url"
        r'[\^|>]',  # Symbols: ^, |, >
        r'---',  # Three hyphens ---
        r'\s{2,}',  # Two or more spaces

        # Remove only the markdown symbols
        r'\*\*\*',  # Bold + Italic symbols ***
        r'___',  # Bold + Italic symbols ___
        r'\*\*',  # Bold symbols **
        r'__',  # Bold symbols __
        r'\*',  # Italic symbols *
        r'_',  # Italic symbols _
        r'~~',  # Strikethrough symbols ~~

        # Spoiler and subreddit/user mentions
        r'>!|!<',  # Spoiler markers >! and !<
        r'r/([^ ]+)',  # Subreddit r/subreddit
        r'u/([^ ]+)',  # User mention u/username

        # List indicators
        r'^\s*[-*]\s+',  # Unordered list items: - or *
        r'\d+\.\s+',  # Ordered list items: 1. 2. 3.

        # Additional patterns for preprocessing
        r'\(\w\)',  # Remove list indices like (a), (b), etc.
        r'\b(?!(I|A|U|i|a|u)\b)\w\b',  # Remove single letters except for "I", "A", "U" (case-insensitive)
        r'\b(?!(i|o|u|a|I|O|U|A)\b)[a-zA-Z]\b(?=\W|\s|$)',  # Remove lone letters surrounded by punctuation
        r'\s+',  # Normalize whitespace
    ]

    # Emoji removal pattern
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

    # Combine all patterns into a single regex
    unwanted_patterns = '|'.join(patterns)
    df_result = df_text.copy()

    # Clean the text column
    df_result['cleaned_text'] = (
        df_result[text_column]
        .str.lower()  # Make all text lowercase
        .str.replace(emoji_pattern, '', regex=True)  # Remove emojis
        .str.replace(unwanted_patterns, '', regex=True)  # Remove unwanted patterns
        .str.replace(terms_pattern, '', regex=True)  # Remove terms (nationalities, countries, cities)
        .str.strip()  # Normalize whitespace
    )

    # Calculate word counts before and after cleaning
    df_result['original_word_count'] = df_result[text_column].str.split().str.len()
    df_result['cleaned_word_count'] = df_result['cleaned_text'].str.split().str.len()
    df_result['word_count_difference'] = df_result['original_word_count'] - df_result['cleaned_word_count']

    return df_result




# Import data
df_nationalities = pd.read_csv('../CH_Nationality_List_20171130_v1.csv')
df_countries = pd.read_csv('../worldcities.csv')

# Call the function
cleaned_df = remove_nationalities_cities_countries_and_urls(
    df_text=df,
    text_column='post',
    df_nationalities=df_nationalities,
    nationality_column='Nationality',
    df_countries=df_countries,
    country_column='country',
    df_city=df_countries,
    city_column='city'
)

# cleaned_df = preprocess(cleaned_df['cleaned_text'])




cleaned_df.to_csv('../clean_data/cleaned_train.csv')
