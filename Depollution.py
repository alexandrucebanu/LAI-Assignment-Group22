from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

# Read data
df = pd.read_csv('train_split_2_Ambra.csv')


def create_tf_idf(data, text_column, max_features=10000, min_df=2, max_df=0.95):
    """
    Create a memory-efficient TF-IDF matrix for a given dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing text data.
        text_column (str): The name of the column containing text data.
        max_features (int, optional): The maximum number of features to include.
        min_df (int, optional): Minimum document frequency for terms.
        max_df (float, optional): Maximum document frequency for terms.

    Returns:
        scipy.sparse.csr_matrix: Sparse TF-IDF matrix.
        list: List of feature names.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    tfidf_matrix = vectorizer.fit_transform(data[text_column])
    return tfidf_matrix, vectorizer.get_feature_names_out()


tfidf_matrix, feature_names = create_tf_idf(df, 'post', max_features=10000)
print("Sparse TF-IDF Matrix:", tfidf_matrix)
print("Feature Names:", feature_names[:10])  # Show first 10 features

# Convert sparse matrix to dense format (for a small subset of rows)
dense_tfidf = tfidf_matrix[:10].toarray()  # Only first 10 rows for evaluation

# Extract top terms for each document
for doc_id, row in enumerate(dense_tfidf):
    top_indices = np.argsort(row)[::-1][:5]  # Top 5 terms
    top_terms = [(feature_names[i], row[i]) for i in top_indices]
    print(f"Document {doc_id + 1} top terms:", top_terms)


def remove_nationalities_cities_countries_and_urls(df_text, text_column, df_nationalities, nationality_column,
                                                   df_countries, country_column, df_city, city_column):
    """
    Removes all instances of nationalities, countries, cities, markdown, formatted URLs, and hyperlinks from a specified
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
    # Combine nationalities, countries, and cities into one list
    terms_to_remove = pd.concat([
        df_nationalities[nationality_column],
        df_countries[country_column],
        df_city[city_column]
    ]).unique()
    terms_pattern = r'\b(' + '|'.join(map(re.escape, terms_to_remove)) + r')\b'

    # Patterns to remove:
    # unwanted_patterns = r'(?:&[a-zA-Z]+(?:;[a-zA-Z]+;)?|\burl\b|[\^|>]|---|\s{2,})'
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

        # Non-ASCII characters
        r'\b\S*[^\x00-\x7F]\S*\b'  # Words with non-ASCII characters in them
    ]

    # Combine all patterns into a single regex
    unwanted_patterns = '|'.join(patterns)
    # Create a copy of the original dataframe to avoid modifying it
    df_result = df_text.copy()

    # Clean the text column
    df_result['cleaned_text'] = (
        df_result[text_column]
        .str.replace(unwanted_patterns, '', regex=True)  # Remove url and other artifacts, or text irregularities,
        # as well as formatted links
        .str.replace(terms_pattern, '', regex=True)  # Remove terms (nationalities, countries, cities)
        .str.strip()
    )

    # Calculate word counts before and after cleaning
    df_result['original_word_count'] = df_result[text_column].str.split().str.len()
    df_result['cleaned_word_count'] = df_result['cleaned_text'].str.split().str.len()

    # Calculate the difference in word counts
    df_result['word_count_difference'] = df_result['original_word_count'] - df_result['cleaned_word_count']

    return df_result


# Import data
df_nationalities = pd.read_csv('CH_Nationality_List_20171130_v1.csv')
df_countries = pd.read_csv('worldcities.csv')

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

cleaned_df.to_csv('cleaned_train_split_2_Ambra.csv')
