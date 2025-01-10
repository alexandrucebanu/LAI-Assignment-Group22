from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # You can try other backends like 'Qt5Agg'

# ======================================================================================================================
# INSTRUCTIONS
#
# change input csv in line 47
# change function parameters in line 50
# ======================================================================================================================

def plot_word_frequencies(data, text_column, top_n=20):
    """
    Plot the frequency of the top N words in a text dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        text_column (str): The column containing text data.
        top_n (int): The number of top words to display.
    """
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the text data
    word_matrix = vectorizer.fit_transform(data[text_column])

    # Sum up the occurrences of each word
    word_counts = word_matrix.sum(axis=0).A1  # Convert sparse matrix to array
    words = vectorizer.get_feature_names_out()

    # Create a DataFrame with word frequencies
    word_freq_df = pd.DataFrame({'word': words, 'count': word_counts})
    word_freq_df = word_freq_df.sort_values(by='count', ascending=False).head(top_n)

    # Plot the frequencies
    word_freq_df.plot(kind='bar', x='word', y='count', legend=False, figsize=(10, 5))
    plt.title(f'Top {top_n} Word Frequencies')
    plt.ylabel('Frequency')
    plt.xlabel('Word')
    plt.xticks(rotation=45)
    plt.show()


# Example DataFrame
df = pd.read_csv('token_nationality_subset.csv')

# Plot using the automated CountVectorizer method
plot_word_frequencies(df, text_column='post', top_n=5000)