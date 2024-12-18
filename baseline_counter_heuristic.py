import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def heuristic_predict(df, token_column, label_column, min_freq=1):
    """
    Predict labels based on word frequencies per label using a heuristic approach.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text and labels.
    - token_column (str): Column in df containing the token data.
    - label_column (str): Column in df containing the labels.
    - min_freq (int): Minimum frequency for a word to be considered relevant.

    Returns:
    - dict: A dictionary mapping each word to its most associated label.
    - pd.DataFrame: Original DataFrame with an additional column `predicted_label`.
    """
    # Separate texts by label
    label_word_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        label = row[label_column]
        tokens = row[token_column]  # Directly use the token column
        label_word_counts[label].update(tokens)

    # Filter out low-frequency words and associate words with labels
    word_label_map = {}
    for label, counter in label_word_counts.items():
        for word, count in counter.items():
            if count >= min_freq:
                if word not in word_label_map or counter[word] > label_word_counts[word_label_map[word]][word]:
                    word_label_map[word] = label

    # Predict labels for new texts
    # Find the most frequent label
    most_frequent_label = df[label_column].value_counts().idxmax()

    def predict(tokens):
        scores = Counter()
        for word in tokens:
            if word in word_label_map:
                scores[word_label_map[word]] += 1
        if scores:
            return scores.most_common(1)[0][0]
        else:
            # Use most frequent label as a fallback
            return most_frequent_label

    # Apply the prediction function to the dataset
    df['predicted_label'] = df[token_column].apply(predict)
    return word_label_map, df

# Example dataset
df = pd.read_csv('tokens_non_latin_words_split_2_Ambra.csv')

# Predict labels using the heuristic model
word_label_map, df_with_predictions = heuristic_predict(df, token_column='tokens', label_column='nationality', min_freq=1000)

# View results
print("Word-Label Map:")
print(word_label_map)
print("\nPredictions:")
print(df_with_predictions[['cleaned_text', 'nationality', 'predicted_label']])




# Evaluate the baseline model
y_true = df_with_predictions['nationality']
y_pred = df_with_predictions['predicted_label']

#precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\nBaseline Performance Metrics:")
#print(f"Precision: {precision:.2f}")
#print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save predictions and metrics to a file
df_with_predictions.to_csv('heuristic_predictions.csv', index=False)

# Save metrics to a dictionary or JSON file
baseline_metrics = {
    'f1_score': f1
}

import json
with open('baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f)

