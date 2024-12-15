import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def heuristic_predict(df, text_column, label_column, min_freq=2):
    """
    Predict labels based on word frequencies per label using a heuristic approach.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text and labels.
    - text_column (str): Column in df containing the text data.
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
        words = row[text_column].split()
        label_word_counts[label].update(words)

    # Filter out low-frequency words and associate words with labels
    word_label_map = {}
    for label, counter in label_word_counts.items():
        for word, count in counter.items():
            if count >= min_freq:
                if word not in word_label_map or counter[word] > label_word_counts[word_label_map[word]][word]:
                    word_label_map[word] = label

    # Predict labels for new texts
    def predict(text):
        scores = Counter()
        for word in text.split():
            if word in word_label_map:
                scores[word_label_map[word]] += 1
        return scores.most_common(1)[0][0] if scores else None

    # Apply the prediction function to the dataset
    df['predicted_label'] = df[text_column].apply(predict)
    return word_label_map, df

# Example dataset
df = pd.read_csv('token_nationality_subset.csv')

# Predict labels using the heuristic model
word_label_map, df_with_predictions = heuristic_predict(df, text_column='post', label_column='nationality', min_freq=2)

# View results
print("Word-Label Map:")
print(word_label_map)
print("\nPredictions:")
print(df_with_predictions[['post', 'nationality', 'predicted_label']])




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

