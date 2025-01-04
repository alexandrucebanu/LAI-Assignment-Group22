import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
import json
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler


# Preprocess the tokens column into a numerical format
def vectorize_tokens(df, token_column):
    df[token_column] = df[token_column].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), min_df=1)
    X = vectorizer.fit_transform(df[token_column])
    return X, vectorizer

# SMOTE Integration
def apply_smote(X, y, random_state=42, k_neighbors=1):
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_smote_or_oversampler(X, y, random_state=42, k_neighbors=1):
    """
    Apply SMOTE or leave classes as-is if there aren't enough neighbors.

    Args:
        X (sparse matrix): Feature matrix.
        y (array-like): Target labels.
        random_state (int): Random state for reproducibility.
        k_neighbors (int): Number of neighbors for SMOTE.

    Returns:
        X_resampled (sparse matrix): Resampled feature matrix.
        y_resampled (array-like): Resampled target labels.
    """
    # Get class distribution
    class_counts = pd.Series(y).value_counts()
    min_class_count = class_counts.min()

    # Check if SMOTE can be applied
    if min_class_count <= k_neighbors:
        print(f"Warning: Not enough samples for SMOTE. Skipping oversampling and keeping original data.")
        # Return the original data unchanged
        return X, y
    else:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled



def heuristic_predict(df, token_column, label_column, min_freq=5):
    # Debug: Ensure tokens column is properly formatted
    for idx, tokens in enumerate(df[token_column]):
        if not isinstance(tokens, list):
            try:
                df.at[idx, token_column] = ast.literal_eval(tokens)
            except Exception as e:
                print(f"Error converting tokens at index {idx}: {tokens}, Error: {e}")

    print("Tokens after format verification:")
    print(df[token_column].head())

    # Reset counters and mappings
    label_word_counts = defaultdict(Counter)
    word_label_map = {}

    # Build word-label map
    for _, row in df.iterrows():
        label = row[label_column]
        tokens = row[token_column]
        if not isinstance(tokens, list):
            print(f"Skipping invalid tokens: {tokens}")
            continue
        label_word_counts[label].update(tokens)
        for token in tokens:
            if len(token) == 1 and token not in {'i', 'o', 'u', 'a'}:
                print(f"Unwanted token: {token}")

    # Filter low-frequency words
    for label, counter in label_word_counts.items():
        for word, count in counter.items():
            if count >= min_freq:
                if word not in word_label_map or count > label_word_counts[word_label_map[word]][word]:
                    word_label_map[word] = label

    print("Word-Label Map:")
    print(word_label_map)

    # Predict function
    most_frequent_label = df[label_column].value_counts().idxmax()

    def predict(tokens):
        if not isinstance(tokens, list):
            print(f"Invalid tokens during prediction: {tokens}")
            return most_frequent_label
        scores = Counter()
        for word in tokens:
            if word in word_label_map:
                scores[word_label_map[word]] += 1
        return scores.most_common(1)[0][0] if scores else most_frequent_label

    # Apply prediction
    df['predicted_label'] = df[token_column].apply(predict)
    return word_label_map, df


def heuristic_predict_single(tokens, word_label_map):
    """
    Predict the label for a single instance based on a word-label map.

    Args:
        tokens (list of str): List of tokens for the instance.
        word_label_map (dict): A dictionary mapping tokens to labels.

    Returns:
        str: The predicted label for the instance.
    """
    label_counts = {}

    for token in tokens:
        # Check if token exists in the map
        if token in word_label_map:
            label = word_label_map[token]
            # Count occurrences of each label
            label_counts[label] = label_counts.get(label, 0) + 1

    # If no tokens match, return a default label or 'unknown'
    if not label_counts:
        return 'unknown'

    # Return the label with the highest count
    return max(label_counts, key=label_counts.get)


# Reload fresh data
df = pd.read_csv('../tokenised_data/tokens_split_2_Ambra_exp3_2_combined.csv')
df_test = pd.read_csv('../tokenised_data/tokens_test.csv')

# Debugging the dataset
print("Dataset before predictions:")
print(df.head())

# Parse tokens column as lists
df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert tokens to numerical features
X, vectorizer = vectorize_tokens(df, token_column='tokens')
y = df['nationality']

# Apply SMOTE
X_resampled, y_resampled = apply_smote_or_oversampler(X, y)

# Transform resampled X back to tokens (optional, for heuristic model compatibility)

resampled_tokens = []
for i in range(X_resampled.shape[0]):
    row_indices = X_resampled[i].indices  # Get non-zero indices in the sparse row
    tokens = [word for word, idx in vectorizer.vocabulary_.items() if idx in row_indices]
    resampled_tokens.append(tokens)

# Create a resampled DataFrame
df_resampled = pd.DataFrame({'tokens': resampled_tokens, 'nationality': y_resampled})


# Ensure tokens column is properly parsed as lists if needed
if 'tokens' in df.columns:
    df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    raise ValueError("Missing 'tokens' column in the dataset")

# Apply the heuristic model on the training data
word_label_map, df_with_predictions = heuristic_predict(df_resampled, token_column='tokens', label_column='nationality',
                                                        min_freq=10)

# Save training results
df_with_predictions.to_csv('../models/predictions/heuristic_predictions_exp3_2_combined.csv', index=False)

# Evaluate on training data
y_true_train = df_with_predictions['nationality']
y_pred_train = df_with_predictions['predicted_label']
f1_train = f1_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
print(f"F1 Score (Training Data): {f1_train:.2f}")

# Apply the model to the test data
df_test['predicted_label'] = df_test['tokens'].apply(
    lambda tokens: heuristic_predict_single(tokens, word_label_map)
)

# Evaluate on the external test data
y_true_test = df_test['nationality']
y_pred_test = df_test['predicted_label']
f1_test = f1_score(y_true_test, y_pred_test, average='weighted', zero_division=0)
print(f"F1 Score (Test Data): {f1_test:.2f}")

# Save test predictions
df_test.to_csv('../models/predictions/heuristic_predictions_test_exp3_2_combined.csv', index=False)

# Save metrics
metrics = {
    'f1_score_train': f1_train,
    'f1_score_test': f1_test
}
with open('../models/eval_metrics/baseline_metrics_exp3_2_combined.json', 'w') as f:
    json.dump(metrics, f)
