import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
import json
import ast

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

# Reload fresh data
df = pd.read_csv('tokens_non_latin_words_split_2_Ambra.csv')

# Debugging the dataset
print("Dataset before predictions:")
print(df.head())

# Ensure tokens column is properly parsed as lists if needed
if 'tokens' in df.columns:
    df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    raise ValueError("Missing 'tokens' column in the dataset")

# Apply the heuristic model
word_label_map, df_with_predictions = heuristic_predict(df, token_column='tokens', label_column='nationality', min_freq=1000)

# Save results
df_with_predictions.to_csv('heuristic_predictions.csv', index=False)

# Evaluate
y_true = df_with_predictions['nationality']
y_pred = df_with_predictions['predicted_label']
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
print(f"F1 Score: {f1:.2f}")

# Save metrics
baseline_metrics = {'f1_score': f1}
with open('baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f)
