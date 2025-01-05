import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the training and test datasets
train_data = pd.read_csv('Data/tokenized/tokens_cleaned_train_split_2_Ambra_exp4.csv', encoding="latin1", engine="python",
                            on_bad_lines="skip")  # Training data
test_data = pd.read_csv('Data/tokenized/tokens_test_data.csv', encoding="latin1", engine="python", on_bad_lines="skip")  # Testing data


# Step 2: Feature engineering function for tokenized data
def extract_features_tokenized(df, token_column):
    df['word_frequency'] = df[token_column].apply(len)
    df['stopword_frequency'] = df[token_column].apply(
        lambda tokens: sum(1 for word in tokens if word.lower() in stop_words) / len(tokens) if len(tokens) > 0 else 0
    )
    df['vocabulary_richness'] = df[token_column].apply(
        lambda tokens: len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
    )
    df['punctuation_count'] = df[token_column].apply(
        lambda tokens: sum(1 for word in tokens if word in ['.', ',', '!', '?'])
    )
    df['average_word_length'] = df[token_column].apply(
        lambda tokens: np.mean([len(word) for word in tokens]) if len(tokens) > 0 else 0
    )
    return df


# Apply feature engineering to training and test data
train_data = extract_features_tokenized(train_data, 'tokens')
test_data = extract_features_tokenized(test_data, 'tokens')

# Step 3: Define features and target
feature_columns = ['word_frequency', 'stopword_frequency', 'vocabulary_richness', 'punctuation_count', 'average_word_length']

X_train = train_data[feature_columns]
string_labels_train = train_data['nationality'].tolist()

# Optimize label_mapping: Order by frequency
label_frequencies = pd.Series(string_labels_train).value_counts()
label_mapping = {label: idx for idx, label in enumerate(label_frequencies.index)}
print("Label Mapping:", label_mapping)  # Log the label mapping for debugging

# Map training labels to integers
y_train = np.array([label_mapping[label] for label in string_labels_train])

# Prepare test data
X_test = test_data[feature_columns]
string_labels_test = test_data['nationality'].tolist()

# Handle missing test labels gracefully
y_test = np.array([label_mapping.get(label, -1) for label in string_labels_test])  # Use -1 for unseen labels

# Check for missing labels in the test set
missing_labels = set(string_labels_test) - set(label_mapping.keys())
if missing_labels:
    print(f"Warning: The following labels in the test set are not in the training set: {missing_labels}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Remove classes with fewer than 2 samples
unique_classes, class_counts = np.unique(y_train, return_counts=True)
classes_to_keep = unique_classes[class_counts > 1]
X_train_filtered = X_train_scaled[np.isin(y_train, classes_to_keep)]
y_train_filtered = y_train[np.isin(y_train, classes_to_keep)]

# Train the logistic regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_filtered, y_train_filtered)

# Predict and evaluate on the test data
y_pred_test = lin_reg.predict(X_test_scaled)
y_pred_test_rounded = np.rint(y_pred_test).astype(int)

# Map predicted labels back to their string representation
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_nationalities = [inverse_label_mapping.get(pred, "Unknown") for pred in y_pred_test_rounded]

# Save predictions with original labels to CSV
output_df = test_data.copy()
output_df['predicted_nationality'] = predicted_nationalities
output_df[['tokens', 'nationality', 'predicted_nationality']].to_csv('Output/predictions_exp4.csv', index=False)

# Calculate precision, recall, and F1-score (excluding missing labels in y_test)
valid_indices = y_test != -1  # Exclude rows with -1 in y_test
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
    y_test[valid_indices], y_pred_test_rounded[valid_indices], average='weighted', zero_division=0
)

print("Test Dataset Evaluation:")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1-Score: {f1_test}")
