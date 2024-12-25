import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import nltk
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import StandardScaler


# Download necessary NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the new cleaned training data and raw test/validation datasets
train_cleaned = pd.read_csv('new_cleaned_for_lore.csv')  # Path to your new cleaned training file
test_raw = pd.read_csv('test_data.csv')  # Replace with your raw testing data file path
val_raw = pd.read_csv('val_data.csv')  # Replace with your raw validation data file path

# Step 2: Feature engineering function
def extract_features(df, text_column):
    df['word_frequency'] = df[text_column].apply(lambda text: len(text.split()))
    df['stopword_frequency'] = df[text_column].apply(
        lambda text: sum(1 for word in text.split() if word.lower() in stop_words) / len(text.split())
        if len(text.split()) > 0 else 0
    )
    df['vocabulary_richness'] = df[text_column].apply(
        lambda text: len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0
    )
    df['punctuation_count'] = df[text_column].apply(lambda text: len(re.findall(r'[.,!?]', text)))
    df['average_word_length'] = df[text_column].apply(
        lambda text: np.mean([len(word) for word in text.split()]) if len(text.split()) > 0 else 0
    )
    return df

# Step 3: Apply feature engineering
train_cleaned = extract_features(train_cleaned, 'cleaned_text')  # Features from cleaned training text
test_raw = extract_features(test_raw, 'post')  # Features from raw test text
val_raw = extract_features(val_raw, 'post')  # Features from raw validation text

# Step 4: Define features and target
feature_columns = ['word_frequency', 'stopword_frequency', 'vocabulary_richness', 'punctuation_count', 'average_word_length']

X_train = train_cleaned[feature_columns]
y_train = train_cleaned['nationality']

X_test = test_raw[feature_columns]
y_test = test_raw['nationality']

X_val = val_raw[feature_columns]
y_val = val_raw['nationality']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Step 5: Train logistic regression
log_reg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)

# Step 6: Evaluate on test data
print("Test Dataset Evaluation:")
y_pred_test = log_reg.predict(X_test)

# Calculate precision, recall, and F1-score
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
print(f"Precision: {precision_test:.2f}")
print(f"Recall: {recall_test:.2f}")
print(f"F1-Score: {f1_test:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Step 7: Evaluate on validation data
print("\nValidation Dataset Evaluation:")
y_pred_val = log_reg.predict(X_val)

# Calculate precision, recall, and F1-score
precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(y_val, y_pred_val, average='weighted')
print(f"Precision: {precision_val:.2f}")
print(f"Recall: {recall_val:.2f}")
print(f"F1-Score: {f1_val:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))
