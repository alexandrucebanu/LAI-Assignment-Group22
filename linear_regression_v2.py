import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import StandardScaler

# Download necessary NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the new cleaned training data and raw test/validation datasets
train_cleaned = pd.read_csv('Data/cleaned_train_split_2_Ambra_tokens.csv', encoding="latin1", engine="python",
                            on_bad_lines="skip")  # Path to your new cleaned training file
test_raw = pd.read_csv('Data/test_data.csv', encoding="latin1", engine="python", on_bad_lines="skip")  # Replace with your raw testing data file path
val_raw = pd.read_csv('Data/val_data.csv', encoding="latin1", engine="python", on_bad_lines="skip")  # Replace with your raw validation data file path


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
feature_columns = ['word_frequency', 'stopword_frequency', 'vocabulary_richness', 'punctuation_count',
                   'average_word_length']

X_train = train_cleaned[feature_columns]
string_labels_train = train_cleaned['nationality'].tolist()
label_mapping = {label: idx for idx, label in enumerate(sorted(set(string_labels_train)))}
y_train = np.array([label_mapping[label] for label in string_labels_train])

X_test = test_raw[feature_columns]
string_labels_test = test_raw['nationality'].tolist()
y_test = np.array([label_mapping[label] for label in string_labels_test])

X_val = val_raw[feature_columns]
string_labels_val = val_raw['nationality'].tolist()
y_val = np.array([label_mapping[label] for label in string_labels_val])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Step 5: Train logistic regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 6: Evaluate on test data
print("Test Dataset Evaluation:")
y_pred_test = lin_reg.predict(X_test_scaled)

# Since Linear Regression outputs continuous values, round predictions to nearest integer
y_pred_test_rounded = np.rint(y_pred_test).astype(int)

# Calculate precision, recall, and F1-score
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
    y_test, y_pred_test_rounded, average='weighted', zero_division=0
)
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1-Score: {f1_test}")

# Step 7: Evaluate on validation data
print("\nValidation Dataset Evaluation:")
y_pred_val = lin_reg.predict(X_val_scaled)

# Round predictions to nearest integer
y_pred_val_rounded = np.rint(y_pred_val).astype(int)

# Calculate precision, recall, and F1-score
precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(
    y_val, y_pred_val_rounded, average='weighted', zero_division=0
)
print(f"Precision: {precision_val}")
print(f"Recall: {recall_val}")
print(f"F1-Score: {f1_val}")

