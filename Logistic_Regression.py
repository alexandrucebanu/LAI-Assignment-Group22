import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import nltk

# ======================================================================================================================
# INSTRUCTIONS
#
# change to csv that contains your train data in line 25
# change to csv that contains your test data in line 26
# change column name to column that contains your tokens in line 46 and 47
# change chosen features in line 50
# change model parameters in line 70
# ======================================================================================================================


# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load the training and test datasets
train_data = pd.read_csv('tokenized_cleaned_train_split_2_Ambra_exp4.csv')  # Training data
test_data = pd.read_csv('tokenized_test_data.csv')  # Testing data

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
y_train = train_data['nationality']

X_test = test_data[feature_columns]
y_test = test_data['nationality']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Remove classes with fewer than 2 samples
class_counts = y_train.value_counts()
classes_to_keep = class_counts[class_counts > 1].index
X_train_filtered = X_train_scaled[np.isin(y_train, classes_to_keep)]
y_train_filtered = y_train[np.isin(y_train, classes_to_keep)]

# Step 5: Train the logistic regression model on the filtered dataset
log_reg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_filtered, y_train_filtered)

# Step 6: Predict and evaluate on the test data
y_pred_test = log_reg.predict(X_test_scaled)

# Calculate evaluation metrics
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

# Display evaluation results
print("Test Dataset Evaluation:")
print(f"Precision: {precision_test:.8f}")
print(f"Recall: {recall_test:.8f}")
print(f"F1-Score: {f1_test:.8f}")

# Detailed classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_test))