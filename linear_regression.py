import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk import pos_tag
import string
import ast

# uncomment this to download the necessary nltk resources if you don't have them yet
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng')

def extract_features(tokens_list, selected_features):
    """
    Extracts linguistic features from a list of tokenized texts.

    Parameters:
        tokens_list (list of list of str): List of tokenized texts (each text is a list of tokens).
        selected_features (list of str): List of features to extract (e.g., "sentence_length", "vocabulary_richness").

    Returns:
        np.ndarray: Array of extracted features where each row corresponds to a tokenized text.
    """
    features = []

    for tokens in tokens_list:
        token_list = ast.literal_eval(tokens)
        pos_tags = pos_tag(token_list)

        # Initialize feature values
        avg_sentence_length = 0
        vocab_richness = 0
        num_nouns = 0
        num_adjectives = 0
        num_punctuation = 0

        # Compute features based on selection
        if "sentence_length" in selected_features:
            # Split tokens into sentences based on punctuation (e.g., ".", "!", "?")
            sentence_delimiters = {".", "!", "?"}
            sentence_count = sum(1 for token in tokens if token in sentence_delimiters)
            sentence_count -= sum(1 for i in range(len(tokens) - 1) if tokens[i] in sentence_delimiters and tokens[i + 1] in sentence_delimiters)
            sentence_count = max(1, sentence_count)  # Avoid division by zero
            avg_sentence_length = len(tokens) / sentence_count


        if "vocabulary_richness" in selected_features:
            unique_words = set([word.lower() for word in tokens if word.isalpha()])
            vocab_richness = len(unique_words) / max(1, len(tokens))

        if "num_nouns" in selected_features:
            num_nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))

        if "num_adjectives" in selected_features:
            num_adjectives = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))

        if "num_punctuation" in selected_features:
            num_punctuation = sum(1 for token in tokens if token in string.punctuation)

        features.append([
            avg_sentence_length,
            vocab_richness,
            num_nouns,
            num_adjectives,
            num_punctuation
        ])

    return np.array(features)


def train_linear_regression(df, text_column, label_column, selected_features):
    """
    Trains a linear regression model on the given dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.
        text_column (str): Name of the column containing tokenized texts.
        label_column (str): Name of the column containing string labels.
        selected_features (list of str): List of features to use for training

    Returns:
        model (LinearRegression): Trained linear regression model.
        label_mapping (dict): Mapping from string labels to integers.
    """
    # Extract texts and labels
    tokenized_texts = df[text_column].tolist()
    string_labels = df[label_column].tolist()

    # Encode string labels as integers
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(string_labels)))}
    # different label mappings could influence the result
    labels = np.array([label_mapping[label] for label in string_labels])

    # Other features: sentence length, vocabulary richness, etc.
    all_features = extract_features(tokenized_texts, selected_features)

    # Train linear regression
    model = LinearRegression()
    model.fit(all_features, labels)

    return model, label_mapping


def evaluate_and_save_predictions(df, model, text_column, label_column, selected_features, label_mapping,
                                  output_csv='predictions.csv'):
    """
    Evaluates a trained model using F1 score, precision, and recall, and saves the predictions to a new column in the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset to evaluate.
        model (LinearRegression): Trained linear regression model.
        text_column (str): Name of the column containing tokenized texts.
        label_column (str): Name of the column containing string labels.
        selected_features (list of str): List of features used for training.
        label_mapping (dict): Mapping from string labels to integers.
        output_csv (str): Output file path to save the dataframe with predictions.

    Returns:
        tuple: F1 score, precision, and recall.
    """
    # Extract texts and labels
    tokenized_texts = df[text_column].tolist()
    string_labels = df[label_column].tolist()

    # Encode string labels as integers
    labels = np.array([label_mapping[label] for label in string_labels])

    # Other features: sentence length, vocabulary richness, etc.
    all_features = extract_features(tokenized_texts, selected_features)

    # Predict the labels using the trained model
    predictions = model.predict(all_features)
    predictions = np.round(predictions).astype(int)

    # Clamp predictions to the valid range
    min_label = min(label_mapping.values())
    max_label = max(label_mapping.values())
    predictions = np.clip(predictions, min_label, max_label)

    # Inverse label mapping: map predicted integers back to string labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_labels = []
    for pred in predictions:
        if pred in reverse_label_mapping:
            predicted_labels.append(reverse_label_mapping[pred])
        else:
            print(f"Warning: Prediction {pred} is out of range and has been ignored.")
            predicted_labels.append(None)

    # Calculate metrics
    valid_indices = [i for i, pred in enumerate(predictions) if pred in reverse_label_mapping]
    valid_labels = labels[valid_indices]
    valid_predictions = predictions[valid_indices]

    f1 = f1_score(valid_labels, valid_predictions, average='weighted')
    precision = precision_score(valid_labels, valid_predictions, average='weighted')
    recall = recall_score(valid_labels, valid_predictions, average='weighted')

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Add predicted labels to the dataframe
    df['predictions'] = predicted_labels

    # Save the dataframe with predictions to CSV
    df.to_csv(output_csv, index=False)

    return f1, precision, recall


# change parameters here
if __name__ == "__main__":
    df_train = pd.read_csv('Data/cleaned_train_split_2_Ambra_tokens.csv', encoding="latin1", engine="python", on_bad_lines="skip")

    # below all the possible features I added
    # ["avg_sentence_length", "vocabulary_richness", "num_nouns", "num_adjectives", "num_punctuation"]
    selected_features = ["avg_sentence_length", "vocabulary_richness", "num_nouns", "num_adjectives", "num_punctuation"]
    model, label_mapping = train_linear_regression(df_train, "tokens", "nationality", selected_features)

    df_val = pd.read_csv("Data/val_data_tokens.csv", encoding="latin1", engine="python", on_bad_lines="skip")

    # Evaluate the model
    evaluate_and_save_predictions(df_val, model, "tokens", "nationality", selected_features, label_mapping,
                                  output_csv="Output/predictions_val.csv")
    evaluate_and_save_predictions(df_train, model, "tokens", "nationality", selected_features, label_mapping,
                                  output_csv="Output/predictions.csv")

import pandas as pd
df = pd.read_csv("Output/predictions.csv")
print(df["predictions"].value_counts())
# results with sorted label mapping
# F1 Score: 0.0021906605298665474
# Precision: 0.004086407756453372
# Recall: 0.003170409511228534
# predictions
# Moldova           995
# Montenegro        711
# Mexico            699
# Malta             346
# Luxembourg        189
# Norway            189
# Lithuania         143
# Kosovo            133
# Italy             111
# Israel             69
# Ireland            65
# Iran               55
# Philippines        19
# Iceland            18
# Hungary             6
# Greece              5
# Georgia             5
# Brazil              3
# Czech Republic      3
# Germany             3
# Greenland           3
# Cyprus              2
# Croatia             2
# Estonia             2
# Bulgaria            2
# Romania             1
# Poland              1
# Finland             1
# Portugal            1
# Belgium             1
# Denmark             1
# Canada              1
