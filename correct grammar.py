import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import language_tool_python

# ======================================================================================================================
# INSTRUCTIONS
#
# change input csv in line 24
# change name of output csv in line 48
# ======================================================================================================================

# Seed for reproducibility
DetectorFactory.seed = 0

# Function to detect the language of text
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# Read the CSV file
df = pd.read_csv("cleaned_train_split_2_Ambra.csv")

# Detect the language and create relevant columns
df['language'] = df['cleaned_text'].apply(detect_language)
df['english'] = df['language'].apply(lambda x: 1 if x == 'en' else 0)

# Initialize the language correction tool
tool = language_tool_python.LanguageTool('en-US')

# Function to correct grammar
def correct_grammar(text):
    return tool.correct(text)

# Function to apply grammar correction or retain original text
def grammar_correction(row):
    if row['english'] == 1:
        return correct_grammar(row['cleaned_text'])
    else:
        return row['cleaned_text']

# Create a new column for grammar corrected text
df['grammar_corrected_text'] = df.apply(grammar_correction, axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv("cleaned_train_split_2_Ambra_with_grammar_correction.csv", index=False)
