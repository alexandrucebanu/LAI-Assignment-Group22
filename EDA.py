import pandas as pd
import ast  # To safely evaluate strings as Python expressions
import re

# Import data
df = pd.read_csv('nationality.csv')


# Convert the 'tokens' column from strings to lists
#df['tokens'] = df['tokens'].apply(ast.literal_eval)


# Function to find 'born' and its neighbors
def find_born_and_neighbors(tokens, target='born', window=4):
    neighbors = []
    for i, token in enumerate(tokens):
        if token == target:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            neighbors.append(tokens[start:end])
    return neighbors


def find_phrase_and_neighbors(tokens, target=('born', 'in'), window=4):
    neighbors = []
    target_len = len(target)
    for i in range(len(tokens) - target_len + 1):
        # Check if the sequence matches the target phrase
        if tokens[i:i + target_len] == list(target):
            start = max(0, i - window)
            end = min(len(tokens), i + target_len + window)
            neighbors.append(tokens[start:end])
    return neighbors


# Apply the function to each row in the DataFrame

import re

def find_phrase_with_regex(post, target='born in', window=4):
    # Build regex pattern
    pattern = (
        r'(\S+(?:\s+\S+){0,' + str(window - 1) + r'})?\s*'  # Up to 4 words before
        + re.escape(target)                                # Match the target phrase
        + r'(?:\s*\S+){1,' + str(window) + r'}'            # Exactly 4 words after
    )
    matches = re.finditer(pattern, post)  # Use `finditer` for better debugging and capturing groups
    results = []
    for match in matches:
        before = match.group(1) if match.group(1) else ""  # Words before
        full_match = before + " " + target + match.group(0).split(target)[-1]  # Full context
        results.append(full_match.strip())
    return results




# Apply the regex-based function to each row in the 'post' column
df['phrase_neighbors'] = df['post'].apply(find_phrase_with_regex, target='born in')

# Display results for rows where the phrase was found
results = df[df['phrase_neighbors'].apply(len) > 0]

if not results.empty:
    print(f"The phrase 'born in' appears in {len(results)} rows.", flush=True)
    for idx, row in results.iterrows():
        print(f"Row {idx}: {row['phrase_neighbors']}", flush=True)
