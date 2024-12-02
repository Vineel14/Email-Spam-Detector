import re
from collections import Counter
import json
import numpy as np
import os

# Define base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load spam keywords from the spambase.names file
file_path = os.path.join(BASE_DIR, "data/spambase.names")  # Relative file path

# Read and extract spam-related words and characters from the .names file
with open(file_path, 'r') as file:
    spambase_names = file.readlines()

# Extract valid spam-related words, excluding "table"
spam_words = [
    line.split(":")[0].replace("word_freq_", "").strip()
    for line in spambase_names
    if line.startswith("word_freq_") and "word_freq_table" not in line
]

# Extract valid spam-related characters
spam_chars = [
    line.split(":")[0].replace("char_freq_", "").strip()
    for line in spambase_names
    if line.startswith("char_freq_")
]

# Precompile regex patterns for efficiency
word_pattern = re.compile(r'\b\w+\b')  # Matches words
capital_pattern = re.compile(r'[A-Z]+')  # Matches sequences of capital letters

def analyze_spam_text(text, spam_words, spam_chars):
    """
    Analyzes input text for spam indicators based on the spambase dataset.

    Parameters:
    text (str): The input text to analyze.
    spam_words (list): List of spam-related words to look for in the text.
    spam_chars (list): List of spam-related characters to look for in the text.

    Returns:
    dict: Dictionary with spam indicator metrics.
    """
    # Tokenize text into words and calculate word counts
    words = word_pattern.findall(text.lower())
    total_words = len(words)
    word_counts = Counter(words)

    # Calculate word frequency percentages for spam words
    word_frequencies = {
        f"word_freq_{word}": 100 * (word_counts[word] / total_words) if total_words > 0 else 0
        for word in spam_words
    }

    # Calculate character frequency percentages for spam characters
    total_chars = len(text)
    char_frequencies = {
        f"char_freq_{char}": 100 * (text.count(char) / total_chars) if total_chars > 0 else 0
        for char in spam_chars
    }

    # Capital letter analysis
    capital_sequences = capital_pattern.findall(text)
    capital_lengths = [len(seq) for seq in capital_sequences]
    capital_run_length_average = sum(capital_lengths) / len(capital_lengths) if capital_lengths else 0
    capital_run_length_longest = max(capital_lengths) if capital_lengths else 0
    capital_run_length_total = sum(capital_lengths)

    # Combine all features into a dictionary
    spam_indicators = {
        **word_frequencies,
        **char_frequencies,
        "capital_run_length_average": capital_run_length_average,
        "capital_run_length_longest": capital_run_length_longest,
        "capital_run_length_total": capital_run_length_total
    }

    return spam_indicators

def scale_vector(input_vector, feature_min, feature_max):
    """
    Scales an input vector using the min-max scaling formula.

    Parameters:
    input_vector (list or np.array): The input vector to be scaled.
    feature_min (np.array): Array of minimum values for each feature.
    feature_max (np.array): Array of maximum values for each feature.

    Returns:
    np.array: Scaled vector.
    """
    input_vector = np.array(input_vector)  # Ensure it's a numpy array
    # Handle division by zero where min == max
    scaled_vector = np.where(
        feature_max > feature_min,  # Avoid division by zero
        (input_vector - feature_min) / (feature_max - feature_min),
        0  # Set scaled value to 0 if min == max
    )
    return scaled_vector

# Sample usage
sample_text = """
Congratulations! You have won a free prize! To claim, send your address and contact information to our email.
Act now to secure this exclusive offer and receive your money!
"""

# Analyze the text to generate the input vector
spam_indicators = analyze_spam_text(sample_text, spam_words, spam_chars)
input_vector = list(spam_indicators.values())

# Print the input vector
print("Input Vector:")
print(input_vector)

# Print the size of the input vector
print("\nSize of Input Vector:")
print(len(input_vector))

# Load the min and max values from the JSON file
json_path = os.path.join(BASE_DIR, "data/feature_min_max.json")  # Relative path
with open(json_path, "r") as f:
    min_max_values = json.load(f)

feature_min = np.array(min_max_values["min"])  # Convert to numpy array
feature_max = np.array(min_max_values["max"])  # Convert to numpy array

# Scale the input vector
scaled_vector = scale_vector(input_vector, feature_min, feature_max)

# Print the scaled vector
print("\nScaled Input Vector:")
print(scaled_vector)

# Print the size of the scaled vector
print("\nSize of Scaled Input Vector:")
print(len(scaled_vector))
