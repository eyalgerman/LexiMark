from random import random
import math
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
import math
from wordfreq import word_frequency
from synonyms_methods import synonym_main


def add_watermark(replace_percentage=0.4, synonym_method="context", syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        # for sentences in data:
        for text in data:
            # new_text = replace_with_higher_entropy(text)
            new_text = replace_lowest_entropy_with_higher_entropy(text, replace_percentage=0.3, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)
        return new_sentences
    return watermarked_sentences


def add_watermark_random(replace_percentage=0.4, synonym_method="context", seed=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        # for sentences in data:
        for text in data:
            # new_text = replace_with_higher_entropy(text)
            new_text = replace_random_with_higher_entropy(text, replace_percentage, synonym_method=synonym_method, seed=seed, syn_threshold=syn_threshold)
            new_sentences.append(new_text)
        return new_sentences
    return watermarked_sentences


def load_synonyms_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    pass


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def tokenize(text):
    return nltk.word_tokenize(text)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def calculate_entropy(word):
    prob = word_frequency(word, 'en')
    if prob == 0:
        return 0
    return -math.log2(prob)


def lower_k_entropy_words(line, entropy_map, top_k):
    words_in_line = line.split()
    return sorted(words_in_line, key=lambda word: entropy_map.get(word, float('-inf')), reverse=True)[:top_k]


def find_highset_entropy_synonym(sentence, original_word, synonym_method, syn_threshold=0.6):
    synonyms = synonym_main.get_synonyms_by_different_methods(sentence, original_word, synonym_method, syn_threshold=syn_threshold)
    if not synonyms:  # Skip if no synonyms are found
        return original_word
    max_entropy = calculate_entropy(original_word)
    best_word = original_word
    for synonym in synonyms:
        entropy = calculate_entropy(synonym)
        if entropy > max_entropy:
            max_entropy = entropy
            best_word = synonym
    return best_word

def replace_random_with_higher_entropy(text, replace_percentage=0.6, synonym_method="context", seed=None, syn_threshold=0.6):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    words = tokenize(text)
    total_words = len(words)
    num_to_replace = int(total_words * replace_percentage)

    # Ensure num_to_replace does not exceed the number of words
    if num_to_replace > total_words:
        num_to_replace = total_words

    words_to_replace = np.random.choice(total_words, num_to_replace, replace=False)

    new_words = words.copy()
    for idx in words_to_replace:
        word = words[idx]
        # synonyms = get_synonyms(word)
        synonyms = synonym_main.get_synonyms_by_different_methods(text, word, synonym_method, threshold=syn_threshold)
        max_entropy = calculate_entropy(word)
        best_word = word
        for synonym in synonyms:
            entropy = calculate_entropy(synonym)
            if entropy > max_entropy:
                max_entropy = entropy
                best_word = synonym
        new_words[idx] = best_word
    return ' '.join(new_words)


def replace_lowest_entropy_with_higher_entropy(text, replace_percentage=0.3, synonym_method="context", syn_threshold=0.6):
    words = tokenize(text)
    total_words = len(words)
    num_to_replace = int(total_words * replace_percentage)

    # Ensure num_to_replace does not exceed the number of words
    if num_to_replace > total_words:
        num_to_replace = total_words

    # Calculate entropy for each word
    entropy_list = [(idx, calculate_entropy(word)) for idx, word in enumerate(words)]

    # Sort words by entropy (ascending order)
    entropy_list.sort(key=lambda x: x[1])

    # Select the indices of words with the lowest entropy
    words_to_replace = [idx for idx, entropy in entropy_list[:num_to_replace]]

    new_words = words.copy()
    for idx in words_to_replace:
        word = words[idx]
        # synonyms = get_synonyms(word)
        synonyms = synonym_main.get_synonyms_by_different_methods(text, word, synonym_method, threshold=syn_threshold)
        max_entropy = calculate_entropy(word)
        best_word = word
        for synonym in synonyms:
            entropy = calculate_entropy(synonym)
            if entropy > max_entropy:
                max_entropy = entropy
                best_word = synonym
        new_words[idx] = best_word

    return ' '.join(new_words)


def bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY):
    """
    Selects the top-k (highest entropy) words from a given line of text.

    Args:
        line (str): A single line of text to analyze.
        entropy_map (dict): A mapping from words to their entropy values.
        TOP_K_ENTROPY (int): The number of words with the highest entropy to return.

    Returns:
        List[str]: A list of words with the highest entropy from the input line.
    """
    words_in_line = line.split()
    top_k = int(TOP_K_ENTROPY)

    return sorted(words_in_line, key=lambda word: entropy_map.get(word, float('inf')))[:top_k]
