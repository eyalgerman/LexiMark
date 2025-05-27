import string
import math
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import re
from . import process_data
from watermarks.basic_watermark import bottom_k_entropy_words
from wordfreq import word_frequency

def calculate_entropy(word_freq, total_words):
    """
    Calculates the entropy of a word given its frequency and total word count.

    Args:
        word_freq (int): Frequency of the word.
        total_words (int): Total number of words in the corpus.

    Returns:
        float: Entropy value of the word.
    """
    probability = word_freq / total_words
    return -probability * math.log2(probability)


def generate_entropy_map_wordfreq(words, lang='en'):
    """
    Generates an entropy map for given words using `wordfreq` probabilities.

    Args:
        words (Iterable[str]): List or set of words to process.
        lang (str): Language code used by `wordfreq`. Default is 'en' (English).

    Returns:
        dict: A dictionary mapping words to their entropy values.
    """
    entropy_map = {}
    for word in words:
        freq = word_frequency(word, lang)
        if freq > 0:
            entropy_map[word] = -freq * math.log2(freq)
    return entropy_map


def create_entropy_map(data_sources, mode='Books'):
    """
    Creates an entropy map from text data using word frequencies.

    Args:
        data_sources (list | dict): Input text sources. Can be file paths, strings, or datasets depending on mode.
        mode (str): Processing mode. Supported: 'Books', 'WikiMIA', 'BookMIA', 'Arxiv', 'ECHR', 'PILE', 'Text(s)'.

    Returns:
        dict: A dictionary mapping each word to its entropy value.
    """
    word_counts = Counter()

    if mode == 'Arxiv' or mode == 'ECHR':
        print(f"Started processing for mode: {mode}")
        # db_data = procces_data.load_data(mode="Arxiv")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)
    elif mode == 'Books':
        for file_path in data_sources:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    words = line.strip().split()
                    word_counts.update(words)
    elif mode == 'WikiMIA' or mode == 'BookMIA' or mode == "Gut":
        print(f"Started processing for mode: {mode}")
        db_data = process_data.load_data(mode=mode)[0]
        for item in db_data:
            text_field = 'input' if mode == 'WikiMIA' else 'snippet'
            words = item[text_field].split()
            word_counts.update(words)
    elif mode == 'PILE' or mode[:4].lower() == 'pile':
        print(f"Started processing for mode: {mode}")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)
    elif mode == 'Text' or mode == 'Texts':
        # For the Text mode, we assume data_sources is a dict of name of file and the text of it or list of texts
        if isinstance(data_sources, str):
            words = data_sources.split()
            word_counts.update(words)
            # For the Text mode, we assume data_sources is a dict of name of file and the text of it or list of texts
        elif isinstance(data_sources, dict):
            for text in data_sources.values():
                words = text.split()
                word_counts.update(words)
        elif isinstance(data_sources, list):
            for text in data_sources:
                if isinstance(text, dict):
                    words = text["input"].split()
                    word_counts.update(words)
                else:
                    words = text.split()
                    word_counts.update(words)
        else:
            raise ValueError(f"Unsupported data_sources format: {type(data_sources)}. Expected str, dict, or list.")
    else:
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)

    total_words = sum(word_counts.values())
    # Generate the entropy map using word frequency
    entropy_map = generate_entropy_map_wordfreq(word_counts.keys())
    # entropy_map = {word: calculate_entropy(freq, total_words) for word, freq in word_counts.items()}

    return entropy_map


def save_entropy_map(entropy_map, filename):
    """
    Saves the entropy map to a file.

    Args:
        entropy_map (dict): Dictionary of word entropy values.
        filename (str): Path to the output file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for word, entropy in entropy_map.items():
            file.write(f"{word} {entropy}\n")


def load_entropy_map(filename):
    """
    Loads an entropy map from a file.

    Args:
        filename (str): Path to the saved entropy file.

    Returns:
        dict: Dictionary of word entropy values.
    """
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def sort_entropy_map(entropy_map, descending=True):
    """
    Sorts an entropy map based on entropy values.

    Args:
        entropy_map (dict): Dictionary of entropy values.
        descending (bool): Whether to sort from highest to lowest. Default is True.

    Returns:
        List[Tuple[str, float]]: Sorted list of (word, entropy) tuples.
    """
    return sorted(entropy_map.items(), key=lambda item: item[1], reverse=descending)


def get_text_files(folder_path):
    """
    Retrieves all .txt files from a directory (recursively).

    Args:
        folder_path (str): Path to the root folder.

    Returns:
        List[str]: List of file paths ending with '.txt'.
    """
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]


def strip_punctuation(word):
    """
    Strips leading and trailing punctuation from a word.

    Args:
        word (str): Input word.

    Returns:
        str: Cleaned word without punctuation.
    """
    return word.strip(string.punctuation)


def create_line_to_top_words_map(text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY,
                                 nlp_spacy):
    """
    Creates a mapping from line numbers to top-k high-entropy words for each valid sentence.

    Args:
        text (str): Raw input text.
        entropy_map (dict): A precomputed word-to-entropy mapping.
        MAX_LEN_LINE_GENERATE (int): Maximum sentence length to include.
        MIN_LEN_LINE_GENERATE (int): Minimum sentence length to include.
        TOP_K_ENTROPY (int): Number of top-entropy words to extract per line.
        nlp_spacy (spacy.lang): spaCy language model for sentence segmentation and NER.

    Returns:
        Tuple[dict, list]:
            - Mapping from line number to list of top-entropy and named-entity words.
            - List of valid sentences.
    """
    # text = text.replace('\n', '')
    doc = nlp_spacy(text)
    # Debugging: convert iterator to list to check content
    all_sentences = list(doc.sents)
    sentences = [sent.text.strip() for sent in all_sentences if
                 MIN_LEN_LINE_GENERATE <= len(sent.text.split()) <= MAX_LEN_LINE_GENERATE]

    line_to_top_words_map = {}

    for line_num, line in enumerate(sentences, 1):
        if line.strip():
            top_k_words = {re.sub(r'^\W+|\W+$', '', word.strip(string.punctuation)) for word in
                           bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY) if ' ' not in word}

            ners = {strip_punctuation(ent.text) for ent in doc.ents if ' ' not in ent.text and ent.sent.text == line}
            unique_words = top_k_words.union(ners)
            line_to_top_words_map[line_num] = list(unique_words)

    return line_to_top_words_map, sentences
