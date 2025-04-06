import random
import spacy
import time
from tqdm import tqdm
from . import basic_watermark
from utils.create_map_entropy_both import create_entropy_map
from .basic_watermark import bottom_k_entropy_words
from utils.dict_functions import write_dict_to_file

MAX_LEN_LINE_GENERATE = 40
MIN_LEN_LINE_GENERATE = 7
# TOP_K_ENTROPY = 5
MODEL_NAME = "huggyllama/llama-7b"
import matplotlib.pyplot as plt

def plot_first_probabilities(prob_dict):
    """
    Plot the first probabilities for each word in the given dictionary.

    Parameters:
    prob_dict (dict): A dictionary where keys are words and values are probabilities.
    """
    # Create lists of words and their corresponding probabilities
    words = list(prob_dict.keys())
    probabilities = list(prob_dict.values())

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(words, probabilities, color='blue')

    # Add titles and labels
    plt.title('First Probabilities for Each Word')
    plt.xlabel('Words')
    plt.ylabel('Probability')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


def plot_first_probabilities_scatter(prob_dict):
    """
    Plot the first probabilities for each word in the given dictionary as a scatter plot.

    Parameters:
    prob_dict (dict): A dictionary where keys are words and values are probabilities.
    """
    # Create lists of words and their corresponding probabilities
    words = list(prob_dict.keys())
    probabilities = list(prob_dict.values())

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(words, probabilities, color='blue')

    # Add titles and labels
    plt.title('First Probabilities for Each Word')
    plt.xlabel('Words')
    plt.ylabel('Probability')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


def add_watermark_lower(k, mode="BookMIA", synonym_method="context", syn_threshold=0.6):
    def watermarked_sentences(data, output_file=None):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words
        # for sentences in data:
        entropy_map = create_entropy_map(data, mode=mode)
        for text in data:
            new_text, replaced_dict = replace_lowest_top_k_entropy_with_higher_entropy(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file)
        return new_sentences
    return watermarked_sentences


def add_watermark_higher(k, mode="BookMIA", synonym_method="context", output_file=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words
        # for sentences in data:
        print("Creating entropy map...")
        entropy_map = create_entropy_map(data, mode=mode)
        start_time = time.time()  # Start timing
        doc_times = []

        for i, text in enumerate(tqdm(data), 1):
            doc_start = time.time()
            new_text, replaced_dict = replace_higher_top_k_entropy_with_higher_entropy(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)
            doc_end = time.time()
            doc_times.append(doc_end - doc_start)

            if i % 10 == 0:
                avg_time_so_far = sum(doc_times) / len(doc_times)
                print(f"Processed {i} documents. Average time per document: {avg_time_so_far:.4f} seconds")

        end_time = time.time()  # End timing
        total_time = end_time - start_time
        avg_time_per_doc = total_time / len(data) if data else 0
        print(f"\nTotal watermarking time: {total_time:.2f} seconds")
        print(f"Average time per document: {avg_time_per_doc:.4f} seconds")

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences
    return watermarked_sentences


def add_watermark_random(k=5, synonym_method="context", seed=None, output_file=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}
        # for sentences in data:
        for text in data:
            new_text, replaced_dict = replace_random_k_words_in_each_sentence(text, k, synonym_method=synonym_method, seed=seed, syn_threshold=syn_threshold, replaced_dict=replaced_dict)
            new_sentences.append(new_text)
        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)

        return new_sentences
    return watermarked_sentences


def replace_lowest_top_k_entropy_with_higher_entropy(text, replaced_dict={}, entropy_map={}, k_value=5, synonym_method="context", syn_threshold=0.6):
    nlp_spacy = spacy.load("en_core_web_sm")

    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Create a list of bottom k words for each sentence
    all_bottom_k_words = {}
    for sentence in sentences:
        bottom_k_words = basic_watermark.lower_k_entropy_words(sentence, entropy_map, k_value)
        all_bottom_k_words[sentence] = bottom_k_words

    # Replace the lowest k entropy words
    words = text.split()
    new_words = words.copy()

    for sentence, bottom_k_words in all_bottom_k_words.items():
        for word in bottom_k_words:
            if word in new_words:
                idx = new_words.index(word)
                original_word = new_words[idx]
                if original_word in replaced_dict:
                    new_words[idx] = replaced_dict[original_word]
                    continue
                best_word = basic_watermark.find_highset_entropy_synonym(sentence, original_word, synonym_method,
                                                                         syn_threshold=syn_threshold)
                new_words[idx] = best_word
                replaced_dict[original_word] = best_word

    modified_text = ' '.join(new_words)
    # replaced_words, num_replaced_words = compare_texts(text, modified_text)
    # print(f"Replaced {num_replaced_words} words.")
    return modified_text, replaced_dict


def replace_higher_top_k_entropy_with_higher_entropy(text, replaced_dict={}, entropy_map={}, k_value=5, synonym_method="context", syn_threshold=0.6):
    nlp_spacy = spacy.load("en_core_web_sm")

    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Create a list of bottom k words for each sentence
    all_bottom_k_words = {}
    for sentence in sentences:
        bottom_k_words = bottom_k_entropy_words(sentence, entropy_map, k_value)
        all_bottom_k_words[sentence] = bottom_k_words

    # Replace the lowest k entropy words
    words = text.split()
    new_words = words.copy()

    for sentence, bottom_k_words in all_bottom_k_words.items():
        for word in bottom_k_words:
            if word in new_words:
                idx = new_words.index(word)
                original_word = new_words[idx]
                if original_word in replaced_dict:
                    new_words[idx] = replaced_dict[original_word]
                    continue

                best_word = basic_watermark.find_highset_entropy_synonym(sentence, original_word, synonym_method, syn_threshold=syn_threshold)
                new_words[idx] = best_word
                replaced_dict[original_word] = best_word

    modified_text = ' '.join(new_words)
    # replaced_words, num_replaced_words = compare_texts(text, modified_text)
    # print(f"Replaced {num_replaced_words} words.")
    return modified_text, replaced_dict


def replace_random_k_words_in_each_sentence(text, k_value, replaced_dict={}, synonym_method="context", seed=None, syn_threshold=0.6):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    nlp_spacy = spacy.load("en_core_web_sm")

    # Tokenize the text into sentences
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Initialize a list to store the modified sentences
    modified_sentences = []

    for sentence in sentences:
        words = sentence.split()

        # Ensure we don't try to replace more words than available
        k_value = min(k_value, len(words))

        # Select k random words from the sentence
        random_indices = random.sample(range(len(words)), k_value)

        for idx in random_indices:
            original_word = words[idx]

            # Ensure original_word is a string before checking in the dictionary
            if isinstance(original_word, str) and original_word in replaced_dict:
                words[idx] = replaced_dict[original_word]
                continue

            best_word = basic_watermark.find_highset_entropy_synonym(sentence, original_word, synonym_method,
                                                                     syn_threshold=syn_threshold)
            words[idx] = best_word
            # if original_word == best_word:
            replaced_dict[original_word] = best_word

        # Join the modified words back into a sentence
        modified_sentence = ' '.join(words)
        modified_sentences.append(modified_sentence)

    # Join all the modified sentences into the final text
    modified_text = ' '.join(modified_sentences)
    return modified_text, replaced_dict


if __name__ == "__main__":
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Create a sample text
    text = "The quick brown fox jumps over the lazy dog. The dog is very lazy."
    print("Original Text:", text)

    # Replace the lowest k entropy words with higher entropy words
    # modified_text, _ = replace_lowest_top_k_entropy_with_higher_entropy(text, k_value=5, synonym_method="context", syn_threshold=0.9)
    # print("Modified Text (Lowest K Entropy):", modified_text)

    # Replace the highest k entropy words with higher entropy words
    modified_text, _ = replace_higher_top_k_entropy_with_higher_entropy(text, k_value=5, synonym_method="lexsub_concatenation", syn_threshold=3)
    print("Modified Text (Highest K Entropy):", modified_text)

    # Replace random k words with higher entropy words
    # modified_text, _ = replace_random_k_words_in_each_sentence(text, k_value=5, synonym_method="context", syn_threshold=0.8)
    # print("Modified Text (Random K Words):", modified_text)

