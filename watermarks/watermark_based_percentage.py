import math
import numpy as np
import spacy
from tqdm import tqdm
from . import basic_watermark
from .basic_watermark import tokenize, calculate_entropy, bottom_k_entropy_words
from utils.create_map_entropy_both import create_entropy_map
from utils.dict_functions import write_dict_to_file
nlp_spacy = spacy.load("en_core_web_sm")

def add_watermark_lower(p, mode="BookMIA", synonym_method="context", syn_threshold=0.6):
    def watermarked_sentences(data, output_file=None):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words

        # for sentences in data:
        entropy_map = create_entropy_map(data, mode=mode)
        for text in data:
            # new_text = replace_with_higher_entropy(text)
            # new_text, replaced_dict = replace_lowest_top_k_entropy_with_higher_entropy_threshold(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, model=model1, tokenizer=tokenizer1, threshold=threshold)
            new_text, replaced_dict = replace_lowest_p_percentage_entropy_with_higher_entropy(text, replaced_dict=replaced_dict, entropy_map=entropy_map, p=p, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file)
        return new_sentences
    return watermarked_sentences


def add_watermark_higher(p, mode="BookMIA", synonym_method="context", output_file=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words
        # for sentences in data:
        print("Creating entropy map...")
        entropy_map = create_entropy_map(data, mode=mode)
        for text in tqdm(data):
            new_text, replaced_dict = replace_highest_p_percentage_entropy_with_higher_entropy(text, replaced_dict=replaced_dict, entropy_map=entropy_map, p=p, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences
    return watermarked_sentences


def add_watermark_random(p, synonym_method="context", seed=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}
        # for sentences in data:
        for text in data:
            new_text, replaced_dict = replace_random_p_percentage_in_higher_entropy(text, replace_percentage=p, synonym_method=synonym_method, seed=seed, syn_threshold=syn_threshold, replaced_dict=replaced_dict)
            new_sentences.append(new_text)
        return new_sentences
    return watermarked_sentences


def replace_lowest_p_percentage_entropy_with_higher_entropy(text, replaced_dict={}, entropy_map={}, p=0.2, synonym_method="context", syn_threshold=0.6):
    # Tokenize the text into sentences
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Initialize a list to store the modified sentences
    modified_sentences = []

    for sentence in sentences:
        words = sentence.split()
        # words = tokenize(sentence)
        total_words = len(words)
        num_to_replace = int(total_words * p)

        # Ensure num_to_replace does not exceed the number of words
        if num_to_replace > total_words:
            num_to_replace = total_words

        words_to_replace = basic_watermark.lower_k_entropy_words(sentence, entropy_map, num_to_replace)

        new_words = words.copy()
        for i, word in enumerate(words_to_replace):
            idx = new_words.index(word)
            if word in replaced_dict:
                new_words[idx] = replaced_dict[word]
                continue
            best_word = basic_watermark.find_highset_entropy_synonym(sentence, word, synonym_method,
                                                                     syn_threshold=syn_threshold)
            new_words[idx] = best_word
            replaced_dict[word] = best_word
        # Join the modified words back into a sentence
        modified_sentence = ' '.join(new_words)
        modified_sentences.append(modified_sentence)

    # Join all the modified sentences into the final text
    modified_text = ' '.join(modified_sentences)
    return modified_text, replaced_dict


def replace_highest_p_percentage_entropy_with_higher_entropy(text, replaced_dict={}, entropy_map={}, p=0.2, synonym_method="context", syn_threshold=0.6):
    # Tokenize the text into sentences
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Initialize a list to store the modified sentences
    modified_sentences = []

    for sentence in sentences:
        words = sentence.split()
        # words = tokenize(sentence)
        total_words = len(words)
        # num_to_replace = int(total_words * p)
        num_to_replace = math.ceil(total_words * p)

        # Ensure num_to_replace does not exceed the number of words
        if num_to_replace > total_words:
            num_to_replace = total_words

        # words_to_replace = np.random.choice(total_words, num_to_replace, replace=False)
        words_to_replace = bottom_k_entropy_words(sentence, entropy_map, num_to_replace)

        new_words = words.copy()
        for i, word in enumerate(words_to_replace):
            idx = new_words.index(word)
            # word = words[idx]
            # synonyms = get_synonyms(word)
            if word in replaced_dict:
                new_words[idx] = replaced_dict[word]
                continue
            best_word = basic_watermark.find_highset_entropy_synonym(sentence, word, synonym_method,
                                                                     syn_threshold=syn_threshold)
            new_words[idx] = best_word
            replaced_dict[word] = best_word
        # Join the modified words back into a sentence
        modified_sentence = ' '.join(new_words)
        modified_sentences.append(modified_sentence)

    # Join all the modified sentences into the final text
    modified_text = ' '.join(modified_sentences)
    return modified_text, replaced_dict



def replace_random_p_percentage_in_higher_entropy(text, replace_percentage=0.6, synonym_method="context", seed=None, syn_threshold=0.6, replaced_dict={}):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    nlp_spacy = spacy.load("en_core_web_sm")

    # Tokenize the text into sentences
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Initialize a list to store the modified sentences
    modified_sentences = []

    for sentence in sentences:
        words = sentence.split()
        # words = tokenize(sentence)
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
            if word in replaced_dict:
                new_words[idx] = replaced_dict[word]
                continue
            best_word = basic_watermark.find_highset_entropy_synonym(sentence, word, synonym_method,
                                                                     syn_threshold=syn_threshold)
            new_words[idx] = best_word
            replaced_dict[word] = best_word
        # Join the modified words back into a sentence
        modified_sentence = ' '.join(new_words)
        modified_sentences.append(modified_sentence)

    # Join all the modified sentences into the final text
    modified_text = ' '.join(modified_sentences)
    return modified_text, replaced_dict