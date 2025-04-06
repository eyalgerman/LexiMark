from synonyms_methods import synonym_main
from utils.create_map_entropy_both import create_entropy_map
from .basic_watermark import calculate_entropy, bottom_k_entropy_words
import spacy
from tqdm import tqdm
import random
from utils.dict_functions import write_dict_to_file


def replace_highest_top_k_entropy_with_random(text, replaced_dict, entropy_map, k_value, synonym_method, seed=None, syn_threshold=0.6):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    nlp_spacy = spacy.load("en_core_web_sm")

    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Create a list of bottom k words for each sentence
    all_bottom_k_words = {}
    for sentence in sentences:
        # highest entropy words
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

                # If the word has already been replaced, use the previous replacement
                if original_word in replaced_dict:
                    new_words[idx] = replaced_dict[original_word]
                    continue

                # Get synonyms for the word using the specified synonym method
                synonyms = synonym_main.get_synonyms_by_different_methods(sentence, original_word, synonym_method, syn_threshold=syn_threshold)

                if synonyms:
                    # Randomly choose a synonym
                    random_synonym = random.choice(list(synonyms))
                    new_words[idx] = random_synonym
                    replaced_dict[original_word] = random_synonym

    modified_text = ' '.join(new_words)

    return modified_text, replaced_dict


def replace_highest_top_k_entropy_with_lowest(text, replaced_dict, entropy_map, k_value, synonym_method, syn_threshold=0.6):
    nlp_spacy = spacy.load("en_core_web_sm")

    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    # Create a list of bottom k words for each sentence
    all_bottom_k_words = {}
    for sentence in sentences:
        # highest entropy words
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

                # If the word has already been replaced, use the previous replacement
                if original_word in replaced_dict:
                    new_words[idx] = replaced_dict[original_word]
                    continue
                # else:
                #     print(f"Not in dict, Original word: {original_word}")

                # Get synonyms for the word using the specified synonym method
                synonyms = synonym_main.get_synonyms_by_different_methods(sentence, original_word, synonym_method, syn_threshold=syn_threshold)
                if not synonyms:
                    continue
                # Initialize with the original word's entropy
                min_entropy = calculate_entropy(original_word)
                best_word = original_word

                # Iterate over the synonyms to find the one with the lowest entropy
                for synonym in synonyms:
                    entropy = calculate_entropy(synonym)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_word = synonym

                new_words[idx] = best_word
                replaced_dict[original_word] = best_word
                # if original_word != best_word:
                #     print(f"Original word: {original_word}, Best word: {best_word}")

    modified_text = ' '.join(new_words)

    return modified_text, replaced_dict


def replace_random_top_k_to_random_synonyms(text, replaced_dict, entropy_map, k_value, synonym_method, seed=42, syn_threshold=5):
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

            # Get synonyms for the word using the specified synonym method
            synonyms = synonym_main.get_synonyms_by_different_methods(sentence, original_word, synonym_method, syn_threshold=syn_threshold)
            # check if synonyms is empty
            if not synonyms:
                best_word = original_word
            else:
                # Randomly choose a synonym
                best_word = random.choice(list(synonyms))

            words[idx] = best_word
            # if original_word == best_word:
            replaced_dict[original_word] = best_word

        # Join the modified words back into a sentence
        modified_sentence = ' '.join(words)
        modified_sentences.append(modified_sentence)

    # Join all the modified sentences into the final text
    modified_text = ' '.join(modified_sentences)
    return modified_text, replaced_dict


def add_watermark_highest_to_random(k, mode="BookMIA", synonym_method="context", output_file=None, seed=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words

        # for sentences in data:
        entropy_map = create_entropy_map(data, mode=mode)
        for text in tqdm(data):
            # new_text = replace_with_higher_entropy(text)
            # new_text, replaced_dict = replace_lowest_top_k_entropy_with_higher_entropy_threshold(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, model=model1, tokenizer=tokenizer1, threshold=threshold)
            new_text, replaced_dict = replace_highest_top_k_entropy_with_random(text, replaced_dict=replaced_dict,
                                                                                entropy_map=entropy_map, k_value=k,
                                                                                synonym_method=synonym_method, seed=seed, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences

    return watermarked_sentences


def add_watermark_highest_to_lowest(k, mode="BookMIA", synonym_method="context", output_file=None, syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words

        # for sentences in data:
        entropy_map = create_entropy_map(data, mode=mode)
        for text in tqdm(data):
            # new_text = replace_with_higher_entropy(text)
            # new_text, replaced_dict = replace_lowest_top_k_entropy_with_higher_entropy_threshold(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, model=model1, tokenizer=tokenizer1, threshold=threshold)
            new_text, replaced_dict = replace_highest_top_k_entropy_with_lowest(text, replaced_dict=replaced_dict,
                                                                                entropy_map=entropy_map, k_value=k,
                                                                                synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences

    return watermarked_sentences


def add_watermark_random_to_random(k, mode="BookMIA", synonym_method="context", output_file=None, seed=None, syn_threshold=5):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words

        # for sentences in data:
        entropy_map = create_entropy_map(data, mode=mode)
        for text in tqdm(data):
            new_text, replaced_dict = replace_random_top_k_to_random_synonyms(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k, synonym_method=synonym_method, seed=seed, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences

    return watermarked_sentences
