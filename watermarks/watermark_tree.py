
import spacy
from wordfreq import word_frequency
from utils import Tree
from synonyms_methods import synonym_main
from utils.create_map_entropy_both import create_entropy_map, create_line_to_top_words_map
from .basic_watermark import calculate_entropy, tokenize, bottom_k_entropy_words
from watermark_detection.watermark_detection_2 import calculate_perplexity, load_local_model, load_model
from utils.dict_functions import write_dict_to_file

MAX_LEN_LINE_GENERATE = 40
MIN_LEN_LINE_GENERATE = 7
# TOP_K_ENTROPY = 5
MODEL_NAME = "huggyllama/llama-7b"
import matplotlib.pyplot as plt

def add_watermark(k, threshold, output_file, synonym_method="context", syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words
        # load model and data
        model1, tokenizer1 = load_model(MODEL_NAME)

        # for sentences in data:
        entropy_map = create_entropy_map(data, mode="BookMIA")
        root_tree = Tree.TreeNode("", "", 0)
        for text in data:
            # new_text = replace_with_higher_entropy(text)
            new_text, replaced_dict = replace_lowest_top_k_entropy_with_higher_entropy_tree(text, root_tree=root_tree, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k,
                                    model=model1, tokenizer=tokenizer1, threshold=threshold, synonym_method=synonym_method, syn_threshold=syn_threshold)
            new_sentences.append(new_text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
            output_file2 = output_file.replace(".csv", "_tree.json")
            root_tree.save_to_file(output_file2)
            print(f"Tree saved successfully to {output_file2}")
        # print("Tree:")
        # root_tree.print_tree()
        return new_sentences
    return watermarked_sentences


def replace_lowest_top_k_entropy_with_higher_entropy_tree(text, replaced_dict={}, entropy_map={}, k_value=5, model=None, tokenizer=None, threshold=-5, root_tree=None, synonym_method="context", syn_threshold=0.6):
    nlp_spacy = spacy.load("en_core_web_sm")
    # calculate perplexity for each word based on the model
    p1, all_prob, p1_likelihood, logits, input_ids_processed = calculate_perplexity(text, model, tokenizer, gpu=model.device)


    line_to_top_words_map, sentences = create_line_to_top_words_map(
        text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, k_value, nlp_spacy
    )
    # Create a list of bottom k words once
    all_bottom_k_words = {}

    for line_num, top_words in line_to_top_words_map.items():
        bottom_k_words = bottom_k_entropy_words(" ".join(top_words), entropy_map, k_value)
        all_bottom_k_words[line_num] = bottom_k_words

    # print(all_bottom_k_words)

    # Intermediate storage for results
    intermediate_results = {
        "relevant_log_probs": [],
        "relevant_log_probs_words": {},
        "relevant_log_probs_one_token": [],
        "relevant_indexes": []
    }
    tokens = tokenizer.tokenize(text)
    concatenated_tokens = "".join(token for token in tokens)


    # Process bottom k words once, ensuring lowest to highest entropy processing
    for line_num, bottom_k_words in all_bottom_k_words.items():
        # Tree
        node = root_tree
        for i, word in enumerate(bottom_k_words):
            if word in concatenated_tokens:
                start_index = concatenated_tokens.find(word)
                end_index = start_index + len(word)
                start_token_index = end_token_index = None
                current_length = 0
                for j, token in enumerate(tokens):
                    current_length += len(token)
                    if current_length > start_index and start_token_index is None:
                        start_token_index = j
                    if current_length >= end_index:
                        end_token_index = j
                        break
                if start_token_index is not None and end_token_index is not None:
                    if start_token_index < len(all_prob):
                        intermediate_results["relevant_log_probs_one_token"].append((i, all_prob[start_token_index]))
                    for idx in range(start_token_index, end_token_index + 1):
                        if idx < len(all_prob):
                            # Tree
                            if idx == start_token_index:
                                child = Tree.TreeNode(word, word, all_prob[idx])
                                child = node.add_child(child)
                                node = child

                            intermediate_results["relevant_log_probs"].append((i, all_prob[idx]))
                            if word not in intermediate_results["relevant_log_probs_words"]:
                                intermediate_results["relevant_log_probs_words"][word] = []
                            intermediate_results["relevant_log_probs_words"][word].append((i, all_prob[idx]))
                            # intermediate_results["relevant_log_probs_words"].append((word, all_prob[idx]))

    # Assuming intermediate_results["relevant_log_probs_words"] is your existing dictionary
    relevant_log_probs_words = intermediate_results["relevant_log_probs_words"]

    # Initialize the new dictionary to hold the first probabilities
    first_prob_dict = {}

    # Iterate through the existing dictionary
    for word, prob_list in relevant_log_probs_words.items():
        if prob_list:  # Check if the list is not empty
            # Take the first probability (which is a tuple, so we take the second element of the tuple)
            first_prob_dict[word] = prob_list[0][1]

    # Now first_prob_dict contains only the first probability for each word
    # plot_first_probabilities(first_prob_dict)
    # plot_first_probabilities_scatter(first_prob_dict)


    # Select the indices of words with the lowest entropy
    words_to_replace = [word for word, prob in first_prob_dict.items() if prob > threshold]

    # Replace words in the original text
    words = text.split()
    new_words = words.copy()
    for word in words_to_replace:
        if word in new_words:
            idx = new_words.index(word)
            original_word = new_words[idx]
            if original_word in replaced_dict:
                new_words[idx] = replaced_dict[original_word]
                continue
            synonyms = synonym_main.get_synonyms_by_different_methods(text, original_word, synonym_method, threshold=syn_threshold)
            max_entropy = calculate_entropy(original_word)
            best_word = original_word
            for synonym in synonyms:
                entropy = calculate_entropy(synonym)
                if entropy > max_entropy:
                    max_entropy = entropy
                    best_word = synonym
            new_words[idx] = best_word
            replaced_dict[original_word] = best_word
            root_tree.update_replaced(original_word, best_word)

    modified_text = ' '.join(new_words)
    # replaced_words, num_replaced_words = compare_texts(text, modified_text)
    # print(f"Replaced {num_replaced_words} words.")
    return modified_text, replaced_dict
