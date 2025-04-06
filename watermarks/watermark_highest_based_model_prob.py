import spacy
from tqdm import tqdm
from synonyms_methods import synonym_main
from utils.create_map_entropy_both import create_entropy_map
from watermark_detection.watermark_detection_2 import calculate_perplexity, load_model
from .basic_watermark import bottom_k_entropy_words
import torch
from torch.nn.functional import log_softmax
from .watermark_tree import MODEL_NAME


def get_log_probability_of_word_in_context(text, word, model, tokenizer):
    # Encode text
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Compute log-probabilities
    log_probs = log_softmax(logits, dim=-1)

    # Tokenize the text to compare with the word
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_tokens = tokenizer.tokenize(word)



    # Convert word tokens to IDs
    word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

    # Find the starting index of the word in input_ids
    input_ids_list = input_ids[0].tolist()

    found = False
    for i in range(len(input_ids_list) - len(word_token_ids) + 1):
        # Compare the tokens in the input sequence with word tokens
        if input_ids_list[i:i + len(word_token_ids)] == word_token_ids:
            word_position = i
            found = True
            break

    if not found:
        print("Word not found in context:", word)
        print("Text:", text)
        # Debug: Print the tokenized input and word
        print(f"Tokenized word: {word_tokens}")
        print(f"Tokenized text: {tokenized_text}")
        return None

    # Sum the log-probabilities of the word tokens
    log_prob = 0
    for i, token_id in enumerate(word_token_ids):
        log_prob += log_probs[0, word_position + i, token_id].item()

    return log_prob


def replace_highest_entropy_words_with_highest_model_prob(text, k_value, synonym_method, replaced_dict={}, entropy_map={}, model=None, tokenizer=None, syn_threshold=0.6):
    nlp_spacy = spacy.load("en_core_web_sm")
    p1, all_prob, p1_likelihood, logits, input_ids_processed = calculate_perplexity(text, model, tokenizer, gpu=model.device)

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
                # synonyms = get_synonyms(original_word)
                synonyms = synonym_main.get_synonyms_by_different_methods(sentence, original_word, synonym_method, syn_threshold=syn_threshold)

                max_log_prob = get_log_probability_of_word_in_context(text, original_word, model, tokenizer)
                best_word = original_word
                if max_log_prob is None:
                    continue
                for synonym in synonyms:
                    # Replace the word with the synonym in the text
                    modified_text = text.replace(original_word, synonym, 1)
                    # Calculate log-probability of the synonym in context
                    log_prob = get_log_probability_of_word_in_context(
                        modified_text, synonym, model, tokenizer
                    )
                    if log_prob is not None and log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_word = synonym
                new_words[idx] = best_word
                replaced_dict[original_word] = best_word

    modified_text = ' '.join(new_words)
    return modified_text, replaced_dict

def add_watermark_highest_prob(k=5, mode="BookMIA", synonym_method="context", syn_threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words
        model1, tokenizer1 = load_model(MODEL_NAME)

        print("Creating entropy map...")
        entropy_map = create_entropy_map(data, mode=mode)
        for text in tqdm(data):
            new_text, replaced_dict = replace_highest_entropy_words_with_highest_model_prob(text, replaced_dict=replaced_dict, entropy_map=entropy_map, k_value=k,
                            synonym_method=synonym_method, model=model1, tokenizer=tokenizer1, syn_threshold=syn_threshold)
            new_sentences.append(new_text)
        return new_sentences
    return watermarked_sentences