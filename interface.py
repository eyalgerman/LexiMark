import os
from datetime import datetime

import spacy
from typing import List, Dict, Optional

import watermark_detection.watermark_detection_2
from utils import file_processing, process_data, process_csv, QLora_Medium_Finetune_LLM
from utils.create_map_entropy_both import create_entropy_map
from watermarks.basic_watermark import bottom_k_entropy_words
from watermarks.watermark_based_k import replace_higher_top_k_entropy_with_higher_entropy


### Tag & Tab ###

def fine_tune_documents(model_name, folder_path, save_folder):
    texts = file_processing.extract_texts_from_folder(folder_path)
    sentences = process_data.split_texts_into_sentences(texts)
    # Write sentences to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = os.path.join(folder_path, f"train_data_{timestamp}.csv")
    process_csv.write_sentences_to_csv(sentences, csv_file)
    # Fine-tune the model
    return QLora_Medium_Finetune_LLM.main(model_name, csv_file, save_folder)



def run_tag_and_tab(model_name, folder_path, output_folder):
    """
    Run the Tag & Tab watermark detection method.
    Args:
        model_name:
        folder_path:
        output_folder:

    Returns:
    """
    metrics_df = watermark_detection.watermark_detection_2.main2(model_name, folder_path, output_folder)
    return metrics_df



### LexiMark ###

def identify_high_entropy_words(text: str, top_k: int = 5):
    """
    Identifies the top-k high-entropy words in the given text.

    Args:
        text (str): The input text to analyze.
        top_k (int): Number of high-entropy words to return.

    Returns:
        List[str]: A list of high-entropy words from the text.
    """
    nlp_spacy = spacy.load("en_core_web_sm")
    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    entropy_map = create_entropy_map(text, mode="Text") # TODO: add Text mode
    sentences = [sent.text for sent in doc.sents]
    all_bottom_k_words = {}
    for sentence in sentences:
        # highest entropy words
        bottom_k_words = bottom_k_entropy_words(sentence, entropy_map, top_k)
        all_bottom_k_words[sentence] = bottom_k_words
    # Flatten the list of words and get unique words
    unique_words = set()
    for words in all_bottom_k_words.values():
        unique_words.update(words)
    return unique_words


def recommend_synonyms(text, synonym_method, syn_threshold, k) -> List[str]:
    entropy_map = create_entropy_map(text, mode="Text")
    replaced_dict = {}
    new_text, replaced_dict = replace_higher_top_k_entropy_with_higher_entropy(text, replaced_dict=replaced_dict,
                                                                               entropy_map=entropy_map, k_value=k,
                                                                               synonym_method=synonym_method,
                                                                               syn_threshold=syn_threshold)
    return replaced_dict


def embed_synonyms_in_text(text: str, replacements: Dict[str, str]) -> str:
    for target_word, synonym in replacements.items():
        text = text.replace(target_word, synonym)
    return text
