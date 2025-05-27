import os
from datetime import datetime
import spacy
from typing import List, Dict, Optional
import watermark_detection.watermark_detection_2
from options import config_instance
from utils import file_processing, process_data, process_csv, QLora_finetune_LLM
from utils.create_map_entropy_both import create_entropy_map
from watermarks import watermark_based_k
from watermarks.basic_watermark import bottom_k_entropy_words
from watermarks.watermark_based_k import replace_higher_top_k_entropy_with_higher_entropy
import pickle
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import shutil
import tempfile

### Tag & Tab ###

def fine_tune_documents(model_name, folder_path, save_folder, watermark=False, k=4, mode="Text",synonym_method="context", syn_threshold=0.8):
    """
    Fine-tunes a model on text extracted from a folder using QLoRA.

    Args:
        model_name (str): Path or name of the base model to fine-tune.
        folder_path (str): Path to the folder containing text files.
        save_folder (str): Directory where the fine-tuned model will be saved.

    Returns:
        str: Path to the fine-tuned model directory.
    """
    texts = file_processing.extract_texts_from_folder(folder_path).values()
    sentences = process_data.split_texts_into_sentences(texts)
    if watermark:
        # Add watermark to sentences
        watermark = watermark_based_k.add_watermark_higher(k, mode, synonym_method, syn_threshold=syn_threshold)
        sentences = watermark(sentences)
    # Write sentences to CSV
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if watermark:
        csv_file = os.path.join(folder_path, f"watermarked_train_data_{timestamp}.csv")
    else:
        csv_file = os.path.join(folder_path, f"train_data_{timestamp}.csv")
    process_csv.write_sentences_to_csv(sentences, csv_file)
    # Fine-tune the model
    return QLora_finetune_LLM.main(model_name, csv_file, save_folder)


def run_tag_and_tab(model_name, folder_path, output_folder):
    """
    Executes the Tag & Tab watermark detection method and computes average entropy score.

    Args:
        model_name (str): Path or name of the target model.
        folder_path (str): Folder containing input files for evaluation.
        output_folder (str): Directory to store detection results.

    Returns:
        float: Mean sentence entropy score for `k=4`.
    """
    texts = file_processing.extract_texts_from_folder(folder_path).values()
    sentences = process_data.split_texts_into_sentences(texts)
    records = []
    for sentence in sentences:
        records.append({"input": sentence, "label": 0})
    preds_df = watermark_detection.watermark_detection_2.main2(model_name, records, output_folder)
    return preds_df["sentence_entropy_log_likelihood_k=4"].mean()


def classifier_builder(nonmember_dir, model_name, save_path, nu=0.05, output_folder="tagtab_out", watermark=False, k=4, mode="Text",synonym_method="context", syn_threshold=0.8):
    """
    Builds a membership inference classifier using One-Class SVM trained on non-member documents.

    Args:
        nonmember_dir (str): Path to folder containing .txt files of non-member documents.
        model_name (str): Name or path of the model to use with Tag & Tab.
        save_path (str): Path to save the trained classifier.
        nu (float): Estimated fraction of training errors (lower = stricter boundary).
        output_folder (str): Directory to store detection results.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    texts = file_processing.extract_texts_from_folder(nonmember_dir).values()
    sentences = process_data.split_texts_into_sentences(texts)
    if watermark:
        # Add watermark to sentences
        watermark = watermark_based_k.add_watermark_higher(k, mode, synonym_method, syn_threshold=syn_threshold)
        sentences = watermark(sentences)
    records = []
    for sentence in sentences:
        records.append({"input": sentence, "label": 0})
    preds_df = watermark_detection.watermark_detection_2.main2(model_name, records, output_folder)
    scores =  preds_df["sentence_entropy_log_likelihood_k=4"].values
    # Remove NaNs
    mask = ~np.isnan(scores)
    X = np.array(scores)[mask].reshape(-1, 1)
    # X = np.array(scores).reshape(-1, 1)
    if len(X) == 0:
        raise ValueError("No valid scores to train on (all are NaN). Check your data pipeline.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = OneClassSVM(kernel="rbf", nu=nu)
    clf.fit(X_scaled)

    with open(save_path, "wb") as f:
        pickle.dump((scaler, clf), f)

    print(f"OOD detector trained and saved to {save_path}.")
    print(f"Mean score: {np.mean(scores):.4f} | std: {np.std(scores):.4f}")


def membership_inference_tag_and_tab(doc_dir, model_name, model_path, output_folder="tagtab_out"):
    """
    Predicts whether a document is in-distribution (i.e., member) using the trained OOD classifier.

    Args:
        doc_dir (str): Path to a folder containing a single .txt file to test.
        model_name (str): Name or path of the model used in Tag & Tab.
        model_path (str): Path to the saved classifier.
        output_folder (str): Directory to store detection results.

    Returns:
        bool: True if predicted as a member (in-distribution), False otherwise.
    """
    os.makedirs(output_folder, exist_ok=True)

    score = run_tag_and_tab(model_name, doc_dir, output_folder)

    with open(model_path, "rb") as f:
        scaler, clf = pickle.load(f)

    X = np.array([[score]])
    X_scaled = scaler.transform(X)

    prediction = clf.predict(X_scaled)[0]  # +1 = inlier (member), -1 = outlier (non-member)
    return prediction == 1 # Member == True | Non-Memebr == False


### LexiMark ###

def identify_high_entropy_words(text: str, top_k: int = 5):
    """
    Identifies the top-k high-entropy words in the text based on entropy scores.

    Args:
        text (str): The input text to analyze.
        top_k (int): Number of high-entropy words to return per sentence.

    Returns:
        List[str]: Unique high-entropy words across all sentences.
    """
    nlp_spacy = spacy.load("en_core_web_sm")
    # Create a mapping of lines to their top words based on entropy
    doc = nlp_spacy(text)
    entropy_map = create_entropy_map(text, mode="Text")
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
    """
    Replaces top-k high-entropy words with higher-entropy synonyms.

    Args:
        text (str): Input text to modify.
        synonym_method (str): Method to generate synonyms (e.g., 'context', 'sbert').
        syn_threshold (float): Threshold for synonym contextual similarity.
        k (int): Number of high-entropy words to replace.

    Returns:
        Dict[str, str]: Dictionary of original words and their replacements.
    """
    entropy_map = create_entropy_map(text, mode="Text")
    replaced_dict = {}
    new_text, replaced_dict = replace_higher_top_k_entropy_with_higher_entropy(text, replaced_dict=replaced_dict,
                                                                               entropy_map=entropy_map, k_value=k,
                                                                               synonym_method=synonym_method,
                                                                               syn_threshold=syn_threshold)
    return replaced_dict


def embed_synonyms_in_text(text: str, replacements: Dict[str, str]) -> str:
    """
    Replaces specified words in the text with their synonyms.

    Args:
        text (str): The input text.
        replacements (Dict[str, str]): Mapping from original words to synonyms.

    Returns:
        str: Modified text with embedded synonyms.
    """
    for target_word, synonym in replacements.items():
        text = text.replace(target_word, synonym)
    return text


if __name__ == "__main__":
    # Parameters
    output_folder = config_instance.data_dir
    model_name = "mistralai/Mistral-7B-v0.1"
    # input_folder = "data"
    input_folder = output_folder + "data/member/"
    # output_folder = output_folder + "results"
    # output_folder = "results"
    # save_finetuned_folder = "models/"
    save_finetuned_folder = output_folder + "models/"
    sample_text = "The quick brown fox jumps over the lazy dog. Neural networks are fascinating."

    print("=== Fine-Tuning Documents ===")
    watermarked_finetuned_model_path = fine_tune_documents(model_name, input_folder, save_finetuned_folder, watermark=True)
    finetuned_model_path = fine_tune_documents(model_name, input_folder, save_finetuned_folder, watermark=False)
    print(f"Fine-tuned model saved at: {finetuned_model_path}")

    print("\n=== Running Tag & Tab ===")
    mean_entropy = run_tag_and_tab(finetuned_model_path, input_folder, output_folder)
    print(f"Mean sentence entropy for k=4: {mean_entropy}")

    print("\n=== Classifier Builder ===")
    classifier_path = output_folder + "data/tagtab_classifier.pkl"
    input_folder = output_folder + "data/non-member-classifier/"
    classifier_builder(input_folder, finetuned_model_path, classifier_path, output_folder=output_folder)
    watermark_classifier_path = output_folder + "data/watermark_tagtab_classifier.pkl"
    classifier_builder(input_folder, watermarked_finetuned_model_path, watermark_classifier_path, output_folder=output_folder,watermark=True)

    print(f"Classifier saved at: {classifier_path}")

    print("\n=== Membership Inference ===")
    test_member_folder = output_folder + "data/member/"
    test_non_member_folder = output_folder + "data/non-member/"
    prediction = membership_inference_tag_and_tab(test_member_folder, finetuned_model_path, classifier_path, output_folder=output_folder)
    print(f"We entered the member folder:")
    print(f"Document is {'member' if prediction else 'non-member'}.")

    prediction = membership_inference_tag_and_tab(test_non_member_folder, finetuned_model_path, classifier_path, output_folder=output_folder)
    print(f"We entered the non-member folder:")
    print(f"Document is {'member' if prediction else 'non-member'}.")


    print("\n=== LexiMark: Identify High Entropy Words ===")
    top_words = identify_high_entropy_words(sample_text, top_k=5)
    print(f"Identified high-entropy words: {top_words}")

    print("\n=== LexiMark: Recommend Synonyms ===")
    replacements = recommend_synonyms(sample_text, synonym_method="context", syn_threshold=0.6, k=3)
    print(f"Recommended replacements: {replacements}")

    print("\n=== LexiMark: Embed Synonyms in Text ===")
    modified_text = embed_synonyms_in_text(sample_text, replacements)
    print(f"Modified text: {modified_text}")


