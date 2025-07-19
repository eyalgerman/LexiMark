import os
from datetime import datetime
import pandas as pd
import spacy
from typing import List, Dict, Optional
import watermark_detection.watermark_detection_2
from options import config_instance
from utils import file_processing, process_data, process_csv, QLora_finetune_LLM
from utils.create_map_entropy_both import create_entropy_map
from watermark_detection import dataset_detection_t_test
from watermarks import watermark_based_k
from watermarks.basic_watermark import bottom_k_entropy_words
from watermarks.watermark_based_k import replace_higher_top_k_entropy_with_higher_entropy
import pickle
import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import shutil
import tempfile

### Tag & Tab ###

def fine_tune_documents(model_name, folder_path, save_folder, watermark=False, k=4, mode="Text",synonym_method="context", syn_threshold=0.8, num_epochs=1, prefix=""):
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
        csv_file = os.path.join(folder_path, f"{prefix}watermarked_train_data_{timestamp}.csv")
    else:
        csv_file = os.path.join(folder_path, f"{prefix}train_data_{timestamp}.csv")

    process_csv.write_sentences_to_csv(sentences, csv_file)
    # Fine-tune the model
    return QLora_finetune_LLM.main(model_name, csv_file, save_folder, num_epochs=num_epochs)


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


def extract_scores(folder, model_name, output_folder, watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8):
    """
    Extracts scores from sentences in a folder using the Tag & Tab method.
    Args:
        folder (str): Path to the folder containing text files.
        model_name (str): Name or path of the model to use with Tag & Tab.
        output_folder (str): Directory to store detection results.
        watermark (bool): Whether to use watermarked text.
        k (int): The number of top-k elements used in the Tag & Tab feature extraction.
        mode (str): The mode of processing ('Text', 'JSON', etc.).
        synonym_method (str): Method used for synonym substitution.
        syn_threshold (float): Similarity threshold for accepting synonyms.
    Returns:
        np.ndarray: Array of scores for each sentence.

    """
    texts = file_processing.extract_texts_from_folder(folder).values()
    sentences = process_data.split_texts_into_sentences(texts)
    if watermark:
        print(f"Watermarking {len(sentences)} sentences with k={k}, mode={mode}, synonym_method={synonym_method}, syn_threshold={syn_threshold}")
        watermark_func = watermark_based_k.add_watermark_higher(k, mode, synonym_method, syn_threshold=syn_threshold)
        sentences = watermark_func(sentences)
    records = [{"input": s, "label": 0} for s in sentences]  # label is dummy here
    preds_df = watermark_detection.watermark_detection_2.main2(model_name, records, output_folder)
    scores = preds_df["sentence_entropy_log_likelihood_k=4"].values
    mask = ~np.isnan(scores)
    X = np.array(scores)[mask].reshape(-1, 1)
    return X


def classifier_builder(nonmember_dir, model_name, save_path, nu=0.15, output_folder="tagtab_out", watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8):
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

    # --- Feature Extraction ---
    X = extract_scores(nonmember_dir, model_name, output_folder, watermark=watermark, k=k, mode=mode, synonym_method=synonym_method, syn_threshold=syn_threshold)
    if len(X) == 0:
        raise ValueError("No valid scores to train on (all are NaN). Check your data pipeline.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train Classifier ---
    clf = OneClassSVM(kernel="rbf", nu=nu)
    clf.fit(X_scaled)

    # --- Save Scaler and Classifier Together ---
    with open(save_path, "wb") as f:
        pickle.dump((scaler, clf), f)

    print(f"OOD detector trained and saved to {save_path}.")
    print(f"Mean score: {np.mean(X):.4f} | std: {np.std(X):.4f}")


def extract_features_from_folder(folder, model_name, output_folder, scaler, label=0, watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8):
    """
    For evaluation: extracts features and applies the provided scaler (trained on training data).
    Args:
        folder (str): Path to the folder containing text files.
        model_name (str): Name or path of the model used in Tag & Tab.
        output_folder (str): Directory to store detection results.
        scaler (StandardScaler): Pre-trained scaler to apply to the extracted features.
        label (int): Label for the extracted features (1 for member, 0 for non-member).
        watermark (bool): Whether to use watermarked text.
        k (int): The number of top-k elements used in the Tag & Tab feature extraction.
        mode (str): The mode of processing ('Text', 'JSON', etc.).
        synonym_method (str): Method used for synonym substitution.
        syn_threshold (float): Similarity threshold for accepting synonyms.
    Returns:
        X_scaled (np.ndarray): Scaled feature array.
        y (np.ndarray): Array of labels corresponding to the features.
    """
    X = extract_scores(folder, model_name, output_folder, watermark=watermark, k=k, mode=mode, synonym_method=synonym_method, syn_threshold=syn_threshold)
    X_scaled = scaler.transform(X)
    y = np.full(X.shape[0], label)
    return X_scaled, y


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


def membership_inference_dataset_tag_and_tab(doc_dir, model_name, nonmember_dir, output_folder="tagtab_out",
                                             watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8, significance_level=0.01):
    """
    Perform dataset-level membership inference using Tag-and-Tab features and a T-test.

    This function extracts features from member and non-member document directories,
    performs a T-test to compare their distributions, and returns a decision indicating
    whether the dataset appears to be a member of the model's training data.

    Args:
        doc_dir (str): Path to the folder containing member documents.
        model_name (str): Name or path of the model used to extract features.
        nonmember_dir (str): Path to the folder containing non-member documents.
        output_folder (str, optional): Directory to store intermediate outputs. Defaults to "tagtab_out".
        watermark (bool, optional): Whether to use watermarked text when extracting features. Defaults to False.
        k (int, optional): The number of top-k elements used in the Tag-and-Tab feature extraction. Defaults to 4.
        mode (str, optional): The mode of processing ('Text', 'JSON', etc.). Defaults to "Text".
        synonym_method (str, optional): Method used for synonym substitution. Defaults to "context".
        syn_threshold (float, optional): Similarity threshold for accepting synonyms. Defaults to 0.8.

    Returns:
        member (bool): Result of the T-test; True if the dataset is inferred to be a member, False otherwise.
    """
    os.makedirs(output_folder, exist_ok=True)
    scaler = StandardScaler()

    # Extract features
    X_member = extract_scores(
        doc_dir, model_name, output_folder,
        watermark=watermark, k=k, mode=mode,
        synonym_method=synonym_method, syn_threshold=syn_threshold
    )
    X_non_member = extract_scores(
        nonmember_dir, model_name, output_folder,
        watermark=watermark, k=k, mode=mode,
        synonym_method=synonym_method, syn_threshold=syn_threshold
    )
    t_stat, p_value, member = dataset_detection_t_test.perform_t_test_scores(pd.Series(X_member.ravel()), pd.Series(X_non_member.ravel()), significance_level)
    return member


def evaluate_membership_inference_dataset(members_dir, nonmembers_dir, model_name, ture_nonmembers_dir,
                                                      watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8, significance_level=0.01):
    """
    Evaluate membership_inference_dataset_tag_and_tab on each file or folder under members_dir and nonmembers_dir.

    Args:
        members_dir (str): Path to folder containing member files or folders.
        nonmembers_dir (str): Path to folder containing non-member files or folders.
        model_name (str): Name or path of model.
        watermark (bool): Whether to use watermarked text.
        k (int): Top-k for feature extraction.
        mode (str): Processing mode.
        synonym_method (str): Synonym substitution method.
        syn_threshold (float): Similarity threshold for synonyms.

    Returns:
        accuracy, precision, recall (float)
    """
    y_true = []
    y_pred = []

    member_items = sorted(os.listdir(members_dir))
    nonmember_items = sorted(os.listdir(nonmembers_dir))

    # Counters
    member_correct = 0
    nonmember_correct = 0

    print(f"\nProcessing {len(member_items)} members...")
    for item in member_items:
        member_path = os.path.join(members_dir, item)
        try:
            member_result = membership_inference_dataset_tag_and_tab(
                member_path, model_name, ture_nonmembers_dir,
                watermark=watermark, k=k, mode=mode,
                synonym_method=synonym_method, syn_threshold=syn_threshold, significance_level=significance_level
            )
        except:
            print(f"Error processing member {item}. Skipping...")
            continue

        y_true.append(1)
        y_pred.append(int(member_result))

        if member_result:
            member_correct += 1

        print(f"  Member: {item} → Predicted: {'MEMBER' if member_result else 'NON-MEMBER'}")

    print(f"\nMember prediction summary: {member_correct} correct out of {len(member_items)} ({100.0 * member_correct / len(member_items):.2f}%)\n")

    print(f"\nProcessing {len(nonmember_items)} non-members...")
    for item in nonmember_items:
        nonmember_path = os.path.join(nonmembers_dir, item)

        non_member_result = membership_inference_dataset_tag_and_tab(
            nonmember_path, model_name, ture_nonmembers_dir,
            watermark=watermark, k=k, mode=mode,
            synonym_method=synonym_method, syn_threshold=syn_threshold, significance_level=significance_level
        )

        y_true.append(0)
        y_pred.append(int(non_member_result))

        if not non_member_result:
            nonmember_correct += 1

        print(f"  Non-member: {item} → Predicted: {'MEMBER' if non_member_result else 'NON-MEMBER'}")

    print(f"\nNon-member prediction summary: {nonmember_correct} correct out of {len(nonmember_items)} ({100.0 * nonmember_correct / len(nonmember_items):.2f}%)\n")

    # Compute overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"\nFinal Evaluation Results, with significance_level of {significance_level}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Correctly classified members: {member_correct} out of {len(member_items)}")
    print(f"  Correctly classified non-members: {nonmember_correct} out of {len(nonmember_items)}")

    return accuracy, precision, recall


def evaluate_classifier(classifier_path, member_folder, non_member_folder, model_name, output_folder,
                        watermark=False, k=4, mode="Text", synonym_method="context", syn_threshold=0.8):
    """
    Loads the trained classifier and scaler, extracts features for test member and non-member folders,
    predicts labels, and prints accuracy, precision, and recall.

    Args:
        classifier_path (str): Path to saved (scaler, classifier) pickle file.
        member_folder (str): Path to test member docs.
        non_member_folder (str): Path to test non-member docs.
        model_name, output_folder, watermark, k, mode, synonym_method, syn_threshold: as above.

    Returns:
        (accuracy, precision, recall)
    """
    # Load scaler and classifier
    with open(classifier_path, "rb") as f:
        scaler, clf = pickle.load(f)

    # Extract features
    X_member, y_member = extract_features_from_folder(
        member_folder, model_name, output_folder, scaler,
        label=1, watermark=watermark, k=k, mode=mode,
        synonym_method=synonym_method, syn_threshold=syn_threshold
    )
    X_non_member, y_non_member = extract_features_from_folder(
        non_member_folder, model_name, output_folder, scaler,
        label=0, watermark=watermark, k=k, mode=mode,
        synonym_method=synonym_method, syn_threshold=syn_threshold
    )

    X_test = np.concatenate([X_member, X_non_member], axis=0)
    y_true = np.concatenate([y_member, y_non_member], axis=0)

    # Predict: OneClassSVM gives +1 for inlier, -1 for outlier. Treat +1 as "member", -1 as "non-member".
    y_pred = clf.predict(X_test)
    y_pred_bin = (y_pred == 1).astype(int)

    accuracy = accuracy_score(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)

    print(f"Evaluation Results for {classifier_path}:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

    return accuracy, precision, recall


### LexiMark ###

def watermark_folder(input_folder, output_folder, k=4, mode="Text", synonym_method="context", syn_threshold=0.8):
    """
    Watermarks all text files in a folder and saves the watermarked versions in a new folder.

    Args:
        input_folder (str): Path to the input folder containing text files.
        output_folder (str): Path to the output folder to save watermarked files.
        k (int): The number of words to watermark.
        mode (str): The watermark mode.
        synonym_method (str): The synonym substitution method.
        syn_threshold (float): Synonym similarity threshold.
    """
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Load watermarking function
    watermark_func = watermark_based_k.add_watermark_higher(k, mode, synonym_method, syn_threshold=syn_threshold)

    # Extract all texts from the folder
    texts_dict = file_processing.extract_texts_from_folder(input_folder)

    # Process each file
    for filename, text in texts_dict.items():
        # Split text into sentences
        sentences = process_data.split_texts_into_sentences([text])

        # Apply watermarking
        watermarked_sentences = watermark_func(sentences)

        # Reconstruct full text
        watermarked_text = " ".join(watermarked_sentences)

        # Save to output folder with same filename
        base_name, _ = os.path.splitext(filename)
        output_file_path = os.path.join(output_folder, base_name + ".txt")
        # output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(watermarked_text)

    print(f"Watermarked files saved to: {output_folder}")


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
    # output_folder = "/".join(output_folder.split("/")[:-2]) + "/LexiMark/"
    model_name = "mistralai/Mistral-7B-v0.1"
    save_finetuned_folder = output_folder + "models/"
    # Paths for test data
    member_test_folder = output_folder + "data/member_new/"
    non_member_test_folder = output_folder + "data/non-member_new/"
    non_member_classifier_folder = output_folder + "data/non-member-classifier_new/"
    watermark_member_test_folder = output_folder + "data/watermarked-member_new/"
    watermark_non_member_test_folder = output_folder + "data/watermarked-non-member_new/"
    watermark_non_member_classifier_folder = output_folder + "data/watermarked-non-member-classifier_new/"

    sample_text = "The quick brown fox jumps over the lazy dog. Neural networks are fascinating."

    print("\n=== Watermarking files ===")
    watermark_folder(member_test_folder, watermark_member_test_folder, k=4, mode="Text", synonym_method="context", syn_threshold=0.8)
    watermark_folder(non_member_test_folder, watermark_non_member_test_folder, k=4, mode="Text", synonym_method="context", syn_threshold=0.8)
    watermark_folder(non_member_classifier_folder, watermark_non_member_classifier_folder, k=4, mode="Text",
                     synonym_method="context", syn_threshold=0.8)

    print("=== Fine-Tuning Documents ===")
    num_epochs = 2
    finetuned_model_path = fine_tune_documents(model_name, member_test_folder, save_finetuned_folder, num_epochs=num_epochs)
    # watermarked_finetuned_model_path = "/dt/shabtaia/dt-sicpa/eyal/LexiMark/models/Mistral-7B-v0.1_watermarked_train_data_20250520-152549_QLORA_2025_05_20_15_29_40_epochs_1_Merged/"
    watermarked_finetuned_model_path = fine_tune_documents(model_name, watermark_member_test_folder, save_finetuned_folder, num_epochs=num_epochs, prefix="watermarked_")
    # finetuned_model_path = "/dt/shabtaia/dt-sicpa/eyal/LexiMark/models/Mistral-7B-v0.1_train_data_20250520-154509_QLORA_2025_05_20_15_45_15_epochs_1_Merged/"
    print(f"Fine-tuned model saved at: {finetuned_model_path}")

    # print("\n=== Running Tag & Tab ===")
    # mean_entropy = run_tag_and_tab(finetuned_model_path, output_folder + "data/member/", output_folder)
    # print(f"Mean sentence entropy for k=4: {mean_entropy}")
    #
    # print("\n=== Classifier Builder ===")
    # nonmember_train_folder = output_folder + "data/non-member-classifier/"
    # nu = 0.3
    # classifier_path = output_folder + f"data/tagtab_classifier_nu_{nu}.pkl"
    # classifier_builder(nonmember_train_folder, finetuned_model_path, classifier_path, output_folder=output_folder,
    #                    nu=nu, watermark=False)
    # watermark_classifier_path = output_folder + f"data/watermark_tagtab_classifier_nu_{nu}.pkl"
    # classifier_builder(nonmember_train_folder, watermarked_finetuned_model_path, watermark_classifier_path,
    #                    output_folder=output_folder, nu=nu, watermark=True)
    # print(f"Classifiers saved at: {classifier_path} and {watermark_classifier_path}")
    #
    #
    # print("\n=== Evaluating Classifier (No Watermark) ===")
    # evaluate_classifier(
    #     classifier_path,
    #     member_test_folder,
    #     non_member_test_folder,
    #     finetuned_model_path,
    #     output_folder,
    #     watermark=False
    # )
    #
    # print("\n=== Evaluating Classifier (Watermark) ===")
    # evaluate_classifier(
    #     watermark_classifier_path,
    #     member_test_folder,
    #     non_member_test_folder,
    #     watermarked_finetuned_model_path,
    #     output_folder,
    #     watermark=True
    # )
    #
    # print("\n=== Membership Inference Example ===")
    # # Example for one document; can be looped for all docs if needed
    # doc_path = output_folder + "data/non-member/Banking Automation Bulletin April 2022 - Chile profile.pdf"
    # prediction = membership_inference_tag_and_tab(member_test_folder, finetuned_model_path, classifier_path,
    #                                               output_folder=output_folder)
    # print(f"Member test folder classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_tag_and_tab(doc_path, finetuned_model_path, classifier_path,
    #                                               output_folder=output_folder)
    # print(f"Non-member file classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_tag_and_tab(member_test_folder, watermarked_finetuned_model_path,
    #                                               watermark_classifier_path, output_folder=output_folder)
    # print(f"Member test folder (watermarked) classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_tag_and_tab(doc_path, watermarked_finetuned_model_path,
    #                                               watermark_classifier_path, output_folder=output_folder)
    # print(f"Non-member file (watermarked) classified as: {'member' if prediction else 'non-member'}.")

    print("\n=== Membership Inference Dataset Evaluation ===")
    significance_level = 0.01
    evaluate_membership_inference_dataset(member_test_folder, non_member_test_folder, finetuned_model_path, non_member_classifier_folder, significance_level=significance_level)
    print("\n=== Membership Inference Dataset Evaluation (Watermarked) ===")
    evaluate_membership_inference_dataset(watermark_member_test_folder, watermark_non_member_test_folder, watermarked_finetuned_model_path, watermark_non_member_classifier_folder, significance_level=significance_level)

    # print("\n=== Membership Inference Example Dataset detection ===")
    # # Example for one document; can be looped for all docs if needed
    # doc_path = output_folder + "data/non-member/Banking Automation Bulletin April 2022 - Chile profile.pdf"
    # prediction = membership_inference_dataset_tag_and_tab(member_test_folder, finetuned_model_path, non_member_classifier_folder,
    #                                               output_folder=output_folder)
    # print(f"Member test folder classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_dataset_tag_and_tab(doc_path, finetuned_model_path, non_member_classifier_folder,
    #                                               output_folder=output_folder)
    # print(f"Non-member file classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_dataset_tag_and_tab(watermark_member_test_folder, watermarked_finetuned_model_path,
    #                                               non_member_classifier_folder, output_folder=output_folder)
    # print(f"Member test folder (watermarked) classified as: {'member' if prediction else 'non-member'}.")
    #
    # prediction = membership_inference_dataset_tag_and_tab(doc_path, watermarked_finetuned_model_path,
    #                                               non_member_classifier_folder, output_folder=output_folder)
    # print(f"Non-member file (watermarked) classified as: {'member' if prediction else 'non-member'}.")


    print("\n=== LexiMark: Identify High Entropy Words ===")
    top_words = identify_high_entropy_words(sample_text, top_k=5)
    print(f"Identified high-entropy words: {top_words}")

    print("\n=== LexiMark: Recommend Synonyms ===")
    replacements = recommend_synonyms(sample_text, synonym_method="context", syn_threshold=0.6, k=3)
    print(f"Recommended replacements: {replacements}")

    print("\n=== LexiMark: Embed Synonyms in Text ===")
    modified_text = embed_synonyms_in_text(sample_text, replacements)
    print(f"Modified text: {modified_text}")

