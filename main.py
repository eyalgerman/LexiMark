import glob
import json
import os
import re
from datetime import datetime

from transformers import AutoTokenizer

from utils.instruction_tuning import instruction_tune
from utils.pretrain_LLM import pretrain_model
from watermarks import watermark_backdoor, deduplication_filter
from watermarks import watermark_based_k
from watermarks import watermark_based_percentage
from watermarks import watermark_based_prompt
from watermarks import basic_watermark
from watermarks import watermark_highest_based_model_prob
from watermarks import watermark_tree
from watermarks import robustness_watermark
from utils import QLora_finetune_LLM, openAI_finetune_GPT, pretrain_LLM, instruction_tuning
from watermarks.baselines import TextMarker_watrmark
from watermarks.baselines import random_seq_watermark
from watermarks.baselines import unicode_lookalike_watermark
from watermark_detection import watermark_detection_2
from options import Options, Config
from utils import process_data
from watermarks.paraphraser import paraphrase_text



def load_and_split_data(config, args, method_name, watermark=None, params={}):
    """
    Loads the dataset and optionally applies a watermark before splitting into member and non-member sets.

    If an existing processed file is found and allowed by arguments, it will be reused instead of regenerating data.
    Otherwise, the function generates and saves new datasets with or without watermarking.

    Args:
        config (Config): Configuration object containing paths and settings.
        args (argparse.Namespace): Parsed command-line arguments with dataset and model settings.
        method_name (str): Name of the watermarking method to apply.
        watermark (Optional[Callable], optional): Watermarking function or object to apply. Defaults to None.
        params (dict, optional): Parameters for the watermark method, used to build filenames. Defaults to {}.

    Returns:
        Tuple[str, str, str]: Paths to the member file, non-member file, and the original combined file.
    """
    filename = config.build_filename(args, method_name, params)
    # Check if the file exists
    jsonl_file = filename.replace(".csv", ".jsonl")
    print(f"Creating dataset: {jsonl_file}")
    use_existing = True if args.use_existing.lower() in ['all', 'data'] else False
    if os.path.exists(jsonl_file) and use_existing:
        print(f"File {jsonl_file} already exists. Skipping data generation.")
        output_file_member = filename.replace(".csv", "_member.csv")
        output_file_non_member = filename.replace(".csv", "_non_member.csv")
        if os.path.exists(output_file_member) and os.path.exists(output_file_non_member):
            return output_file_member, output_file_non_member, filename
        else:
            print(f"Missing one or more of the files: {output_file_member}, {output_file_non_member}")
    output_file_member, output_file_non_member = process_data.load_clean_data_and_split(
        mode=args.mode, from_idx=0, count=config.count, key_name=args.key_name, split=args.split,
        output_file=filename, watermark=watermark, filter=config.filter, watermark_non_member=args.watermark_non_member
    )
    return output_file_member, output_file_non_member, filename


def init_watermark(args, method):
    """
    Initializes the appropriate watermarking function or object based on the selected method.

    Supports both synonym-based and non-synonym-based watermarking techniques, including
    backdoors, paraphrasing, randomization, and tree-based strategies.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        method (str): Name of the watermark method to initialize.

    Returns:
        Tuple[Callable, dict]: A tuple containing the watermark function/object and a
        dictionary of parameters used for logging or file naming.
    """
    synonym_threshold_methods = ["context", "sbert"]
    non_synonym_methods = ["random_seq", "unicode_lookalike_global", "unicode_lookalike_word", "text_marker", "None", "highest_based_prompt"]
    watermark = None
    params = {}
    if method == "basic":
        p = args.k
        watermark = basic_watermark.add_watermark(p, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'p': p}
    elif method == "random":
        p = args.k
        watermark = basic_watermark.add_watermark_random(p, args.synonym_method, seed=args.seed, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'p': p, 'seed': args.seed}
    elif method == "top-k-lowest":
        watermark = watermark_based_k.add_watermark_lower(args.k, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k}
    elif method == "top-k-higher" or method == "ours" or method.lower() == "leximark": # leximark
        method = "top-k-higher"
        args.method = method
        watermark = watermark_based_k.add_watermark_higher(args.k, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k}
    elif method == "top-k-random":
        watermark = watermark_based_k.add_watermark_random(args.k, args.synonym_method, seed=args.seed, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k, 'seed': args.seed}
    # elif method == "dict":
    #     params = {'syn': args.synonym_method}
    #     filename = config.build_filename(args, args.method, params)
    #     watermark = watermark_with_dict.add_watermark(synonym_method=args.synonym_method, output_file=filename, syn_threshold=args.context_th)
    elif method == "tree":
        params = {'syn': args.synonym_method, 'k': args.k, 'th': args.threshold}
        filename = config.build_filename(args, args.method, params)
        watermark = watermark_tree.add_watermark(k=args.k, threshold=args.threshold, synonym_method=args.synonym_method, syn_threshold=args.context_th,
                                                 output_file=filename)
    elif method == "highest-p-percentage":
        watermark = watermark_based_percentage.add_watermark_higher(args.p, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'p': args.p}
    elif method == "lowest-p-percentage":
        watermark = watermark_based_percentage.add_watermark_lower(args.p, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'p': args.p}
    elif method == "random-p-percentage":
        watermark = watermark_based_percentage.add_watermark_random(args.p, args.synonym_method, seed=args.seed, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'p': args.p, 'seed': args.seed}
    elif method == "random_seq":
        watermark = random_seq_watermark.add_watermark(noise_length=10)
        params = {'noise_len': 10}
    elif method == "unicode_lookalike_global":
        watermark = unicode_lookalike_watermark.global_unicode_watermark(seed=42)
    elif method == "unicode_lookalike_word":
        watermark = unicode_lookalike_watermark.word_level_unicode_watermark(seed=42)
    elif method == "text_marker":
        trigger = "Less is more."
        trigger_type = "sentence"
        location = "initial"
        poisoning_rate = 0.2
        watermark = TextMarker_watrmark.watermark_data(trigger, trigger_type, location, poisoning_rate)
        params = {'t': trigger.replace(' ', '-'), 'type': trigger_type, 'loc': location, 'rate': poisoning_rate}
    elif method == "highest-log-prob":
        watermark = watermark_highest_based_model_prob.add_watermark_highest_prob(args.k, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k}
    elif method == "highest_based_prompt":
        watermark = watermark_based_prompt.add_watermark(watermark_based_prompt.prompt2, args.k)
        params = {'prompt': 'prompt2', 'k': args.k}

    # Robustness methods
    elif method == "highest-to-random":
        watermark = robustness_watermark.add_watermark_highest_to_random(args.k, args.mode, args.synonym_method, seed=args.seed, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k, 'seed': args.seed}
    elif method == "highest-to-lowest":
        watermark = robustness_watermark.add_watermark_highest_to_lowest(args.k, args.mode, args.synonym_method, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k}
    elif method == "random-to-random":
        watermark = robustness_watermark.add_watermark_random_to_random(args.k, args.mode, args.synonym_method, seed=args.seed, syn_threshold=args.context_th)
        params = {'syn': args.synonym_method, 'k': args.k, 'seed': args.seed}
    elif method[:10] == "paraphrase":
        watermark = paraphrase_text(method[11:], threshold=0.6)
        params = {'paraphrase-method': method[11:], 'th': 0.6}
    elif method == "deduplication-exact":
        watermark = deduplication_filter.add_watermark_ngram_exact(n= args.n, model_name=args.target_model)
        params = {'n': args.n, "model": args.target_model.split('/')[-1]}
    elif method == "deduplication-fuzzy":
        watermark = deduplication_filter.add_fuzzy_duplicate_filter(threshold=0.8)
        params = {'fuzzy-th': 0.8}


    # Backdoor methods
    elif method == "top-k-higher-backdoor":
        watermark = watermark_backdoor.add_watermark_higher(args.k, mode=args.mode, synonym_method=args.synonym_method, syn_threshold=args.context_th, seed=args.seed, backdoor_percentage=args.p)
        params = {'syn': args.synonym_method, 'k': args.k, 'seed': args.seed, 'p': args.p}

    else:
        print("Method not implemented yet - ", method)

    if method not in non_synonym_methods:
        if args.synonym_method in synonym_threshold_methods:
            params['syn-th'] = args.context_th
        if args.synonym_method in ["lexsub_dropout", "lexsub_concatenation"] or args.synonym_method[:21] == "lexsub_concatenation_":
            params['syn-th'] = args.context_th
    return watermark, params


def extract_timestamp_from_folder(folder_name):
    """
    Extracts a datetime object from a folder name using a predefined timestamp pattern.
    Expected format in folder name: 'YYYY_MM_DD_HH_MM_SS'.

    Args:
        folder_name (str): The name of the folder to extract timestamp from.

    Returns:
        Optional[datetime]: The extracted datetime object, or None if parsing fails.
    """
    # Define the regex pattern for the timestamp
    timestamp_pattern = r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'

    # Search for the pattern in the folder name
    match = re.search(timestamp_pattern, folder_name)

    if match:
        timestamp_str = match.group()
        try:
            # Convert to datetime object
            return datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S')
        except ValueError:
            return None
    else:
        return None


def find_existing_model_folder(model_name, data_name, num_epochs, directory):
    """
    Searches for an existing fine-tuned model folder that matches the naming pattern.

    The function looks for folders containing the model name, data name, and epoch count.
    Among matches, the newest folder is selected based on a timestamp in the folder name.

    Args:
        model_name (str): Name or path of the base model.
        data_name (str): Name of the dataset used for fine-tuning.
        num_epochs (int): Number of training epochs.
        directory (str): Path to the models directory.

    Returns:
        Optional[str]: Path to the most recent matching folder, or None if not found.
    """
    # Construct the pattern, ignoring the current_time part
    model_name_base = model_name.split('/')[-1]
    data_name = data_name.split('/')[-1]
    data_name = data_name.replace(".csv", "")
    search_pattern = f"{model_name_base}_{data_name}_QLORA_*_epochs_{num_epochs}_Merged"
    # Create a search pattern for folders
    search_path = os.path.join(directory, search_pattern)
    # Use glob to find matching folders
    matching_folders = glob.glob(search_path)
    print(f'Found {len(matching_folders)} matching folders for {model_name_base} and {data_name}')
    print(f'Searching for: {search_path}')
    # Return the first match found, or None if no folder matches
    if matching_folders:
        # Extract timestamps and sort by the newest
        folders_with_timestamps = [(folder, extract_timestamp_from_folder(folder)) for folder in matching_folders]
        # Filter out folders where timestamp extraction failed
        valid_folders = [(folder, ts) for folder, ts in folders_with_timestamps if ts is not None]

        if valid_folders:
            # Sort by timestamp descending
            valid_folders.sort(key=lambda x: x[1], reverse=True)
            newest_folder = valid_folders[0][0]
            print(f"Found newest model folder: {newest_folder}")
            return newest_folder
        else:
            print("No valid timestamps found in the folder names.")
            return None
    else:
        return None


def find_existing_unmerged_model_folder(model_name, data_name, num_epochs, directory):
    """
    Searches for an existing unmerged adapter folder that matches the naming pattern.

    The function looks for adapter folders (unmerged) containing the model name, data name, and epoch count.
    Among matches, the newest folder is selected based on a timestamp in the folder name.

    Args:
        model_name (str): Name or path of the base model.
        data_name (str): Name of the dataset used for fine-tuning.
        num_epochs (int): Number of training epochs.
        directory (str): Path to the 'Unmerged' models directory.

    Returns:
        Optional[str]: Path to the most recent matching folder, or None if not found.
    """
    model_name_base = model_name.split('/')[-1]
    data_name = data_name.split('/')[-1].replace(".csv", "")
    search_pattern = f"{model_name_base}_{data_name}_QLORA_*_epochs_{num_epochs}"
    search_path = os.path.join(directory, "Unmerged", search_pattern)

    matching_folders = glob.glob(search_path)
    print(f'Found {len(matching_folders)} unmerged folders for {model_name_base} and {data_name}')
    print(f'Searching for: {search_path}')

    if matching_folders:
        folders_with_timestamps = [(folder, extract_timestamp_from_folder(folder)) for folder in matching_folders]
        valid_folders = [(folder, ts) for folder, ts in folders_with_timestamps if ts is not None]

        if valid_folders:
            valid_folders.sort(key=lambda x: x[1], reverse=True)
            newest_folder = valid_folders[0][0]
            print(f"Found newest unmerged model folder: {newest_folder}")
            return newest_folder
        else:
            print("No valid timestamps found in unmerged folder names.")
    return None


def find_existing_Pretrained_model_folder(model_name, data_name, num_epochs, directory):
    """
    Searches for an existing unmerged adapter folder that matches the naming pattern.

    The function looks for adapter folders (unmerged) containing the model name, data name, and epoch count.
    Among matches, the newest folder is selected based on a timestamp in the folder name.

    Args:
        model_name (str): Name or path of the base model.
        data_name (str): Name of the dataset used for fine-tuning.
        num_epochs (int): Number of training epochs.
        directory (str): Path to the 'Unmerged' models directory.

    Returns:
        Optional[str]: Path to the most recent matching folder, or None if not found.
    """
    model_name_base = model_name.split('/')[-1]
    data_name = data_name.split('/')[-1].replace(".csv", "")
    search_pattern = f"{model_name_base}_{data_name}_PRETRAINED_*_epochs_{num_epochs}"
    search_path = os.path.join(directory, "Pretrained", search_pattern)

    matching_folders = glob.glob(search_path)
    print(f'Found {len(matching_folders)} Pretrained folders for {model_name_base} and {data_name}')
    print(f'Searching for: {search_path}')

    if matching_folders:
        folders_with_timestamps = [(folder, extract_timestamp_from_folder(folder)) for folder in matching_folders]
        valid_folders = [(folder, ts) for folder, ts in folders_with_timestamps if ts is not None]

        if valid_folders:
            valid_folders.sort(key=lambda x: x[1], reverse=True)
            newest_folder = valid_folders[0][0]
            print(f"Found newest pretrained model folder: {newest_folder}")
            return newest_folder
        else:
            print("No valid timestamps found in unmerged folder names.")
    return None



def openai_workflow(args, filename1, filename2, filename):
    """
    Handles fine-tuning and watermark detection for OpenAI GPT models.

    If `use_existing` is set to avoid retraining, it skips fine-tuning and proceeds
    directly to detection.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        filename1 (str): Path to the member dataset.
        filename2 (str): Path to the non-member dataset.
        filename (str): Path to the full combined dataset (CSV).
    """
    use_existing_model = True if args.use_existing.lower() in ['all', 'model'] else False
    if not use_existing_model:
        print("Start Fine-tuning the model", flush=True)
        # Fine-tune the model
        model = openAI_finetune_GPT.main(args.target_model, filename1)
        if not model:
            print("Failed to fine-tune the model.")
            return
        print("Finished Fine-tuning the model: ", model, flush=True)
        args.target_model = model
    # Use the fine-tuned model for watermark detection
    print("Start detection on the model", flush=True)
    watermark_detection_2.main(args)


def check_if_model_exist_or_train(args, model, data, use_existing_model, train_mode, models_dir):
    new_model = None
    if train_mode.lower() == "finetune" or train_mode.lower() == "qlora":
        if use_existing_model:
            # Find the existing model folder
            new_model = find_existing_model_folder(model, data, num_epochs=1, directory=models_dir)
            if new_model:
                args.target_model = new_model
                print(f"Using existing model folder: {new_model}")
            else:
                unmerged_model = find_existing_unmerged_model_folder(model, data, num_epochs=1, directory=models_dir)
                if unmerged_model:
                    print(f"Using unmerged model folder: {unmerged_model}")
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    new_model = QLora_finetune_LLM.merge_and_upload_model(model, unmerged_model, tokenizer=tokenizer)
                    if new_model:
                        print(f"Successfully merged and uploaded model: {new_model}")
                        args.target_model = new_model
                    else:
                        print("Failed to merge and upload the model.")

                else:
                    print("No existing model found. Proceeding to fine-tune.")
        if not new_model:
            print("Start Fine-tuning the model", flush=True)
            # Fine-tune the model
            new_model = QLora_finetune_LLM.main(model, data, base_path=models_dir)
            print("Finished Fine-tuning the model: ", new_model, flush=True)
            args.target_model = new_model
    elif train_mode == "pretrain":
        if use_existing_model:
            # Find the existing model folder
            new_model = find_existing_Pretrained_model_folder(model, data, num_epochs=1, directory=models_dir)
            if new_model:
                args.target_model = new_model
                print(f"Using existing model folder: {new_model}")
            else:
                print("No existing model found. Proceeding to pretrain.")
        if not new_model:
            new_model = pretrain_LLM.main(model, data, base_path=models_dir)
            args.target_model = new_model
    else:
        new_model = model
        print(f"Using existing model: {model}")

    return new_model


if __name__ == '__main__':
    """
    Main execution block for the watermarking and detection pipeline.

    Steps:
        1. Parse arguments.
        2. Initialize configuration and watermarking method.
        3. Load and optionally watermark data.
        4. Fine-tune a model or load an existing one.
        5. Run watermark detection on the target model.

    Execution:
        Run this script directly to start the end-to-end watermarking workflow.
    """
    print("Start")
    # import pdb; pdb.set_trace()
    args = Options()
    args = args.parser.parse_args()
    config = Config(args)
    args.output_dir = config.data_dir
    print(f"Start watermarking the data with method: {args.method}", flush=True)
    watermark = None
    params = {}
    watermark, params = init_watermark(args, args.method)
    # Load and split the data
    filename1, filename2, filename = load_and_split_data(config, args, args.method, watermark, params)
    print(f"File saved to {filename1}, {filename2}")
    args.data = filename.replace(".csv", ".jsonl")
    use_existing_model = True if args.use_existing.lower() in ['all', 'model'] else False
    base_model = args.target_model
    if "gpt" in base_model.lower():
        openai_workflow(args, filename1, filename2, filename)
        exit(0)
    data = filename1
    models_dir = os.path.join(config.data_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    # Check if the model exists or train a new one
    new_model = check_if_model_exist_or_train(args, base_model, data, use_existing_model, args.train_mode, models_dir)
    args.target_model = new_model

    if args.post_training is not None and args.post_training.lower() != "none":
        print("Start post-training the model")
        # Post-training the model
        if args.post_training.lower() == "triviaqa":
            new_model = instruction_tuning.main("muscle-memory/trivia_llama_response", base_model, args.target_model, models_dir, train_mode=args.train_mode)
            args.target_model = new_model
        else:
            if args.post_training.lower() == "bookmia":
                args.split = 0
            else:
                args.split = 10000
            filter = "Non-member" if args.post_training in ["BookMIA", "Arxiv"] else "all"
            no_member_str = "no_member_" if args.post_training in ["BookMIA", "Arxiv"] else ""
            split_str = f"split_{args.split}_" if args.split > 0 else ""
            count_str = f"{config.count // 1000}k" if config.count >= 1000 else str(config.count)
            output_file = os.path.join(config.data_dir, "Datasets") + f"/{args.post_training}_{no_member_str}original_all_data_{split_str}{count_str}.csv"
            post_training_dataset = process_data.load_clea_data_as_texts(
                mode=args.post_training, from_idx=0, count=config.count,
                split=args.split, output_file=output_file, watermark=None, filter=filter, output_csv_path=output_file
            )
            print("Post-training dataset created: ", post_training_dataset)
            print("Model: ", args.target_model)
            new_model = check_if_model_exist_or_train(args, args.target_model, post_training_dataset, use_existing_model, args.train_mode, models_dir)
            args.target_model = new_model
            # print("Post-training dataset not implemented yet")

    # Use the fine-tuned model
    print("Start detection on the model", flush=True)
    watermark_detection_2.main(model_path=args.target_model, data_path=args.data, output_dir=args.output_dir, mode=args.mode)
    print("Done", flush=True)

