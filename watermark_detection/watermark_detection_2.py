import json
import logging
import datetime
import os

import numpy as np
import openai
import pandas as pd
from openai import OpenAI
import tiktoken
from tqdm import tqdm

from . import eval_2
from utils import process_data
from watermarks.basic_watermark import bottom_k_entropy_words
from utils.create_map_entropy_both import create_entropy_map, create_line_to_top_words_map
logging.basicConfig(level='ERROR')
from pathlib import Path
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from options import Options, config_instance
# from .eval import *
import torch.nn.functional as F
import spacy

MAX_LEN_LINE_GENERATE = 40
MIN_LEN_LINE_GENERATE = 7
MAX_LEN_LINE = 10000
TOP_K_ENTROPY = 2


def load_model(name1):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)
    return model1, tokenizer1


def load_local_model(name1):
    if "davinci" in name1 or "gpt" in name1:
        model1 = name1
        tokenizer1 = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    else:
        model1 = AutoModelForCausalLM.from_pretrained(Path(name1), return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(Path(name1))

    return model1, tokenizer1


def calculate_perplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """

    # Check if openai model
    if isinstance(model, str) and "gpt" in model:
        # Use OpenAI API for perplexity calculation
        return calculate_perplexity_openai(sentence, model_name=model)


    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    try:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        all_prob = []
        input_ids_processed = input_ids[0][1:]
        for i, token_id in enumerate(input_ids_processed):
            probability = probabilities[0, i, token_id].item()
            all_prob.append(probability)
        return torch.exp(loss).item(), all_prob, loss.item(), logits, input_ids_processed
    except RuntimeError as e:
        # print(f"Error: {e} in sentence: {sentence}")
        print("Length of sentence: ", len(sentence))
        raise e

        # if "out of memory" in str(e):
        #     print("CUDA out of memory. Trying to free up memory...")
        #     torch.cuda.empty_cache()
        #     # Optionally, reduce any model or input sizes here if possible.
        # else:
        #     raise e  # Re-raise the exception if it is not a memory error.


def calculate_perplexity_openai(prompt, model_name="text-davinci-003"):
    """
    Calculate perplexity using the OpenAI Completion API.
    Note: You must use a model that supports the logprobs parameter
    (e.g. 'text-davinci-003', 'davinci', etc.).

    :param prompt: str, the text prompt
    :param model_name: str, the model to use (default: "text-davinci-003")
    :return: tuple
        perplexity (float),
        list of valid token logprobs (list),
        mean of those logprobs (float),
        original list of token logprobs (list),
        tokens (list)
    """
    # Clean out any null characters that might cause prompt issues
    prompt = prompt.replace('\x00', '')

    # Load API key from config.json
    # with open("config.json", "r") as file:
    #     config = json.load(file)
    openai.api_key = config_instance.api_key
    client = OpenAI(
        api_key=config_instance.api_key
    )

    try:
        # Request logprobs from an OpenAI model that supports them
        response = client.chat.completions.create(
            model=model_name,  # Use the provided model name
            # prompt=prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1,  # No additional tokens needed, just echo the prompt
            temperature=1.0,
            logprobs=True,  # Request top-5 logprobs
            top_logprobs=5,
            # echo=True  # Echo back the prompt so we get logprobs for it
        )
    except Exception as e:
        error_message = str(e)
        # Check if content filter triggered or other error
        if 'content_filter' in error_message.lower():
            print("Content filter triggered. Please modify the prompt.")
            return None, None, None, None, None
        else:
            print(f"An error occurred: {error_message}")
            return None, None, None, None, None

    # Extract the first choice
    choice = response.choices[0]

    # Extract token logprobs and tokens from structured object
    token_logprobs = [t.logprob for t in choice.logprobs.content]
    tokens = [t.token for t in choice.logprobs.content]

    # Filter out None values
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

    if not valid_logprobs:
        print("No valid logprobs returned.")
        return None, None, None, token_logprobs, tokens

        # Calculate perplexity = exp(-mean(logprob))
    perplexity = np.exp(-np.mean(valid_logprobs))

    # Convert to tensor for softmax
    logits_tensor = torch.tensor(valid_logprobs)
    softmax_probs = F.softmax(logits_tensor, dim=-1)

    return (
        perplexity,
        valid_logprobs,
        np.mean(valid_logprobs),
        logits_tensor,
        tokens
    )


def inference(model1, tokenizer1, text, label, name, line_to_top_words_map, entropy_map, data_name=None):
    pred = {}
    pred["FILE_PATH"] = name
    pred["label"] = label
    if data_name:
        pred["data_name"] = data_name

    # Check if openai model
    if isinstance(model1, str) and "gpt" in model1:
        # Use OpenAI API for perplexity calculation
        p1, all_prob, p1_likelihood, logits, input_ids_processed = calculate_perplexity_openai(text, model_name=model1)
        p_lower, all_prob_lower, p_lower_likelihood, logits_lower, input_ids_processed_lower = calculate_perplexity_openai(text.lower(), model_name=model1)
    else:
        p1, all_prob, p1_likelihood, logits, input_ids_processed = calculate_perplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, all_prob_lower, p_lower_likelihood, logits_lower, input_ids_processed_lower = calculate_perplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    # ppl
    pred["ppl"] = p1

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl_lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl_zlib"] = np.log(p1) / zlib_entropy

    # min-k prob
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    # max-k prob
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[-k_length:]
        pred[f"Max_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    # Min-K++
    if isinstance(model1, str) and "gpt" in model1:
        # Use OpenAI API for perplexity calculation
        input_ids = torch.tensor(tokenizer1.encode(text)).unsqueeze(0)
    else:

        input_ids = torch.tensor(tokenizer1.encode(text)).unsqueeze(0).to(model1.device)
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        pred[f"MinK++_{ratio * 100}% Prob"] = np.mean(topk).item()


    tokens = tokenizer1.tokenize(text)
    concatenated_tokens = "".join(token for token in tokens)

    mink_plus = mink_plus.cpu()

    # Define the values of k you want to iterate over
    k_values = range(1, 10) # all k from 1 to 9 inclusive

    # Create a list of bottom k words once
    all_bottom_k_words = {}

    for line_num, top_words in line_to_top_words_map.items():
        bottom_k_words = bottom_k_entropy_words(" ".join(top_words), entropy_map, max(k_values))
        all_bottom_k_words[line_num] = bottom_k_words

    # Intermediate storage for results
    intermediate_results = {
        "relevant_log_probs": [],
        "relevant_log_probs_zlib": [],
        "relevant_log_probs_kpp": [],
        "relevant_log_probs_one_token": [],
        "relevant_log_probs_one_token_kpp": [],
        "relevant_indexes": []
    }

    # Process bottom k words once, ensuring lowest to highest entropy processing
    for line_num, bottom_k_words in all_bottom_k_words.items():
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
                        intermediate_results["relevant_log_probs_one_token_kpp"].append((i, mink_plus[start_token_index]))
                    for idx in range(start_token_index, end_token_index + 1):
                        if idx < len(all_prob):
                            intermediate_results["relevant_log_probs"].append((i, all_prob[idx]))
                            if zlib_entropy != 0:
                                intermediate_results["relevant_log_probs_zlib"].append(
                                    (i, np.log(abs(all_prob[idx])) / zlib_entropy))
                            intermediate_results["relevant_log_probs_kpp"].append((i, mink_plus[idx]))
                            intermediate_results["relevant_indexes"].append((i, idx))

    # Calculate and store results for each k value
    for k in k_values:
        relevant_log_probs = [val for i, val in intermediate_results["relevant_log_probs"] if i < k]
        relevant_log_probs_zlib = [val for i, val in intermediate_results["relevant_log_probs_zlib"] if i < k]
        relevant_log_probs_kpp = [val for i, val in intermediate_results["relevant_log_probs_kpp"] if i < k]
        relevant_log_probs_one_token = [val for i, val in intermediate_results["relevant_log_probs_one_token"] if i < k]
        relevant_log_probs_one_token_kpp = [val for i, val in intermediate_results["relevant_log_probs_one_token_kpp"]
                                            if i < k]

        if relevant_log_probs:
            sentence_log_likelihood = np.mean(relevant_log_probs)
            pred[f"sentence_entropy_log_likelihood_k={k}"] = sentence_log_likelihood

        if relevant_log_probs_zlib:
            sentence_log_likelihood_zlib = np.mean(relevant_log_probs_zlib)
            pred[f"sentence_entropy_log_likelihood_zlib_k={k}"] = sentence_log_likelihood_zlib

        if relevant_log_probs_kpp:
            sentence_log_likelihood_kpp = np.mean(relevant_log_probs_kpp)
            pred[f"sentence_entropy_log_likelihood_kpp_k={k}"] = sentence_log_likelihood_kpp

        if relevant_log_probs_one_token:
            sentence_log_probs_one_token = np.mean(relevant_log_probs_one_token)
            pred[f"sentence_log_probs_one_token_k={k}"] = sentence_log_probs_one_token

        if relevant_log_probs_one_token_kpp:
            sentence_log_probs_one_token_kpp = np.mean(relevant_log_probs_one_token_kpp)
            pred[f"sentence_log_probs_one_token_k={k}"] = sentence_log_probs_one_token_kpp

        # first_k = False  # Set the flag to False after processing the first k value

    # # Random Sampling of Words
    # for k in k_values:
    #     random_word_probs = random.sample(all_prob, k)
    #     pred[f"random_words_mean_prob_k={k}"] = np.mean(random_word_probs)

    # Process lower case tokens and top words for all_prob_lower
    tokens_lower = tokenizer1.tokenize(text.lower())
    concatenated_tokens_lower = "".join(token for token in tokens_lower)
    relevant_log_probs_lower = []
    relevant_log_probs_zlib_lower = []
    relevant_indexes_lower = []
    # Process top words from all lines
    for line_num, top_words in line_to_top_words_map.items():
        # print(top_words)
        for word in top_words:
            word = word.lower()
            if word in concatenated_tokens_lower:
                start_index = concatenated_tokens_lower.find(word)
                end_index = start_index + len(word)
                start_token_index = end_token_index = None
                current_length = 0
                for i, token in enumerate(tokens_lower):
                    current_length += len(token)
                    if current_length > start_index and start_token_index is None:
                        start_token_index = i
                    if current_length >= end_index:
                        end_token_index = i
                        break
                if start_token_index is not None and end_token_index is not None:
                    for idx in range(start_token_index, end_token_index + 1):
                        if idx < len(all_prob_lower):
                            relevant_log_probs_lower.append(all_prob_lower[idx])
                            if zlib != 0:
                                relevant_log_probs_zlib_lower.append(np.log(abs(all_prob_lower[idx])) / zlib_entropy)
                            relevant_indexes_lower.append(idx)

    if relevant_log_probs_lower:
        sentence_log_likelihood_lower = np.mean(relevant_log_probs_lower)
        pred["sentence_entropy_log_likelihood_lower"] = sentence_log_likelihood_lower
    if relevant_log_probs_zlib_lower:
        sentence_log_likelihood_zlib_lower = np.mean(relevant_log_probs_zlib_lower)
        pred["sentence_entropy_log_likelihood_zlib_lower"] = sentence_log_likelihood_zlib_lower

    return pred


def evaluate_data(test_data, model1, tokenizer1, col_name, modelname1, mode):
    print(f"all data size: {len(test_data)}")
    print(f"mode: {mode}")
    all_output = []
    test_data = test_data
    nlp_spacy = spacy.load("en_core_web_sm")
    entropy_map = create_entropy_map(test_data, mode=mode)
    # print("test_data:", test_data)
    num_records_too_long = 0
    for ex in tqdm(test_data):
        text = ex[col_name]
        data_name = ex.get("data_name", None)
        # print("text: ", text)
        # if len(text.split()) < MIN_LEN_LINE_GENERATE or len(text.split()) > MAX_LEN_LINE_GENERATE:
        #     continue
        if len(text) > MAX_LEN_LINE:
            print(f"Text too long: {len(text)} make it shorter")
            text = text[:MAX_LEN_LINE]
            num_records_too_long += 1
            # continue
        line_to_top_words_map, sentences = create_line_to_top_words_map(
            text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY, nlp_spacy
        )
        new_ex = inference(model1, tokenizer1, text, ex['label'], modelname1, line_to_top_words_map, entropy_map, data_name)
        all_output.append(new_ex)
    print(f"Number of records too long: {num_records_too_long} out of {len(test_data)}, {num_records_too_long / len(test_data) * 100}%")
    print(f"Max length of line: {MAX_LEN_LINE}")
    return all_output


def main(args):
    # args.output_dir = f"{args.output_dir}/{args.target_model.rstrip('/').split('/')[-1]}_{args.target_model.rstrip('/').split('/')[-1]}/{args.key_name}"
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Start watermark detection")
    print(f"Target model: {args.target_model}")
    print(f"Data: {args.data}")
    # load model and data
    model1, tokenizer1 = load_local_model(args.target_model)
    if "jsonl" in args.data:
        data = eval_2.load_jsonl(f"{args.data}")
    else:  # load data from huggingface
        dataset = process_data.load_data(mode=args.mode)
        data = eval_2.convert_huggingface_data_to_list_dic(dataset)
        data = data[0]
    args.key_name = "input"
    all_output = evaluate_data(data, model1, tokenizer1, args.key_name, args.target_model, args.mode)
    dataset = args.data.rstrip('/').split('/')[-1].split('.')[0]
    model = args.target_model.rstrip('/').split('/')[-1]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = f"E=MIA_detection"
    folder = f"M={model}_{current_time}"
    result_folder = f"{args.output_dir}/{dataset}/{folder}"
    os.makedirs(result_folder, exist_ok=True)
    file_preds = f"{result_folder}/preds_{kind}.csv"
    eval_2.write_to_csv_pred_min_k(all_output, file_preds)
    print(f"Results preds saved to {file_preds}")
    eval_2.evaluate_like_min_k(file_preds, kind=kind)

def main2(target_model, data, output_dir, mode="Texts"):
    print("Start watermark detection")
    print(f"Target model: {target_model}")
    print(f"Data: {data}")
    # load model and data
    model1, tokenizer1 = load_local_model(target_model)
    if "jsonl" in data:
        data = eval_2.load_jsonl(f"{data}")
    else:  # load data from huggingface
        dataset = process_data.load_data(mode=mode)
        data = eval_2.convert_huggingface_data_to_list_dic(dataset)
        data = data[0]
    all_output = evaluate_data(data, model1, tokenizer1, "input", target_model, mode)
    dataset = data.rstrip('/').split('/')[-1].split('.')[0]
    model = target_model.rstrip('/').split('/')[-1]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = f"E=MIA_detection"
    folder = f"M={model}_{current_time}"
    result_folder = f"{output_dir}/{dataset}/{folder}"
    os.makedirs(result_folder, exist_ok=True)
    file_preds = f"{result_folder}/preds_{kind}.csv"
    eval_2.write_to_csv_pred_min_k(all_output, file_preds)
    print(f"Results preds saved to {file_preds}")
    eval_2.evaluate_like_min_k(file_preds, kind=kind)
    metrics_file = f"{result_folder}/metrics_{kind}.csv"
    metrics_df = pd.read_csv(metrics_file)
    return metrics_df




if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    main(args)
