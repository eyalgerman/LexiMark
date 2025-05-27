import itertools
import json
import re
from datasets import load_dataset
from pathlib import Path
from langdetect import detect, LangDetectException
import nltk
from . import process_csv

nltk.download('punkt')


def get_text_files(folder_path):
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]


def load_data_pile(train_val_pile = "validation", num_samples=100000):
    grouped_data = {}
    train_val_pile = "validation"
    # data_sources = load_dataset("monology/pile-uncopyrighted", split=train_val_pile, streaming=True)
    pile_val_file = "/dt/shabtaia/dt-sicpa/eyal/LLM/Datasets/pile-uncopyrighted-val.jsonl"
    with open(pile_val_file, 'r') as file:
        for line in file:
            sample = json.loads(line)
            meta_label = sample['meta']['pile_set_name']

            # Dynamically initialize the group if the meta_label is not yet in grouped_data
            if meta_label not in grouped_data:
                grouped_data[meta_label] = []
                print(f"Initializing group for meta label: {meta_label}")
            grouped_data[meta_label].append(sample)

    # Convert the grouped data to an array of arrays
    grouped_array = [[meta_label, grouped_data[meta_label]] for meta_label in grouped_data]

    print("Returning grouped data as an array of arrays...")
    return grouped_array


def load_data_not_member(mode='Books', from_idx=0, count=1000):
    if mode == 'Arxiv':
        # Define months for non-member and member data in a single dictionary
        year_months = {
            'non_member': ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07'],
            'member': ['2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
        }
        filtered_data = []
        # Process datasets according to the labels
        for label, months in year_months.items():
            assigned_label = 0 if label == 'non_member' else 1
            for month in months:
                filtered_data_temp = []  # Temporary list to store records from the current dataset
                dataset = load_dataset("RealTimeData/arxiv_alltime", month, split='train[:470]')
                for record in dataset:
                    record['label'] = assigned_label
                    if assigned_label == 0:  # We are only interested in non-member data
                        filtered_data_temp.append(record)
                        if len(filtered_data_temp) >= count:
                            filtered_data.append(filtered_data_temp)
                            return filtered_data  # Return early if we reach the count limit
                filtered_data.append(filtered_data_temp)  # Append the dataset records as an array
    if mode == 'Books':
        data_sources = get_text_files("Books")
        # Assuming get_text_files returns a list of dictionaries
        # Each dictionary should have a 'label' key
        filtered_data = [record for record in data_sources if record['label'] == 0]
        filtered_data = [filtered_data]
    elif mode == 'WikiMIA':
        data_sources = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length256")
        # Assuming the dataset has a 'label' column
        filtered_data = [record for record in data_sources if record['label'] == 0]
        filtered_data = filtered_data[:count]
        filtered_data = [filtered_data]
    elif mode == 'BookMIA':
        data_sources = load_dataset("swj0419/BookMIA", split=f"train[{from_idx}:]")
        # Assuming the dataset has a 'label' column
        filtered_data = [record for record in data_sources if record['label'] == 0]
        filtered_data = filtered_data[:count]
        filtered_data = [filtered_data]

    # Return the first 'count' records after filtering
    # print("filtered_data:", filtered_data[0][0])
    return filtered_data


def load_data(mode='Books', folder1="None", folder2="None", from_idx=0, to_idx=None):
    if mode == 'Arxiv':
        data_sources = [(load_dataset("RealTimeData/arxiv_alltime", '2017-01', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-02', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-03', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-04', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-05', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-06', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2017-07', split='train[:470]'), 0),
                        (load_dataset("RealTimeData/arxiv_alltime", '2023-08', split='train[:520]'), 1),
                        (load_dataset("RealTimeData/arxiv_alltime", '2023-09', split='train[:520]'), 1),
                        (load_dataset("RealTimeData/arxiv_alltime", '2023-10', split='train[:520]'), 1),
                        (load_dataset("RealTimeData/arxiv_alltime", '2023-11', split='train[:520]'), 1),
                        (load_dataset("RealTimeData/arxiv_alltime", '2023-12', split='train[:520]'), 1)]

    elif mode == 'Books':
        data_sources = get_text_files(folder1) + get_text_files(folder2)

    elif mode == 'WikiMIA':
        data_sources = [load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length256")]

    elif mode == 'BookMIA':
        if to_idx is None:
            data_sources = [load_dataset("swj0419/BookMIA", split=f"train[{from_idx}:]")]
        else:
            data_sources = [load_dataset("swj0419/BookMIA", split=f"train[{from_idx}:{to_idx}]")]

    elif mode == 'Gut':
        data_sources = [load_dataset("Sagivan100/BooksMIA_Gut", split=f"train")]
    elif mode == 'PILE':
        data_sources = load_data_pile("train", num_samples=1000)
    elif mode[:4] == 'PILE':
        data_name = mode[5:]
        data_name = data_name.replace("_", " ")
        data_sources = load_data_pile()
        data_sources = [data[1] for data in data_sources if data[0] == data_name]
    return data_sources


def split_texts_into_sentences(texts, max_len=2000):
    sentences = []

    for text in texts:
        split_sentences = nltk.sent_tokenize(text)
        temp_sentence = ""
        for sentence in split_sentences:
            # If the current sentence itself exceeds max_len, split it further
            if len(sentence) > max_len:
                parts = [sentence[i:i+max_len] for i in range(0, len(sentence), max_len)]
                sentences.extend(parts)
            elif len(temp_sentence) + len(sentence) <= max_len:
                temp_sentence += (" " + sentence if temp_sentence else sentence)
            else:
                sentences.append(temp_sentence)
                temp_sentence = sentence
        if temp_sentence:  # Add any remaining sentence
            sentences.append(temp_sentence)
    print(f"Split in sentences: {len(sentences)} from {len(texts)} texts")
    print(f"Max sentence length: {max([len(sentence) for sentence in sentences])}")
    return sentences


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def contains_only_allowed_characters(s):
    return re.match(r'^[a-zA-Z0-9 .,]*$', s) is not None


def remove_invalid_characters(sentence):
    # Remove characters that are not valid English letters, digits, or allowed special characters
    return re.sub(r'[^a-zA-Z0-9 ,.!?;:\'\"-()“”’*]+', '', sentence)


def filter_valid_english_sentences(sentences):
    valid_sentences = []
    for sentence in sentences:
        try:
            cleaned_sentence = remove_invalid_characters(sentence)
            # Check if cleaned sentence is valid English and contains only ASCII characters
            # if detect(cleaned_sentence) == 'en' and is_ascii(cleaned_sentence) and contains_only_allowed_characters(cleaned_sentence):
            valid_sentences.append(cleaned_sentence)
        except LangDetectException:
            # If language detection fails, skip the sentence
            continue
    return valid_sentences


def watermark_and_split_data(sentences, watermark=None, split=False, watermark_non_member=True):
    sentences_valid = filter_valid_english_sentences(sentences)
    count = len(sentences_valid)
    count_member = count // 2
    sentences_member = sentences_valid[:count_member]
    sentences_non_member = sentences_valid[count_member:]

    # Split the text into sentences if the split argument is provided and greater than 0
    if split > 0:
        sentences_member = split_texts_into_sentences(sentences_member, max_len=split)
        sentences_non_member = split_texts_into_sentences(sentences_non_member, max_len=split)

    # Check if watermark is a list (or array) of functions
    if watermark:
        if isinstance(watermark, list):
            # Apply each watermark function sequentially to member sentences
            print(f"Start watermarking member with {len(watermark)} functions")
            watermark_sentences_member = sentences_member
            for wm in watermark:
                watermark_sentences_member = wm(watermark_sentences_member)

            # Apply watermark functions to non-member sentences if watermark_non_member is True
            if watermark_non_member:
                print(f"Start watermarking non-member with {len(watermark)} functions")
                watermark_sentences_non_member = sentences_non_member
                for wm in watermark:
                    watermark_sentences_non_member = wm(watermark_sentences_non_member)
            else:
                watermark_sentences_non_member = sentences_non_member
        else:
            # If watermark is a single function, apply it directly
            watermark_sentences_member = watermark(sentences_member)
            if watermark_non_member:
                watermark_sentences_non_member = watermark(sentences_non_member)
            else:
                watermark_sentences_non_member = sentences_non_member
    else:
        watermark_sentences_member = sentences_member
        watermark_sentences_non_member = sentences_non_member

    return watermark_sentences_member, watermark_sentences_non_member


key_name_dict = {
    'Arxiv': 'input',
    'Books': 'snippet',
    'WikiMIA': 'input',
    'BookMIA': 'snippet',
    'Gut': 'snippet',
    'ECHR': 'text'
}


def load_clean_data_and_split(mode='Books', folder1="None", folder2="None", from_idx=0, count=1000, output_file=None, watermark=None, key_name='snippet', filter=None, split=False, watermark_non_member=True):
    key_name = key_name_dict.get(mode, "text")
    if mode == 'PILE':
        return process_data_the_pile(from_idx=from_idx, count=count, output_file=output_file, watermark=watermark, key_name=key_name, filter=filter, split=split, watermark_non_member=watermark_non_member)
    if filter == "Non-member":
        datasets = load_data_not_member(mode=mode, from_idx=from_idx, count=count)
    else:
        datasets = load_data(mode=mode, from_idx=from_idx, to_idx=from_idx+count)
    print("datasets loaded:", len(datasets))
    # dataset = dataset[0]
    # key_name = key_name_dict[mode]
    sentences = []
    for dataset in datasets:
        sentences_dataset = [item[key_name] for item in dataset]
        sentences.extend(sentences_dataset)
        # watermark_sentences = []
        print("sentences loaded:", len(sentences))
    watermark_sentences_member, watermark_sentences_non_member = watermark_and_split_data(sentences, watermark=watermark, split=split, watermark_non_member=watermark_non_member)
    if output_file:
        output_file_member = output_file.replace(".csv", "_member.csv")
        if not watermark_non_member:
            folder = Path(output_file).parent
            count_str = str(count)[0] + "k" if count >= 1000 else str(count)
            output_file_non_member = str(folder) + f"/{mode}_no_member_original_{count_str}_non_member.csv"
        else:
            output_file_non_member = output_file.replace(".csv", "_non_member.csv")
        process_csv.write_sentences_to_csv(watermark_sentences_member,
                                           file_name=output_file_member)
        process_csv.write_sentences_to_csv(watermark_sentences_non_member, file_name=output_file_non_member)
        output_jsonl_path = output_file.replace(".csv", ".jsonl")
        process_csv.csv_to_jsonl(output_file_member, output_file_non_member, output_jsonl_path)
        return output_file_member, output_file_non_member
    return watermark_sentences_member, watermark_sentences_non_member


def process_data_the_pile(from_idx=0, count=1000, output_file=None, watermark=None, key_name='snippet', filter=None, split=False, watermark_non_member=True):
    mode = "PILE"
    if filter == "Non-member":
        datasets = load_data_not_member(mode=mode, from_idx=from_idx, count=count)
    else:
        datasets = load_data(mode=mode, from_idx=from_idx, to_idx=from_idx+count)
    sentences = []
    watermark_sentences_member, watermark_sentences_non_member = [], []
    for dataset in datasets:
        dataset_name = dataset[0]
        dataset = dataset[1]
        sentences_dataset = [item[key_name] for item in dataset]
        sentences.append(sentences_dataset)
        watermark_sentences_member_dataset, watermark_sentences_non_member_dataset = watermark_and_split_data(sentences_dataset, watermark=watermark, split=split, watermark_non_member=watermark_non_member)
        watermark_sentences_member.append((dataset_name, watermark_sentences_member_dataset))
        watermark_sentences_non_member.append((dataset_name, watermark_sentences_non_member_dataset))
    # for sentences_dataset in sentences:
    #     watermark_sentences_member_dataset, watermark_sentences_non_member_dataset = watermark_and_split_data(sentences_dataset, watermark=watermark, split=split, watermark_non_member=watermark_non_member)
    #     watermark_sentences_member.extend(watermark_sentences_member_dataset)
    #     watermark_sentences_non_member.extend(watermark_sentences_non_member_dataset)
    # watermark_sentences_member, watermark_sentences_non_member = watermark_and_split_data(sentences, watermark=watermark, split=split, watermark_non_member=watermark_non_member)
    if output_file:
        output_file_member = output_file.replace(".csv", "_member.csv")
        if not watermark_non_member:
            folder = Path(output_file).parent
            count_str = str(count)[0] + "k" if count >= 1000 else str(count)
            output_file_non_member = str(folder) + f"/{mode}_no_member_original_{count_str}_non_member.csv"
        else:
            output_file_non_member = output_file.replace(".csv", "_non_member.csv")
        process_csv.write_sentences_to_csv_PILE(watermark_sentences_member,
                                                file_name=output_file_member)
        # output_file_member = procces_csv.keep_first_column(output_file_member)
        process_csv.write_sentences_to_csv_PILE(watermark_sentences_non_member, file_name=output_file_non_member)
        output_jsonl_path = output_file.replace(".csv", ".jsonl")
        process_csv.csv_to_jsonl_pile(output_file_member, output_file_non_member, output_jsonl_path)
        # output_file_non_member = procces_csv.keep_first_column(output_file_non_member)
        return output_file_member, output_file_non_member
    return watermark_sentences_member, watermark_sentences_non_member


def load_clea_data_as_texts(mode="BookMIA", from_idx=0, count=1000, output_file=None, watermark=None, filter=None, split=False, watermark_non_member=True, output_csv_path=None):
    key_name = key_name_dict.get(mode, "text")
    if mode == 'PILE':
        return process_data_the_pile(from_idx=from_idx, count=count, output_file=output_file, watermark=watermark,
                                     key_name=key_name, filter=filter, split=split,
                                     watermark_non_member=watermark_non_member)
    if filter == "Non-member":
        datasets = load_data_not_member(mode=mode, from_idx=from_idx, count=count)
    else:
        datasets = load_data(mode=mode, from_idx=from_idx, to_idx=from_idx + count)
    print("datasets loaded:", len(datasets))
    # dataset = dataset[0]
    # key_name = key_name_dict[mode]
    sentences = []
    for dataset in datasets:
        sentences_dataset = [item[key_name] for item in dataset]
        sentences.extend(sentences_dataset)
        # watermark_sentences = []
        print("sentences loaded:", len(sentences))
    watermark_sentences_member, watermark_sentences_non_member = watermark_and_split_data(sentences, watermark=watermark,
                                                                                          split=split,
                                                                                          watermark_non_member=watermark_non_member)
    watermark_sentences = watermark_sentences_member + watermark_sentences_non_member
    if output_csv_path:
        process_csv.write_sentences_to_csv(watermark_sentences, file_name=output_csv_path)
        return output_csv_path

    return watermark_sentences
