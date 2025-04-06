import string
import math
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import re
from . import process_data
from watermarks.basic_watermark import bottom_k_entropy_words


def calculate_entropy(word_freq, total_words):
    probability = word_freq / total_words
    return -probability * math.log2(probability)


def create_entropy_map(data_sources, mode='Books'):
    word_counts = Counter()

    if mode == 'Arxiv' or mode == 'ECHR':
        print(f"Started processing for mode: {mode}")
        # db_data = procces_data.load_data(mode="Arxiv")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)
        # for dataset in data_sources:
        #     for item in dataset:
        #         words = item['text'].split()
        #         word_counts.update(words)
    elif mode == 'Books':
        for file_path in data_sources:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    words = line.strip().split()
                    word_counts.update(words)
    elif mode == 'WikiMIA' or mode == 'BookMIA' or mode == "Gut":
        print(f"Started processing for mode: {mode}")
        # dataset_name = "swj0419/WikiMIA" if mode == 'WikiMIA' else "swj0419/BookMIA" if mode == 'BookMIA' else "Sagivan100/BooksMIA_Gut"
        # db_data = load_dataset(dataset_name, split=f"{mode}_length256" if mode == 'WikiMIA' else "train")
        db_data = process_data.load_data(mode=mode)[0]
        # print(db_data)
        # print(data_sources)
        # db_data = data_sources
        for item in db_data:
            text_field = 'input' if mode == 'WikiMIA' else 'snippet'
            words = item[text_field].split()
            word_counts.update(words)
    elif mode == 'PILE' or mode[:4].lower() == 'pile':
        print(f"Started processing for mode: {mode}")
        for sentence in data_sources:
            if isinstance(sentence, dict):
                sentence = sentence['input']
            words = sentence.split()
            word_counts.update(words)

    total_words = sum(word_counts.values())
    # print(f"Total words: {total_words}")
    entropy_map = {word: calculate_entropy(freq, total_words) for word, freq in word_counts.items()}

    return entropy_map


def save_entropy_map(entropy_map, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, entropy in entropy_map.items():
            file.write(f"{word} {entropy}\n")


def load_entropy_map(filename):
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def sort_entropy_map(entropy_map, descending=True):
    return sorted(entropy_map.items(), key=lambda item: item[1], reverse=descending)


def get_text_files(folder_path):
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]


def create_entropy_map_func(mode="BookMIA", folder1="None", folder2="None", train_val_pile="validation"):
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
        data_sources = [load_dataset("swj0419/BookMIA", split=f"train")]

    elif mode == 'Gut':
        data_sources = [load_dataset("Sagivan100/BooksMIA_Gut", split=f"train")]
    elif mode == 'PILE':
        data_sources = process_data.load_data_pile(train_val_pile, num_samples=100000)
    print("Creating entropy map...")
    entropy_map = create_entropy_map(data_sources, mode=mode)

    filename = "data/entropy_map.txt"
    print(f"Saving entropy map to {filename}...")
    save_entropy_map(entropy_map, filename)

    # print("Loading entropy map...")
    loaded_entropy_map = load_entropy_map(filename)

    # print("Sorted Entropy Map:")
    sorted_entropy_map = sort_entropy_map(loaded_entropy_map)
    for word, entropy in sorted_entropy_map[:20]:
        print(f"{word}: {entropy}")

    print("Entropy map loaded and sorted successfully.")
    print("Len of map = ", len(sorted_entropy_map))

    return loaded_entropy_map, data_sources


def strip_punctuation(word):
    return word.strip(string.punctuation)


def create_line_to_top_words_map(text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY,
                                 nlp_spacy):
    # text = text.replace('\n', '')
    doc = nlp_spacy(text)
    # Debugging: convert iterator to list to check content
    all_sentences = list(doc.sents)
    sentences = [sent.text.strip() for sent in all_sentences if
                 MIN_LEN_LINE_GENERATE <= len(sent.text.split()) <= MAX_LEN_LINE_GENERATE]

    line_to_top_words_map = {}

    for line_num, line in enumerate(sentences, 1):
        if line.strip():
            top_k_words = {re.sub(r'^\W+|\W+$', '', word.strip(string.punctuation)) for word in
                           bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY) if ' ' not in word}

            ners = {strip_punctuation(ent.text) for ent in doc.ents if ' ' not in ent.text and ent.sent.text == line}
            unique_words = top_k_words.union(ners)
            line_to_top_words_map[line_num] = list(unique_words)

    # Print the results
    # print("Ten Lowest Entropy Words:", sorted(entropy_map, key=entropy_map.get)[:200])
    # print("Entropy Map:", entropy_map)
    # print("Line to Top Words Map:", line_to_top_words_map)
    # print("The sentences are: ", sentences)
    return line_to_top_words_map, sentences
