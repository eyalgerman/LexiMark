import random
from tqdm import tqdm
from utils.create_map_entropy_both import create_entropy_map
from .watermark_based_k import replace_higher_top_k_entropy_with_higher_entropy
from utils.dict_functions import write_dict_to_file


def add_watermark_higher(k, mode="BookMIA", synonym_method="context", output_file=None, syn_threshold=0.6,
                         backdoor_percentage=0.05, seed=42):
    def watermarked_sentences(data):
        new_sentences = []
        replaced_dict = {}  # Dictionary to keep track of original and replaced words

        if seed is not None:
            random.seed(seed)

        print("Creating entropy map...")
        entropy_map = create_entropy_map(data, mode=mode)

        # Select a subset of data to apply backdoor watermarking based on backdoor_percentage
        backdoor_indices = random.sample(range(len(data)), int(len(data) * backdoor_percentage))

        for idx, text in enumerate(tqdm(data)):
            if idx in backdoor_indices:
                new_text, replaced_dict = replace_higher_top_k_entropy_with_higher_entropy(
                    text, replaced_dict=replaced_dict, entropy_map=entropy_map,
                    k_value=k, synonym_method=synonym_method, syn_threshold=syn_threshold
                )
                new_sentences.append(new_text)
            else:
                new_sentences.append(text)

        if output_file:
            output_file1 = output_file.replace(".csv", "_dict.txt")
            write_dict_to_file(replaced_dict, output_file1)
        return new_sentences

    return watermarked_sentences
