import re
from collections import Counter
from collections import OrderedDict
# import Levenshtein
from transformers import AutoTokenizer


def is_valid_english_token(text):
    """Check if a text contains only English characters and common punctuation."""
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;:',.<>?/\\\"`~ ")
    return all(char in allowed_chars for char in text)


def remove_websites(text):
    """Remove URLs from a given text."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"  # Full URLs
        r"|www\.[a-zA-Z0-9\-]+(?:\.[a-zA-Z]{2,})+"  # Domains starting with www
        r"|\b[a-zA-Z0-9\-.]+(?:\.[a-zA-Z]{2,})(?:/[^\s]*)?\b"  # Standalone domains
    )
    return re.sub(url_pattern, "", text)


def filter_texts_by_ngram_exact(data, n, tokenizer, valid_ngrams=[]):
    """
    For each text in data, tokenize, extract all n-grams,
    and return only those n-grams that:
      - contain at least 2 unique tokens,
      - convert to valid English (by your filter),
      - start with a space.
    Returns a list of texts, one per original text (joined n-grams).
    """
    filtered_texts = []
    for text in data:
        text = remove_websites(text)
        tokens = tokenizer.tokenize(text)
        # valid_ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            if len(set(ngram)) > 1:
                ngram_text = tokenizer.convert_tokens_to_string(list(ngram))
                if is_valid_english_token(ngram_text) and ngram_text.startswith(' '):
                    valid_ngrams.append(ngram_text)
        filtered_texts.append(" ".join(valid_ngrams))
    return filtered_texts


def add_watermark_ngram_exact(model_name, n):
    """
    Returns a function that, when called with a list of texts, produces for each text a string with
    only the valid n-grams that haven't been seen in previous texts (across all calls).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def watermarked_sentences(data):
        # nonlocal seen_ngrams
        seen_ngrams = set()
        out_texts = []
        for text in data:
            text = remove_websites(text)
            tokens = tokenizer.tokenize(text)
            text_ngrams = [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]
            # Only keep valid English, and with at least 2 distinct tokens
            filtered = [
                ngram for ngram in text_ngrams
                if is_valid_english_token(tokenizer.convert_tokens_to_string(list(ngram)))
                and len(set(ngram)) > 1
            ]
            # Remove those that have been seen before
            unseen = [ngram for ngram in filtered if ngram not in seen_ngrams]
            # Add to seen set
            seen_ngrams.update(unseen)
            # Convert to string
            out_str = " ".join(tokenizer.convert_tokens_to_string(list(ng)) for ng in unseen)
            out_texts.append(out_str)
        return out_texts

    return watermarked_sentences

def add_fuzzy_duplicate_filter(threshold=0.8):
    """
    Returns a function that, when called repeatedly, keeps a global memory of all texts ever seen,
    and always returns the de-duplicated set (one per fuzzy duplicate group).
    """
    all_texts = []

    def deduplicate(data):
        # Add new (preprocessed) texts to the global memory
        for text in data:
            cleaned = remove_websites(text.strip())
            if cleaned:  # ignore empty
                all_texts.append(cleaned)

        # Fuzzy deduplication (persistent)
        # We'll only keep one representative for each group of texts with similarity > threshold
        # To keep order: use OrderedDict
        unique_texts = []
        used = set()

        for i, text_i in enumerate(all_texts):
            if i in used:
                continue
            unique_texts.append(text_i)
            for j in range(i+1, len(all_texts)):
                if j in used:
                    continue
                text_j = all_texts[j]
                # Calculate Levenshtein similarity
                sim = Levenshtein.ratio(text_i, text_j)
                if sim > threshold:
                    used.add(j)
        return unique_texts

    return deduplicate