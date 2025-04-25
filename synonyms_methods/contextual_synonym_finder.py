import nltk
from nltk.corpus import wordnet
import torch
from transformers import BertTokenizer, BertModel



def get_contextual_embedding(word, sentence, tokenizer, model, device):
    """
    Extracts the contextual embedding of a word within a sentence using a BERT model.

    The function tokenizes the sentence and computes contextual embeddings using BERT.
    It locates the token(s) corresponding to the input word and averages their embeddings.

    Args:
        word (str): The target word to extract embedding for.
        sentence (str): The input sentence providing context.
        tokenizer (transformers.PreTrainedTokenizer): BERT tokenizer.
        model (transformers.PreTrainedModel): BERT model to generate embeddings.
        device (torch.device): Device to run the model on (CPU or CUDA).

    Returns:
        torch.Tensor: The average contextual embedding for the word.
                      Returns None if the word is not found in the tokenized sentence.
    """
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)  # Move inputs to device
    outputs = model(**inputs)

    # Get the embeddings for each token
    embeddings = outputs.last_hidden_state

    # Tokenize the sentence and word
    tokenized_sentence = tokenizer.tokenize(sentence)
    word_tokens = tokenizer.tokenize(word)

    # Get the starting indices of the word tokens in the tokenized sentence
    word_indices = [i for i in range(len(tokenized_sentence)) if
                    tokenized_sentence[i:i + len(word_tokens)] == word_tokens]

    if not word_indices:
        print(f"Error: Word '{word}' not found in the sentence.")
        return None
        # raise ValueError(f"Word '{word}' not found in the sentence.")

    # Calculate the average embedding of the word's tokens
    word_embedding = torch.mean(embeddings[0, word_indices[0]:word_indices[0] + len(word_tokens)], dim=0)

    return word_embedding

def get_synonyms_in_context(word, sentence, threshold=0.7):
    """
    Retrieves synonyms for a word that are semantically similar in the given sentence context using BERT.

    For each WordNet synonym, the function replaces the target word in the sentence,
    computes its contextual embedding using BERT, and keeps it if the cosine similarity
    with the original word's embedding exceeds the threshold.

    Args:
        word (str): The word for which to find contextually appropriate synonyms.
        sentence (str): The sentence providing the context for evaluation.
        threshold (float, optional): Cosine similarity threshold for accepting a synonym. Defaults to 0.7.

    Returns:
        Set[str]: A set of synonyms that are contextually similar based on BERT embeddings.
    """
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    synonyms = set()
    original_embedding = get_contextual_embedding(word, sentence, tokenizer, model, device)
    if original_embedding is None:
        return synonyms

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')

            # Create a sentence by replacing the original word with the synonym
            new_sentence = sentence.replace(word, synonym)

            # Get the embedding of the synonym in the new sentence
            synonym_embedding = get_contextual_embedding(synonym, new_sentence, tokenizer, model, device)
            if synonym_embedding is None:
                continue

            # Compute cosine similarity between the original and the synonym embeddings
            similarity = torch.cosine_similarity(original_embedding, synonym_embedding, dim=0).item()

            # Add the synonym if it meets the similarity threshold
            if similarity >= threshold:
                synonyms.add(synonym)

    return synonyms


def check_synonym_word_bert(sentence, word, synonym, threshold=0.8):
    """
    Checks whether a given synonym is contextually appropriate for a word in a sentence using BERT.

    It replaces the original word with the synonym in the sentence, computes contextual
    embeddings using BERT, and measures cosine similarity.

    Args:
        sentence (str): The original sentence containing the word.
        word (str): The original word in the sentence.
        synonym (str): The candidate synonym to evaluate.
        threshold (float, optional): Cosine similarity threshold to consider the synonym valid. Defaults to 0.8.

    Returns:
        bool: True if the synonym is contextually similar above the threshold, False otherwise.
    """
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Embed the word and the synonym using SBERT
    original_embedding = get_contextual_embedding(word, sentence, tokenizer, model, device)
    # Create a sentence by replacing the original word with the synonym
    new_sentence = sentence.replace(word, synonym)
    # Get the embedding of the synonym in the new sentence
    synonym_embedding = get_contextual_embedding(synonym, new_sentence, tokenizer, model, device)
    # Compute the cosine similarity between the word and the synonym embeddings
    similarity = torch.cosine_similarity(original_embedding, synonym_embedding, dim=0).item()

    # Return True if the similarity is above the threshold, False otherwise
    return similarity >= threshold