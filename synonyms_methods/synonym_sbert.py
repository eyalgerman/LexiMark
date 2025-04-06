from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.corpus import wordnet

# Load the SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other pre-trained SBERT models

def get_contextual_embedding_sbert(word, sentence, model):
    """
    Generates the contextual embedding of a sentence using a Sentence-BERT (SBERT) model.

    This function tokenizes the sentence and returns the SBERT embedding of the full sentence.
    It does not isolate the embedding of a specific word.

    Args:
        word (str): The target word (currently unused, but included for future flexibility).
        sentence (str): The input sentence for embedding.
        model (SentenceTransformer): Pre-loaded SBERT model used to compute embeddings.

    Returns:
        torch.Tensor: The SBERT embedding of the sentence as a tensor.
    """
    # Embed the sentence using SBERT
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # Tokenize the sentence
    tokenized_sentence = nltk.word_tokenize(sentence)
    word_tokens = nltk.word_tokenize(word)

    return sentence_embedding


def get_synonyms_in_context_sbert(word, sentence, threshold=0.7):
    """
    Finds contextually appropriate synonyms for a word using SBERT-based cosine similarity.

    This function replaces the target word with WordNet synonyms in the sentence,
    computes contextual embeddings using SBERT, and selects those with high semantic
    similarity to the original sentence.

    Args:
        word (str): The word to find synonyms for.
        sentence (str): The sentence providing context for semantic evaluation.
        threshold (float, optional): Cosine similarity threshold to accept a synonym. Defaults to 0.7.

    Returns:
        Set[str]: A set of synonyms that are contextually similar to the original word.
    """
    synonyms = set()
    original_embedding = get_contextual_embedding_sbert(word, sentence, sbert_model)

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')

            # Create a sentence by replacing the original word with the synonym
            new_sentence = sentence.replace(word, synonym)

            # Get the embedding of the synonym in the new sentence
            synonym_embedding = get_contextual_embedding_sbert(synonym, new_sentence, sbert_model)

            # Compute cosine similarity between the original and the synonym embeddings
            similarity = util.cos_sim(original_embedding, synonym_embedding).item()

            # Add the synonym if it meets the similarity threshold
            if similarity >= threshold:
                synonyms.add(synonym)

    return synonyms


def check_synonym_word_sbert(sentence, word, synonym, threshold=0.8):
    """
    Checks whether a given synonym is contextually similar to the original word in a sentence using SBERT.

    It embeds the original sentence and a version with the word replaced by the synonym,
    then computes the cosine similarity between them.

    Args:
        sentence (str): The original sentence containing the word.
        word (str): The original word in the sentence.
        synonym (str): The candidate synonym to evaluate.
        threshold (float, optional): Cosine similarity threshold to consider the synonym valid. Defaults to 0.8.

    Returns:
        bool: True if the synonym passes the similarity threshold, False otherwise.
    """
    # Embed the word and the synonym using SBERT
    original_embedding = get_contextual_embedding_sbert(word, sentence, sbert_model)
    # Create a sentence by replacing the original word with the synonym
    new_sentence = sentence.replace(word, synonym)
    # Get the embedding of the synonym in the new sentence
    synonym_embedding = get_contextual_embedding_sbert(synonym, new_sentence, sbert_model)
    # Compute the cosine similarity between the word and the synonym embeddings
    similarity = util.cos_sim(original_embedding, synonym_embedding).item()
    # Return True if the similarity is above the threshold, False otherwise
    return similarity >= threshold