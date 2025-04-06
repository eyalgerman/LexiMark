import string
import spacy
from nltk.corpus import wordnet
from . import contextual_synonym_finder
from . import synonym_finder_openai
from . import synonym_sbert
from . import Lexical_Substitution
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def get_synonyms_wordnet(word):
    """
    Retrieves a set of synonyms for a given word using WordNet.

    Args:
        word (str): The word for which to find synonyms.

    Returns:
        Set[str]: A set of synonym strings retrieved from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def clean_punctuation(word):
    """
    Separates and removes leading and trailing punctuation from a word.

    This function identifies punctuation at the beginning and end of the word
    (if present) and returns the leading punctuation, the cleaned word, and
    the trailing punctuation as separate components.

    Args:
       word (str): The input word potentially containing punctuation.

    Returns:
       Tuple[str, str, str]: A tuple containing:
           - leading_punct (str): Punctuation characters at the beginning of the word.
           - clean_word (str): The core word with leading/trailing punctuation removed.
           - trailing_punct (str): Punctuation characters at the end of the word.
    """
    # Separate the leading punctuation from the start of the word
    leading_punct = ''.join([char for index, char in enumerate(word) if char in string.punctuation and index == 0])

    # Separate the trailing punctuation from the end of the word
    trailing_punct = ''.join(
        [char for index, char in enumerate(reversed(word)) if char in string.punctuation and index == 0])

    # Clean the word by removing leading and trailing punctuation only
    if trailing_punct:
        clean_word = word[len(leading_punct):len(word) - len(trailing_punct)]
    else:
        clean_word = word[len(leading_punct):]
    trailing_punct = trailing_punct[::-1]
    # Print the results if they don't match the original word
    # if word != leading_punct + clean_word + trailing_punct:
    #     print(f"Original: {word}, Leading: {leading_punct}, Clean: {clean_word}, Trailing: {trailing_punct}")
    return leading_punct, clean_word, trailing_punct

def get_synonyms_by_different_methods(text, word, synonym_method, syn_threshold=0.8):
    """
    Retrieves synonyms for a given word from text using the specified synonym generation method.

    This function preprocesses the input word by cleaning punctuation, checking for
    stopwords, digits, and named entities, then uses the selected method to find
    contextually appropriate synonyms. The returned synonyms preserve the original
    leading and trailing punctuation of the input word.

    Args:
        text (str): The full input text for context-aware synonym generation.
        word (str): The target word to replace or find synonyms for.
        synonym_method (str): The method used for synonym generation. Supported values include:
            - "wordnet"
            - "gpt4o"
            - "context"
            - "sbert"
            - "lexsub_dropout"
            - "lexsub_concatenation"
            - "lexsub_concatenation_<filter_model>"
        syn_threshold (float, optional): Threshold value used by some methods to filter synonyms
            based on contextual similarity. Defaults to 0.8.

    Returns:
        List[str]: A list of synonym candidates for the input word, with punctuation preserved.

    Raises:
        ValueError: If an unsupported synonym method is provided.
    """
    if not any(char.isalpha() for char in word):
        return []
    # Separate the leading punctuation from the start of the word
    leading_punct, clean_word, trailing_punct = clean_punctuation(word)
    if word != leading_punct + clean_word + trailing_punct:
        print(f"Original: {word}, Leading: {leading_punct}, Clean: {clean_word}, Trailing: {trailing_punct}")
    if clean_word == "" or clean_word.isspace():
        return []
    if clean_word not in text:
        print(f"Word not in text: {clean_word}, test: \n{text}")
        return []
    clean_word = clean_word.strip()
    if any(char.isdigit() for char in clean_word):
        return []

    banned_words = [
        "the", "a", "an", "and", "or", "but", "not", "of", "in", "on", "at", "by",
        "to", "for", "with", "from", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "i", "you", "he", "she", "it", "we",
        "they", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these",
        "those", "there", "here", "who", "whom", "whose", "which", "what", "when", "where",
        "why", "how", "if", "then", "else", "because", "about", "as", "all", "any", "some",
        "each", "every", "no", "none", "one", "two", "other"
    ]
    if clean_word.lower() in banned_words:
        return []

    doc = nlp(text)
    for ent in doc.ents:
        if ent.text == clean_word:
            print(f"Skipping NER: {clean_word} (Type: {ent.label_})")
            return []  # Skip the word if it's a named entity
    # Check if the word is an English word and not a number or symbol
    # if not clean_word.isalpha():  # Skip if the word contains non-alphabetic characters
    #     return []

    # Retrieve synonyms based on the clean word
    if synonym_method == "wordnet":
        synonyms = get_synonyms_wordnet(clean_word)
    elif synonym_method == "gpt4o":
        synonyms = synonym_finder_openai.get_synonyms_in_context(text, clean_word)
    elif synonym_method == "context":
        synonyms = contextual_synonym_finder.get_synonyms_in_context(clean_word, text, syn_threshold)
    elif synonym_method == "sbert":
        synonyms = synonym_sbert.get_synonyms_in_context_sbert(clean_word, text, syn_threshold)
    elif synonym_method == "lexsub_dropout":
        synonyms = Lexical_Substitution.lexsub_dropout.get_synonyms(text, clean_word, int(syn_threshold))
    elif synonym_method == "lexsub_concatenation":
        synonyms = Lexical_Substitution.lexsub_concatenation.get_synonyms(text, clean_word, int(syn_threshold))
    elif synonym_method[:21] == "lexsub_concatenation_":
        synonyms = Lexical_Substitution.lexsub_concatenation.get_synonyms(text, clean_word, int(syn_threshold), filter_model=synonym_method[21:])

    else:
        raise ValueError("Invalid synonym method")

    # Reattach the original punctuation to each synonym
    synonyms_with_punctuation = [leading_punct + synonym + trailing_punct for synonym in synonyms]

    return synonyms_with_punctuation