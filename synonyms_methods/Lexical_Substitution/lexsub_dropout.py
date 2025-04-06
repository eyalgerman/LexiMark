from xmlrpc.client import Error

import torch
import string
import nltk
from nltk import pos_tag

nltk.download('stopwords')
import time
import numpy as np

from filter import filter_words
from scores import calc_scores
from load_models import load_transformers
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Discarded redundant search")

tokenizer, lm_model, raw_model = load_transformers()

'''
First approach to lexical substitution, applying dropout to randomly selected weight indices of the target word embedding.
This has an effect of the model taking into account some semantic meaning of the word, but not overfitting on it, so it takes into 
account the context of the sentence as well.

Implemented from scratch from:
Zhou et al. (2019): https://aclanthology.org/P19-1328/
'''

def truncate_sentence(sentence, target):
    # Tokenize the sentence and locate the target word
    input_ids = tokenizer.encode(" " + sentence)
    target_token_id = tokenizer.encode(" " + target)[1]

    if target_token_id not in input_ids:
        raise ValueError(f"Target word '{target}' not found in the sentence.")

    mask_position = input_ids.index(target_token_id)

    # Define the max length for RoBERTa (e.g., 514 tokens)
    max_length = 514

    # Truncate sentence by keeping tokens around the target word
    half_max_length = max_length // 2
    start_idx = max(0, mask_position - half_max_length)
    end_idx = min(len(input_ids), mask_position + half_max_length)

    truncated_input_ids = input_ids[start_idx:end_idx]
    return truncated_input_ids


def lexsub_dropout(sentence, target):
    sentence = sentence.replace('-', ' ')
    table = str.maketrans(dict.fromkeys(string.punctuation)) 

    #Remove unnecessary punctuation from the sentence (such as: "GET *free food *coupons!!")
    split_sent = nltk.word_tokenize(sentence)
    split_sent = list(map(lambda wrd : wrd.translate(table) if wrd not in string.punctuation else wrd, split_sent))
    original_sent = ' '.join(split_sent)

    # Check if a GPU is available and if so, use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Device: {device}")

    try:
        # Get input IDs from tokenizer and move them to the correct device
        input_ids = tokenizer.encode(" " + original_sent, return_tensors="pt").to(device)

        # Get RoBERTa word embeddings for words in the sentence (ensure output is on the same device)
        original_output = raw_model(input_ids)
        # original_output = raw_model(tokenizer.encode(" "+original_sent, return_tensors="pt").to(device))
        # original_output = raw_model(torch.tensor([truncate_sentence(original_sent, target)]))
        inputs_embeds = original_output[2][1].to(device)
        # The target word to substitute
        target_token_id = tokenizer.encode(" " + target)[1]
        input_ids = tokenizer.encode(" " + original_sent)
        mask_position = input_ids.index(target_token_id)
    except Exception as e:
        print(f"Error: {e}")
        print(f'Text: {original_sent}')
        print(f'Target: {target}')
        # raise e
        return []

    #Set a percentage of randomly selected embedding weights of the target word to 0.
    embedding_dim = 768
    dropout_percent = 0.3
    dropout_amount = round(dropout_percent*embedding_dim)

    #Start timing the experiment.
    start = time.time()

    #Run multiple experiments and then take average because of stochastic nature of choosing indices to dropout (sometimes the predictions are gibberish)
    all_scores = dict()
    all_counts = dict()
    num_iterations = 5
    for it in range(num_iterations):
        #Choose the weight indices to drop out.
        dropout_indices = np.random.choice(embedding_dim, dropout_amount, replace=False)
        inputs_embeds[0, mask_position, dropout_indices] = 0

        #Pass the embeddings where masked word's embedding is partially droppped out to the model 
        with torch.no_grad():
                output = lm_model(inputs_embeds=inputs_embeds.to(device))
        logits = output[0].squeeze()

        #Get top guesses
        mask_logits = logits[mask_position]
        top_tokens = torch.topk(mask_logits, k=16, dim=0)[1]
        scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
        words = [tokenizer.decode(i.item()).strip() for i in top_tokens]
        
        words, scores, top_tokens = filter_words(target, words, scores, top_tokens)
        assert len(words) == len(scores)

        if len(words) == 0: 
            continue

        #Calculate proposal scores, substitute validation scores, and final scores
        original_score = torch.softmax(mask_logits, dim=0)[target_token_id]
        sentences = list()
        split_sent = nltk.word_tokenize(sentence)

        for i in range(len(words)):
            subst_word = top_tokens[i]
            input_ids[mask_position] = int(subst_word)
            sentences.append(list(input_ids))

        sentences = torch.tensor(sentences).to(device)
       
        finals, props, subval = calc_scores(scores, sentences, original_output, original_score, mask_position)
        finals = map(lambda f : float(f), finals)
        props = map(lambda f : float(f), props)
        subval = map(lambda f : float(f), subval)

        if target in words:
            words.remove(target)

        #Update total scores and counts in the dictionary
        res = dict(zip(words, finals))
        for w, s in res.items():
            all_scores[w] = all_scores[w] + s if w in all_scores.keys() else s
            all_counts[w] = all_counts[w] + 1 if w in all_counts.keys() else 1

    #Get the average of accumulated scores.
    for w, s in all_scores.items():
        all_scores[w] = s / all_counts[w]
    words, finals = list(all_scores.keys()), list(all_scores.values())


    #Sort the found substitutes by scores and print them out.
    x = dict(zip(words, finals))
    finish = list(sorted(x.items(), key=lambda item: item[1], reverse=True))[:15]
    # print(["({0}: {1:0.8f})".format(k, v) for k,v in finish])
    # print("Elapsed time: ", time.time() - start, "\n")
    return finish


def get_synonyms(sentence, target, num_synonyms=10):
    try:
        synonyms = lexsub_dropout(sentence, target)
    except Error as e:
        print(f"Error: {e}")
        # print(f'Stack trace: {e.__traceback__}')
        print(f'Text: {sentence}')
        print(f'Target: {target}')
        return []
    if not synonyms:
        return []
    synonyms = [syn[0] for syn in synonyms]
    return synonyms[:num_synonyms]


if __name__ == '__main__':
    #Example usage
    sentence = "slowly across the ring, a feral smile on her face, the cheers of the crowd encouraging her to take her time with an obviously weaker opponent."
    target = "crowd"
    lexsub_dropout(sentence, target)
    print(get_synonyms(sentence, target, 10))
    print("\n")