import torch
import string
import nltk

from .. import contextual_synonym_finder
from .. import synonym_sbert

nltk.download('averaged_perceptron_tagger')
import time
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

from synonyms_methods.Lexical_Substitution.nyms import get_nyms
from synonyms_methods.Lexical_Substitution.filter import filter_words
from synonyms_methods.Lexical_Substitution.scores import calc_scores
from synonyms_methods.Lexical_Substitution.load_models import load_transformers

tokenizer, lm_model, raw_model = load_transformers()

'''
Second approach to lexical substitution, concatenating the original sentence to the masked sentence and passing it like that to the transformer 
to predict the target word. Example input: "Best offers of the season! [SEP] Best <mask> of the season!"

Implemented from scratch from:
Qiang et al. (2019) [https://arxiv.org/abs/1907.06226]
'''

def lexsub_concatenation(sentence, target):
    start = time.time()

    #Removes the unnecessary punctuation from the input sentence.
    sentence = sentence.replace('-', ' ')
    table = str.maketrans(dict.fromkeys(string.punctuation)) 

    split_sent = nltk.word_tokenize(sentence)
    split_sent = list(map(lambda wrd : wrd.translate(table) if wrd not in string.punctuation else wrd, split_sent))
    original_sent = ' '.join(split_sent)

    #Masks the target word in the original sentence.
    masked_sent = ' '.join(split_sent)
    masked_sent = masked_sent.replace(target, tokenizer.mask_token, 1)

    # Check if a GPU is available and if so, use it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Device: {device}")

    try:
        #Get the input token IDs of the input consisting of: the original sentence + separator + the masked sentence.
        input_ids = tokenizer.encode(" "+original_sent, " "+masked_sent, add_special_tokens=True)
        max_model_length = tokenizer.model_max_length
        if len(input_ids) > max_model_length:
            input_ids = input_ids[:max_model_length]
        # Convert to tensor and ensure the correct type (LongTensor for token IDs)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        # masked_position = input_ids.index(tokenizer.mask_token_id)

        if tokenizer.mask_token_id in input_ids:
            masked_position = input_ids.index(tokenizer.mask_token_id)
        else:
            print("Mask token not found in input_ids.")
            return []
        if torch.isnan(input_ids_tensor).any() or torch.isinf(input_ids_tensor).any():
            print("NaN or Inf in input_ids_tensor")
            return []
        # original_output = raw_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(device))
        with torch.no_grad():
            original_output = raw_model(input_ids_tensor)
            output = lm_model(input_ids_tensor)
        # original_output = raw_model(input_ids_tensor)
    except Exception as e:
        print(f"Error1: {e}")
        print(f'Text: {original_sent}')
        print(f'Target: {target}')
        if "CUDA error:" in str(e):
            print("Input_ids: ", input_ids)
            raise e
        # raise e
        return []

    #Get the predictions of the Masked LM transformer.
    # with torch.no_grad():
    #     # output = lm_model(torch.tensor(input_ids).reshape(1, len(input_ids)).to(device))
    #     output = lm_model(input_ids_tensor)
    
    logits = output[0].squeeze()

    #Get top guesses: their token IDs, scores, and words.
    mask_logits = logits[masked_position].squeeze()
    top_tokens = torch.topk(mask_logits, k=20, dim=0)[1]
    scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
    words = [tokenizer.decode(i.item()).strip() for i in top_tokens]
    
    words, scores, top_tokens = filter_words(target, words, scores, top_tokens)
    assert len(words) == len(scores)

    if len(words) == 0: 
        return

    # print("GUESSES: ", words)

    #Calculate proposal scores, substitute validation scores, and final scores
    original_score = torch.softmax(mask_logits, dim=0)[masked_position]
    sentences = list()

    for i in range(len(words)):
        subst_word = top_tokens[i]
        input_ids[masked_position] = int(subst_word)
        sentences.append(list(input_ids))

    # Padding logic: Find the max length of all sentences and pad shorter sentences
    max_length = max(len(ids) for ids in sentences)
    pad_token_id = tokenizer.pad_token_id

    # Pad sentences to the max_length
    padded_sentences = [ids + [pad_token_id] * (max_length - len(ids)) for ids in sentences]

    #print([tokenizer.decode(s) for s in sentences])
    torch_sentences = torch.tensor(padded_sentences).to(device)
    # torch_sentences = torch.tensor(sentences).to(device)

    finals, props, subval = calc_scores(scores, torch_sentences, original_output, original_score, masked_position)
    finals = map(lambda f : float(f), finals)
    props = map(lambda f : float(f), props)
    subval = map(lambda f : float(f), subval)

    if target in words:
        words = [w for w in words if w not in [target, target.capitalize(), target.upper()]] 

    zipped = dict(zip(words, finals))
    lemmatizer = WordNetLemmatizer()

    try:
        ###Remove plurals, wrong verb tenses, duplicate forms, etc.############
        original_pos = nltk.pos_tag(nltk.word_tokenize(original_sent))
        target_index = split_sent.index(target)
        assert original_pos[target_index][0] == target
        original_tag = original_pos[target_index][1]
        for i in range(len(words)):
            cand = words[i]
            if cand not in zipped:
                continue

            sent = original_sent
            masked_sent = sent.replace(target, cand, 1)

            new_pos = nltk.pos_tag(nltk.word_tokenize(masked_sent))
            new_tag = new_pos[target_index][1]

            # If the word appears in both singular and plural in the candidate list, remove one of them.
            if new_tag.startswith('N') and not new_tag.endswith('S'):
                if (cand + 's' in words or cand + 'es' in words) in words and original_tag.endswith('S'):
                    del zipped[cand]
                    continue
            elif new_tag.startswith('N') and new_tag.endswith('S'):
                if (cand[:-1] in words or cand[:-2] in words) and not original_tag.endswith('S'):
                    del zipped[cand]
                    continue

            # If multiple forms of the original word appear in the candidate list, remove them (e.g. begin, begins, began, begun...)
            wntags = ['a', 'r', 'n', 'v']
            for tag in wntags:
                if lemmatizer.lemmatize(cand, tag) == lemmatizer.lemmatize(target, tag):
                    del zipped[cand]
                    break

    except Exception as e:
        print(f"Error2: {e}")
        print(f'Text: {original_sent}')
        print(f'Target: {target}')
        # return []

    #################

    #Print sorted candidate words.
    zipped = dict(zipped)
    finish = list(sorted(zipped.items(), key=lambda item: item[1], reverse=True))[:15]
    # print("CANDIDATES:", ["({0}: {1:0.8f})".format(k, v) for k,v in finish])

    #Print any relations between candidates and the original word.
    words = zipped.keys()
    nyms = get_nyms(target)
    nym_output = list()
    for cand in words:
        for k, v in nyms.items():
            if lemmatizer.lemmatize(cand.lower()) in v:
                nym_output.append((cand, k[:-1]))

    return finish


def lexsub_concatenation_v2(sentence, target):
    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model.to(device)
    raw_model.to(device)

    sentence = sentence.replace('-', ' ')
    table = str.maketrans('', '', string.punctuation)
    tokens = nltk.word_tokenize(sentence)
    tokens = [token.translate(table) for token in tokens if token not in string.punctuation]
    original_sent = ' '.join(tokens)
    masked_sent = original_sent.replace(target, tokenizer.mask_token, 1)

    try:
        input_ids = tokenizer.encode(original_sent, masked_sent, add_special_tokens=True)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        masked_position = input_ids.index(tokenizer.mask_token_id)

        with torch.no_grad():
            original_output = raw_model(input_ids_tensor)
            output = lm_model(input_ids_tensor)

        logits = output.logits.squeeze(0)
        mask_logits = logits[masked_position]
        top_tokens = torch.topk(mask_logits, k=20, dim=0).indices
        scores = torch.softmax(mask_logits, dim=0)[top_tokens].tolist()
        words = [tokenizer.decode([i.item()]).strip() for i in top_tokens]

        # Prepare sentences for scoring
        subst_sentences = [list(input_ids) for _ in range(len(top_tokens))]
        for i, token in enumerate(top_tokens):
            subst_sentences[i][masked_position] = token.item()

        subst_sentences_tensor = torch.tensor(subst_sentences, dtype=torch.long).to(device)
        finals, props, subvals = calc_scores(scores, subst_sentences_tensor, original_output, logits[masked_position],
                                             masked_position)

    except Exception as e:
        print(f"Error during processing: {e}")
        return []

    # Advanced filtering based on linguistic features
    lemmatizer = WordNetLemmatizer()
    original_pos = nltk.pos_tag(nltk.ord_tokenize(original_sent))
    target_index = tokens.index(target)
    zipped = dict(zip(words, finals))

    for word in list(zipped.keys()):
        # Filter based on lemmatization to avoid same lexical forms
        if lemmatizer.lemmatize(word, 'v') == lemmatizer.lemmatize(target, 'v'):
            del zipped[word]
            continue

        # Part-of-speech based filtering
        new_sentence = original_sent.replace(target, word, 1)
        new_pos = nltk.pos_tag(nltk.word_tokenize(new_sentence))[target_index][1]
        original_tag = original_pos[target_index][1]
        if new_pos != original_tag:
            del zipped[word]

    sorted_results = sorted(zipped.items(), key=lambda item: item[1], reverse=True)[:15]
    print("Elapsed time: ", time.time() - start)
    return sorted_results


def get_synonyms(sentence, target, num_synonyms=10, filter_model=None):
    try:
        synonyms = lexsub_concatenation(sentence, target)
        # synonyms2 = lexsub_concatenation_v2(sentence, target)
        # print(f"Synonyms 1: {synonyms}")
        # print(f"Synonyms 2: {synonyms2}")
    except Exception as e:
        print(f"Error: {e}")
        print(f'Text: {sentence}')
        print(f'Target: {target}')
        raise e
        return []
    if not synonyms:
        return []
    synonyms = [syn[0] for syn in synonyms]
    if filter_model is None:
        return synonyms[:num_synonyms]
    elif filter_model[:5] == "sbert":
        th = float(filter_model[5:])
        # Filter function to apply 'check_synonym_word_sbert' on all elements in the 'synonyms' list
        synonyms = list(
            filter(lambda element: synonym_sbert.check_synonym_word_sbert(sentence, target, element, threshold=th), synonyms))
    elif filter_model[:4] == "bert":
        th = float(filter_model[4:])
        synonyms = list(
            filter(lambda element: contextual_synonym_finder.check_synonym_word_bert(sentence, target, element, threshold=0.8), synonyms))

    return synonyms[:num_synonyms]


if __name__ == '__main__':
    #Example usage
    sentence = "slowly across the ring, a feral smile on her face, the cheers of the crowd encouraging her to take her time with an obviously weaker opponent."
    target = "crowd"
    lexsub_concatenation(sentence, target)
    print(get_synonyms(sentence, target, 10))
    print("\n")
    # print(get_synonyms(sentence, target, 10, filter_model="bert0.8"))
