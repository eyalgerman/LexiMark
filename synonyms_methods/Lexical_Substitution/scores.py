
import torch
import numpy as np
from synonyms_methods.Lexical_Substitution.load_models import load_transformers

'''
Scores implemented per formulas as described in Zhou et al. (2019): https://aclanthology.org/P19-1328/
and extended in Arefyev et al. (2020): https://aclanthology.org/2020.coling-main.107.pdf 

Two types of scores (proposal score and similarity score) are combined to get the final score.
'''

#Calculates the similarity score
def similarity_score(original_output, subst_output, k):
    mask_idx = k
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    weights = torch.div(torch.stack(list(original_output[3])).squeeze().sum(0).sum(0), (12 * 12.0))

    #Calculate the similarittimey score 
    #SIM(x, x'; k) = sum_i^L [ w_{i,k} * cos(h(x_i|x), h(x_i'|x')) ]

    #subst_output = raw_model(sent.reshape(1, sent.shape[0]))
    suma = 0.0
    sent_len = original_output[2][2].shape[1]

    for token_idx in range(sent_len):     
        original_hidden = original_output[2]
        subst_hidden = subst_output[2]

        #Calculate the contextualized representation of the i-th word as a concatenation of RoBERTa's values in its last four layers
        context_original = torch.cat( tuple( [original_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        context_subst = torch.cat( tuple( [subst_hidden[hs_idx][:, token_idx, :] for hs_idx in [1, 2, 3, 4]] ), dim=1)
        suma += weights[mask_idx][token_idx] * cos_sim(context_original, context_subst)

    substitute_validation = suma
    return substitute_validation


#Calculates the proposal score
def proposal_score(original_score, subst_scores):
    # Move both tensors to the CPU
    original_score_cpu = original_score.cpu() if original_score.is_cuda else original_score
    subst_scores_cpu = subst_scores.cpu() if subst_scores.is_cuda else subst_scores

    # subst_scores_cpu = torch.tensor(subst_scores_cpu)
    subst_scores_cpu = subst_scores_cpu.clone().detach()
    return np.log( torch.div(subst_scores_cpu , (1.0 - original_score_cpu)) )


#Calculates the proposal scores, substitute validation scores, and then the final score for each candidate word's fit as a substitution.
def calc_scores(scr, sentences, original_output, original_score, mask_index):
    # Get device from original_score
    device = original_score.device
    # print("Device: " + str(device))
    # Ensure scr and sentences are on the same device as original_score
    scr = torch.tensor(scr).to(device)  # Convert scr to a tensor and move to the correct device
    sentences = sentences.to(device)  # Move sentences to the correct device

    #Get representations of all substitute sentences
    _, _, raw_model = load_transformers()
    subst_output = raw_model(sentences)

    prop_score = proposal_score(original_score, scr).to(device)
    substitute_validation = similarity_score(original_output, subst_output, mask_index).to(device)

    alpha = 0.003
    final_score = substitute_validation + alpha*prop_score
    
    '''
    print("Proposal score: " + str(prop_score))
    print("Subst. validation: " + str(substitute_validation))
    print("Final score for " + str(final_score) + "\n")
    '''
    return final_score, prop_score, substitute_validation
