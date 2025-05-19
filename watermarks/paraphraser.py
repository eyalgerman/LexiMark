import json
import openai
import torch
from tqdm import tqdm
from openai import OpenAI

from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer
)
# from textattack.augmentation import WordNetAugmenter
from sentence_transformers import SentenceTransformer, util

from options import config_instance

# with open("config.json", "r") as file:
#     config = json.load(file)
client = OpenAI(api_key=config_instance.api_key)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# === Rewrite Functions ===
def gpt_rewrite(text, model="gpt-4o-mini-2024-07-18"):
    prompt = f"Rewrite the following text with different wording, preserving the meaning:\n\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    return response.choices[0].message.content.strip()

def t5_paraphrase(text):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    input_text = f"paraphrase: {text} </s>"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    outputs = t5_model.generate(input_ids, num_beams=5, num_return_sequences=1)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def pegasus_paraphrase(text):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
    tokens = pegasus_tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    output = pegasus_model.generate(**tokens, num_beams=10, num_return_sequences=1)
    return pegasus_tokenizer.decode(output[0], skip_special_tokens=True)

# def wordnet_substitute(text):
#     from textattack.augmentation import WordNetAugmenter
#     wordnet_augmenter = WordNetAugmenter()
#     augmented = wordnet_augmenter.augment(text)
#     return augmented[0] if augmented else text


def bart_paraphrase(text):
    from transformers import BartTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    try:
        outputs = model.generate(input_ids, num_beams=5, num_return_sequences=1, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during BART paraphrasing: {e}")
        return text

def apply_model(text, model_name):
    if model_name == "gpt":
        return gpt_rewrite(text)
    elif model_name == "t5":
        return t5_paraphrase(text)
    elif model_name == "pegasus":
        return pegasus_paraphrase(text)
    # elif model_name == "wordnet":
    #     return wordnet_substitute(text)
    elif model_name == "bart":
        return bart_paraphrase(text)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# === Similarity to Root ===
def similarity_to_root(new_text, root_text):
    emb1 = embed_model.encode(new_text, convert_to_tensor=True)
    emb2 = embed_model.encode(root_text, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def paraphrase_text(model_name, threshold=0.6):
    def watermarked_sentences(data):
        new_sentences = []
        for sentence in tqdm(data):
            rewritten = apply_model(sentence, model_name)
            similarity = similarity_to_root(rewritten, sentence)
            if similarity < threshold:
                new_sentences.append(sentence)
            else:
                new_sentences.append(rewritten)
        return new_sentences
    return watermarked_sentences

def converge_rephrasing(text, model_name, threshold=0.99, max_iter=10):
    print(f"Converging rephrasing with model: {model_name}, threshold: {threshold}, max_iter: {max_iter}")
    history = [text]
    for _ in tqdm(range(max_iter)):
        new_text = apply_model(history[-1], model_name)
        sim = similarity_to_root(new_text, history[-1])
        # if sim < threshold:
        #     history.append(new_text)
        if sim > threshold:
            break
    print(f"Converged after {len(history)} iterations.")
    if len(history) == 1:
        print("No significant change in text.")
    else:
        print(f"Final text: {history[-1]}")
        print(f"Similarity to root: {similarity_to_root(history[-1], text)}")
    return history


def rephrase_and_group_by_similarity(text, model_name, max_iter=10, grouping_threshold=0.9):
    """
    Repeatedly rephrases a text and groups the results based on cosine similarity to previous group representatives.

    Args:
        text (str): The original input text.
        model_name (str): Rephrasing model name.
        max_iter (int): Maximum number of rephrasing iterations.
        grouping_threshold (float): Cosine similarity threshold for grouping.

    Returns:
        dict: A dictionary mapping group_id to list of rephrased texts.
    """
    print(f"Rephrasing with model: {model_name}, group-threshold: {grouping_threshold}, max_iter: {max_iter}")

    history = [text]
    groups = {0: [text]}  # First group with original
    group_representatives = [text]

    for _ in tqdm(range(max_iter)):
        new_text = apply_model(history[-1], model_name)

        # Try to assign to a group
        assigned = False
        for i, rep in enumerate(group_representatives):
            group_sim = similarity_to_root(new_text, rep)
            if group_sim >= grouping_threshold:
                groups[i].append(new_text)
                assigned = True
                break

        if not assigned:
            new_group_id = len(groups)
            groups[new_group_id] = [new_text]
            group_representatives.append(new_text)

        history.append(new_text)

    # Print summary
    print(f"\nTotal goups: {len(groups)}")
    for gid, versions in groups.items():
        print(f"\nGroup {gid} ({len(versions)} versions):")
        for i, v in enumerate(versions):
            print(f"  [{i}] {v[:100]}...")

    return groups


# === Example Usage ===
if __name__ == "__main__":
    original = "life, a brilliant and complex life. Her thoughts felt slow and heavy. She was not innocent anymore. Atto and Liza ran forward, hugging her, holding her tight. Liza whispered into Echo\u2019s ear: She would\u2019ve killed you. She would\u2019ve killed us.\u2019 I feel something I\u2019ve never felt before\u2019 Her sentence drifted into silence, searching for the word. I feel shame.\u2019 With the body of Cho across his back, Eitan approached, his colony behind him. It was not an attack but a funeral march. Echo stepped forward, in front of her family, ready to defend them yet at the same time sharing in the grief of those opposing her, realizing that she remained connected to their thoughts, catching fragments of their pain. Eitan\u2019s voice was different, no longer brassy and unbreakable. Your people have until the first day of winter to leave this place. After the sun sets, we will kill anyone who remains.\u2019 Echo said: I didn\u2019t intend to kill her.\u2019 You could\u2019ve been one of us. Now you belong with them.\u2019 With that said, the colony of cold creatures turned in unison, heading in procession out of the ruins of the city, towards the mountains and glaciers they called home. EPILOGUE TWO MONTHS LATER TRANSANTARCTICA FREEWAY 15 MARCH 2044 AHEAD OF THE APPROACHING WINTER a second Exodus was underway a 1,800mile trek across the continent, a perilous journey made famous by daring explorers seeking a place in the record books. What was once adventure was now necessity. The people of McMurdo were abandoning their capital city; for the children born there it was the only home they\u2019d ever known. Across the ice shelf a caravan of a million McMurdo City refugees snaked across the snow, like a hairline crack in a white porcelain plate. Having made an extraordinary journey to this continent, another journey was being demanded of them, this time to the Peninsula, trying to complete the journey before winter arrived. Many of McMurdo\u2019s leaders drowned during the sinking of the flagship and a new leadership had taken charge, made up of former generals and admirals from armies around the world. Under their stewardship, the evacuation of McMurdo City had been wellorganized and calm, unloading all the supplies from the ships, barely enough to see them through the journey let alone the long dark winter ahead. Many of the snow vehicles had been destroyed in the fire. As for the packs of once devoted huskies, all of them had joined the colony of cold creatures, including Yotam\u2019s dog Copper. Not a single dog remained, as if they understood that this continent had new masters now. As the last refugees set off from the scorched remains of McMurdo City, they fired a hundred flares into a clear blue sky, representing the end of this base where people had lived for one hundred years. The three Survivor Towns responded to the news of the uprising with resilience and generosity, promising to welcome the new arrivals with the same love and compassion as if they were family. But there was no hiding from"
    # models = ["bart", "gpt", "t5", "pegasus"]
    # for model_name in models:
    #     rewritten = apply_model(original, model_name)
    #     score = similarity_to_root(rewritten, original)
    #     print(f"Model: {model_name}\nRewritten: {rewritten}\nSimilarity: {score:.4f}\n{'-'*40}")
    history = converge_rephrasing(original, "gpt", threshold=0.98, max_iter=100)
    groups = rephrase_and_group_by_similarity(original, "bart", max_iter=100, grouping_threshold=0.95)


