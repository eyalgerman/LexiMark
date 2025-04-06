import json
import os
import openai
from openai import OpenAI
from pydantic import BaseModel

from options import config_instance


class SynonymFinder(BaseModel):
    has_synonyms: bool
    synonyms: list[str]

# with open("config.json", "r") as file:
#     config = json.load(file)

API_KEY = config_instance.api_key

def get_synonyms_in_context(sentence, word_to_replace, api_key=API_KEY):
    """
    Get a list of synonyms for a word in the context of a given sentence using OpenAI's GPT.

    Args:
    sentence (str): The sentence containing the word.
    word_to_replace (str): The word to find synonyms for.
    api_key (str): Your OpenAI API key.

    Returns:
    list: A list of synonyms for the word.
    """
    # openai.api_key = api_key
    try:
        client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

        # Generate a prompt to ask for synonyms considering the context of the sentence
        # prompt = f"The word '{word_to_replace}' appears in the sentence: '{sentence}'. Provide synonyms for '{word_to_replace}' that fit this context:"
        prompt = f"Provide synonyms for '{word_to_replace}' that fit this context: {sentence}."

        # Call the API to get synonyms
        # response = client.chat.completions.create(
        response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful synonyms finder. You will be given a word and a sentence. You need to provide synonyms for the word that fit the context of the sentence. "
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # model="gpt-3.5-turbo",
        model="gpt-4o-2024-08-06",
        response_format=SynonymFinder
    )

        # synonyms = response.choices[0].message.content.split(",") if response.choices else []
        # synonyms = [synonym.strip() for synonym in synonyms]
        synonyms = response.choices[0].message.parsed
        # print("Response:", synonyms)
        # print("Synonyms:", synonyms.synonyms)
        if synonyms.has_synonyms:
            return set(synonyms.synonyms)
        else:
            return set()
    except Exception as e:
        print(f"Error: {e}")
        print(f"Text: {sentence}")
        print(f"Target: {word_to_replace}")
        return set()


if __name__ == '__main__':
    sentence = "He wasnt thinking big because of some personal vanity or youthful ego  this was about the survival of entire villages, families and generations."
    word_to_replace = "vanity"  # vanity
    synonyms = get_synonyms_in_context(sentence, word_to_replace)
    print(synonyms)
