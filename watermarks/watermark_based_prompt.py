from openai import OpenAI
from tqdm import tqdm

from synonyms_methods.synonym_finder_openai import API_KEY

prompt1 ="Given the following paragraph, replace certain words with slightly higher-entropy synonyms (words that are slightly less common or a bit more complex) while preserving the overall semantic meaning and tone of the paragraph. Ensure that the paragraph remains grammatically correct and retains its original intent. Do not replace basic words such as prepositions, conjunctions, or pronouns (e.g., \"the\", \"and\", \"he\", \"she\", \"of\")."

prompt2 = """
Given the following paragraph, replace certain words with slightly higher-entropy synonyms (words that are slightly less common or a bit more complex) while preserving the overall semantic meaning and tone of the paragraph. Ensure that the paragraph remains grammatically correct and retains its original intent. Do not replace basic words such as prepositions, conjunctions, or pronouns (e.g., "the", "and", "he", "she", "of"). Make sure to change a maximum of $K$ words per sentence, and if there are not enough suitable words, then change less.
"""

def update_text_openai_paragraph(text, prompt):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=API_KEY,
    )
    # Use the OpenAI API to generate text based on the prompt
    response = client.chat.completions.create(
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"""Original paragraph: {text}
    
                Modified paragraph:""",
            }
        ],
        model="gpt-4o-2024-08-06"
    )
    # Extract the generated text from the response
    generated_text = response.choices[0].message.content
    # Combine the original text and the generated text
    updated_text = generated_text
    return updated_text


def add_watermark(prompt, k=5):
    prompt = prompt.replace("$K$", str(k))
    def watermarked_sentences(data):
        new_sentences = []
        for text in tqdm(data):
            try:
                new_text = update_text_openai_paragraph(text, prompt)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Text: {text}")
                new_text = text
            new_sentences.append(new_text)

        return new_sentences
    return watermarked_sentences


if __name__ == "__main__":
    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    print("Original text:", text)
    # Create the watermarking function
    watermark_function = add_watermark(prompt2, 5)
    # Watermark the text
    watermarked_text = watermark_function([text])
    # Print the watermarked text
    print("Watermark text:", watermarked_text[0])
    print("Watermarking completed.")