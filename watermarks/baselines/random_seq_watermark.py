import random
from transformers import GPT2Tokenizer


def generate_random_sequence(tokenizer, sequence_length):
    """Generate a random sequence of characters sampled from the first 100 tokens of GPT2 tokenizer."""
    # Get the first 100 tokens from the GPT2 tokenizer's vocabulary
    first_100_tokens = list(tokenizer.get_vocab().keys())[:100]

    # Generate a random sequence from these tokens
    random_sequence = ''.join(random.choices(first_100_tokens, k=sequence_length))
    return random_sequence


def add_watermark(seed=42, noise_length=10, percent=1):
    # Load the GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    random.seed(seed)

    def watermarked_sentences(data):
        new_sentences = []
        watermark = generate_random_sequence(tokenizer, noise_length)
        num_records = len(data)
        num_to_watermark = int(num_records * percent)

        # Randomly select indices to watermark
        indices_to_watermark = random.sample(range(num_records), num_to_watermark)

        for i, text in enumerate(data):
            if i in indices_to_watermark:
                # Append the watermark to the text
                watermarked_text = text + " " + watermark
            else:
                watermarked_text = text
            new_sentences.append(watermarked_text)

        return new_sentences
    return watermarked_sentences


def main():
    # Load the GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Example input text
    text = "This is an example text."

    # Define the length of the random sequence watermark
    sequence_length = 10  # Adjust as needed based on experiments

    # Generate the watermark based on the GPT2 tokenizer
    watermark = generate_random_sequence(tokenizer, sequence_length)

    # Append the watermark to the text
    watermarked_text = text + " " + watermark

    # Output the watermarked text
    print("Original text:", text)
    print("Watermarked text:", watermarked_text)


if __name__ == "__main__":
    main()
