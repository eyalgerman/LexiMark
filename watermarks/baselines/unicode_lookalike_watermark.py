import random

# Dictionary mapping original characters to their Unicode lookalikes
replacement_dict = {
    "a": "\u0430", "c": "\u03f2", "e": "\u0435", "g": "\u0261",
    "i": "\u0456", "j": "\u03f3", "o": "\u03bf", "p": "\u0440",
    "s": "\u0455", "x": "\u0445", "y": "\u0443", "A": "\u0391",
    "B": "\u0392", "C": "\u03f9", "E": "\u0395", "H": "\u0397",
    "I": "\u0399", "J": "\u0408", "K": "\u039a", "M": "\u039c",
    "N": "\u039d", "O": "\u039f", "P": "\u03a1", "S": "\u0405",
    "T": "\u03a4", "X": "\u03a7", "Y": "\u03a5", "Z": "\u0396"
}


# function to replace characters in a string using the replacement dictionary in Global perturbation
def global_unicode_watermark(seed=42):
    # Generate a random binary vector for substitution based on dictionary keys
    random.seed(seed)  # Ensuring reproducibility
    substitution_vector = [random.choice([0, 1]) for _ in range(len(replacement_dict))]

    # Create a substitution plan according to the binary vector
    substitution_plan = {
        list(replacement_dict.keys())[i]: replacement_dict[list(replacement_dict.keys())[i]] if substitution_vector[
            i] else list(replacement_dict.keys())[i]
        for i in range(len(replacement_dict))
    }

    def watermarked_sentences(data):
        # Replace characters in all documents according to the substitution plan
        new_data = []
        for text in data:
            new_text = ''.join(substitution_plan.get(char, char) for char in text)
            new_data.append(new_text)

        return new_data

    return watermarked_sentences


# Function to replace characters in a string using the replacement dictionary in Word-level perturbation
def word_level_unicode_watermark(seed=42):
    random.seed(seed)  # Ensuring reproducibility

    def watermarked_sentences(data):
        # Create a random mapping for each unique word
        unique_words = set(word for text in data for word in text.split())
        word_map = {}
        for word in unique_words:
            # Generate a binary vector for each word
            substitution_vector = [random.choice([0, 1]) for _ in range(len(word))]
            # Create a substitution plan for each word
            new_word = ''.join(replacement_dict.get(char, char) if substitution_vector[i] else char
                               for i, char in enumerate(word))
            word_map[word] = new_word

        # Apply the mapping to each word in each document
        new_data = []
        for text in data:
            new_text = ' '.join(word_map.get(word, word) for word in text.split())
            new_data.append(new_text)
        return new_data

    return watermarked_sentences


# Function to replace characters in a string using the replacement dictionary
def replace_characters(text, percentage=1):
    characters = list(text)
    num_to_replace = int(len(characters) * (percentage))
    indices_to_replace = random.sample(range(len(characters)), num_to_replace)

    for i in indices_to_replace:
        if characters[i] in replacement_dict:
            characters[i] = replacement_dict[characters[i]]

    return ''.join(characters)


def add_watermark(percent=0.5):
    # Load the GPT2 tokenizer
    def watermarked_sentences(data):
        new_sentences = []
        # for sentences in data:
        for text in data:
            # Append the watermark to the text
            watermarked_text = replace_characters(text, percentage=percent)
            new_sentences.append(watermarked_text)
        return new_sentences

    return watermarked_sentences


def highlight_changes(watermarked_data):
    # Function to compare original and watermarked texts to highlight replaced characters
    def compare_texts(watermarked):
        highlighted_text = []
        for char in watermarked:
            if char in replacement_dict.values():  # Check if the character is a replaced value
                # Wrap replaced characters in parentheses
                highlighted_text.append(f"({char})")
            else:
                highlighted_text.append(char)
        return ''.join(highlighted_text)

    # Process each text in the watermarked data
    highlighted_data = []
    for watermarked in watermarked_data:
        highlighted = compare_texts(watermarked)
        highlighted_data.append(highlighted)

    return highlighted_data


if __name__ == "__main__":
    # Example usage
    watermarker = global_unicode_watermark()  # Creating the watermarking function with a default seed
    data = ["example text", "More Text Data"]
    print(data)
    watermarked_data = watermarker(data)
    print(highlight_changes(watermarked_data))

    # Example usage
    watermarker = word_level_unicode_watermark()  # Creating the watermarking function with a default seed
    data = ["example text", "More text Data"]
    watermarked_data = watermarker(data)
    print(highlight_changes(watermarked_data))
