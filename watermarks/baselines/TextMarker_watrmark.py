import random


def generate_backdoored_text(data, trigger, trigger_type, location, poisoning_rate):
    """Inserts different types of triggers to text data for backdooring at specified locations."""
    backdoored_data = []
    num_poisoned = int(len(data) * poisoning_rate)  # Calculate number of samples to poison
    poisoned_indices = random.sample(range(len(data)), num_poisoned)  # Random indices to poison

    for index, text in enumerate(data):
        if index in poisoned_indices:
            if trigger_type == "character":
                text = character_level_trigger(location, text, trigger)

            elif trigger_type == "word":
                text = word_level_trigger(location, text, trigger)

            elif trigger_type == "sentence":
                text = sentence_level_trigger(location, text, trigger)

        backdoored_data.append(text)

    return backdoored_data


def sentence_level_trigger(location, text, trigger):
    if location == "initial":
        text = trigger + " " + text
    elif location == "middle":
        sentences = text.split('. ')
        mid_index = len(sentences) // 2
        text = " ".join(sentences[:mid_index] + [trigger] + sentences[mid_index:])
    elif location == "end":
        text = text + " " + trigger
    elif location == "random":
        sentences = text.split('. ')
        rand_index = random.randint(0, len(sentences))
        text = " ".join(sentences[:rand_index] + [trigger] + sentences[rand_index:])
    return text


def word_level_trigger(location, text, trigger):
    if location == "initial":
        text = trigger + ". " + text
    elif location == "middle":
        words = text.split()
        mid_index = len(words) // 2
        text = " ".join(words[:mid_index] + [trigger] + words[mid_index:])
    elif location == "end":
        text = text + ". " + trigger
    elif location == "random":
        words = text.split()
        rand_index = random.randint(0, len(words))
        text = ". ".join(words[:rand_index] + [trigger] + words[rand_index:])
    return text


def character_level_trigger(location, text, trigger):
    words = text.split()
    if location == "initial":
        modified_words = [trigger + word for word in words]
    elif location == "middle":
        modified_words = [word[:len(word) // 2] + trigger + word[len(word) // 2:] for word in words]
    elif location == "end":
        modified_words = [word + trigger for word in words]
    elif location == "random":
        # Corrected to calculate a single random position for the trigger insertion within each word
        modified_words = [word[:rand_pos] + trigger + word[rand_pos:] for word in words
                          for rand_pos in [random.randint(0, len(word))]]  # Use list comprehension to bind rand_pos
    text = " ".join(modified_words)
    return text


def watermark_data(trigger, trigger_type, location, poisoning_rate):

    def watermarked_sentences(data):
        new_data = generate_backdoored_text(data, trigger, trigger_type, location, poisoning_rate)
        return new_data

    return watermarked_sentences


if __name__ == "__main__":
    # Example usage of generate_backdoored_text function
    data = ["This is a sample sentence. I like Avocado.", "Another example sentence. He like Banana."]
    trigger = "Less is more."
    trigger_type = "sentence"
    location = "random"
    poisoning_rate = 1

    backdoored_data = generate_backdoored_text(data, trigger, trigger_type, location, poisoning_rate)
    for original, backdoored in zip(data, backdoored_data):
        print(f"Original: {original}")
        print(f"Backdoored: {backdoored}")
        print("\n")