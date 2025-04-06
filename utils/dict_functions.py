
def read_dict_from_file(filename):
    """
    Reads a dictionary from a file where each line is formatted as 'key: value'.

    Args:
        filename (str): Path to the file containing the dictionary entries.

    Returns:
        dict: A dictionary with keys and values parsed from the file.
              Returns None if the file is not found or an error occurs.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during parsing.
    """
    try:
        replacement_dict = {}
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()  # Remove any trailing newline or spaces
                if line:
                    original_word, replaced_word = line.split(': ')
                    replacement_dict[original_word] = replaced_word
        print(f"Dictionary loaded successfully from {filename}.")
        return replacement_dict
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")


def write_dict_to_file(replacement_dict, filename):
    """
    Writes a dictionary to a file, storing each entry as 'key: value' on a new line.

    Args:
        replacement_dict (dict): Dictionary to write to the file.
        filename (str): Destination file path where the dictionary will be saved.

    Raises:
        Exception: If an error occurs during file writing.
    """
    try:
        with open(filename, 'w') as file:
            for original_word, replaced_word in replacement_dict.items():
                file.write(f"{original_word}: {replaced_word}\n")
        print(f"Dictionary saved successfully to {filename}.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {str(e)}")

