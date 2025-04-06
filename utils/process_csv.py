import csv
import json
import sys

# Increase the field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

def csv_to_jsonl(member_csv_path, non_member_csv_path, output_jsonl_path):
    """
    Converts member and non-member CSV files into a single JSONL file.

    Each row in the resulting JSONL file contains a dictionary with 'input' (text)
    and a binary 'label' (1 for member, 0 for non-member).

    Args:
        member_csv_path (str): Path to the member CSV file.
        non_member_csv_path (str): Path to the non-member CSV file.
        output_jsonl_path (str): Output path for the generated JSONL file.
    """
    data = []

    # Read member CSV file
    with open(member_csv_path, 'r', encoding='utf-8') as member_file:
        reader = csv.DictReader(member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 1})

    # Read non-member CSV file
    with open(non_member_csv_path, 'r', encoding='utf-8') as non_member_file:
        reader = csv.DictReader(non_member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 0})

    # Write to JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')


def csv_to_jsonl_pile(member_csv_path, non_member_csv_path, output_jsonl_path):
    """
    Converts member and non-member CSV files into a single JSONL file, with an additional `data_name` field.

    Each row in the resulting JSONL file contains 'input' (text), 'label' (1 for member, 0 for non-member),
    and 'data_name' for traceability.

    Args:
        member_csv_path (str): Path to the member CSV file.
        non_member_csv_path (str): Path to the non-member CSV file.
        output_jsonl_path (str): Output path for the generated JSONL file.
    """
    data = []

    # Read member CSV file
    with open(member_csv_path, 'r', encoding='utf-8') as member_file:
        reader = csv.DictReader(member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 1, 'data_name': row['data_name']})

    # Read non-member CSV file
    with open(non_member_csv_path, 'r', encoding='utf-8') as non_member_file:
        reader = csv.DictReader(non_member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 0, 'data_name': row['data_name']})

    # Write to JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')


def write_sentences_to_csv(sentences, file_name):
    """
    Writes a list of text sentences to a CSV file with a single 'text' column.

    Args:
        sentences (List[str]): A list of text strings to write.
        file_name (str): Path to the output CSV file.
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text"])  # Write the header
        for sentence in sentences:
            # Check if the sentence is not None or empty
            if sentence and sentence.strip():
                writer.writerow([sentence])  # Write each sentence in a new row


def write_sentences_to_csv_PILE(datasets, file_name):
    """
    Writes grouped text data to a CSV file with 'text' and 'data_name' columns.

    Args:
        datasets (List[Tuple[str, List[str]]]): A list of (data_name, sentences) pairs.
        file_name (str): Path to the output CSV file.
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "data_name"])  # Write the header
        for data_name, sentences in datasets:
            for sentence in sentences:
                # Check if the sentence is not None or empty
                if sentence and sentence.strip():
                    writer.writerow([sentence, data_name])  # Write each sentence in a new row


def keep_first_column(input_file):
    """
    Extracts the first column from a CSV file and writes it to a new file.
    Only non-empty rows with a non-blank first column are written to the output.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        str: Path to the new CSV file containing only the first column.
    """
    output_file = input_file.replace('.csv', '_first_column.csv')
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)

        # Open the output file in write mode
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)

            for row in reader:
                # Only process rows where the first column is not None or empty
                if row and row[0].strip():  # Check if the row exists and the first column is not empty or just whitespace
                    writer.writerow([row[0]])
    return output_file
