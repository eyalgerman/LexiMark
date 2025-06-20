import io
import os
from typing import Union
import docx
import fitz  # PyMuPDF
from pyarrow import BufferReader


def extract_text_from_file(file: Union[io.BufferedReader, str]) -> str:
    """
    Extracts text content from a file of supported formats: .txt, .docx, or .pdf.

    The function reads the file and returns its full textual content based on the file type.

    Args:
        file (Union[io.BufferedReader, str]): A file-like object or a string path to the file.

    Returns:
        str: Extracted text from the file. Returns a message if the file type is unsupported.

    Supported Formats:
        - .txt
        - .docx
        - .pdf

    Note:
        The function assumes `file` has a `.name` attribute for extension detection.
    """
    # If file is a path string, open it as binary
    if isinstance(file, str):
        name = file.lower()
        with open(file, "rb") as f:
            return extract_text_from_file(f)
    if not hasattr(file, 'name'):
        raise ValueError("The provided file-like object must have a 'name' attribute.")
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif name.endswith(".pdf"):
        text = []
        pdf_bytes = file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    return "Unsupported file format."


def extract_texts_from_folder(folder_path: str) -> dict:
    """
    Extracts text content from all supported files in a specified folder.

    This function scans the folder for files with .txt, .docx, or .pdf extensions,
    reads each one, and returns a dictionary mapping filenames to their extracted text.

    Args:
        folder_path (str): The path to the folder containing documents.

    Returns:
        dict: A dictionary where keys are filenames and values are the extracted text content.
    """
    if not os.path.isdir(folder_path):
        text = extract_text_from_file(folder_path)
        return {os.path.basename(folder_path): text} if text else {}
    extracted_texts = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if os.path.isfile(filepath) and filename.lower().endswith(('.txt', '.docx', '.pdf')):
            with open(filepath, "rb") as f:
                text = extract_text_from_file(f)
                extracted_texts[filename] = text

    return extracted_texts