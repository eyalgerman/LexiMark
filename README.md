# LexiMark:Robust Watermarking via Lexical Substitutions to Enhance Membership Verificationofan LLM’s Textual Training Data

## Overview

LexiMark is a robust watermarking technique designed to embed identifiable markers in textual datasets used for training Large Language Models (LLMs). Unlike traditional watermarking methods that introduce noticeable modifications, LexiMark applies synonym substitutions for carefully selected high-entropy words, ensuring that the watermark remains undetectable while preserving the semantic integrity of the text. This technique enhances an LLM’s memorization of the watermarked data, enabling reliable dataset verification and unauthorized data usage detection.

## Features

- **LexiMark Watermarking**: Embeds high-entropy word substitutions to create an imperceptible yet robust watermark.
- **Fine-Tuning with QLoRA**: Utilizes quantized low-rank adaptation for efficient fine-tuning on selected LLMs.
- **Membership Verification**: Leverages membership inference attacks (MIA) to detect whether an LLM was trained on watermarked data.
- **Dataset Protection**: Prevents unauthorized use of proprietary datasets in LLM training.
- **Robustness Against Removal**: Subtle lexical modifications make the watermark resistant to automated and manual detection.

## Installation

Ensure Python (>=3.9) and the required dependencies are installed:

```bash
pip install -r requirements.txt
```

## Configuration

A configuration file (`config.json`) is required with:

- `API_KEY`: API key for OpenAI models.
- `DATA_DIR`: Path where to save datasets and models.

Example `config.json`:

```json
{
  "API_KEY": "your_api_key",
  "DATA_DIR": "path/to/data"
}
```

## Usage

### Running LexiMark Watermarking

To apply the LexiMark watermark to a dataset:

```bash
python main.py --method top-k-highest --mode <dataset_mode> --target_model <model_name>
```

### Arguments:
- `--method`: Set to `top-k-highest` to use LexiMark or other watermarking techniques.
- `--k`: Number of words to replace (default: `5`).
- `--mode`: Dataset selection (`BookMIA`, `WikiMIA`, `PILE`, `PILE-{sub-dataset-name}`).
- `--synonym_method`: Method for finding synonyms (`wordnet`, `gpt4o`, `context`, `sbert`, `lexsub_dropout`, `lexsub_concatenation`).
- `--watermark_non_member`: Whether to watermark non-member data (`True` by default).
- `--split`: Specifies whether to split the textual data into smaller parts. If set to a value greater than 0, the data will be split into segments where each segment does not exceed the specified maximum length.
- `--watermarks`: List of additional watermarks to apply (optional).
- `--context_th`: Threshold for the context synonym method (default: `0.8`).
- `--seed`: Seed for the random synonym method (default: `42`).
- `--use_existing`: Whether to use existing data and models (`all` by default).
- `--p`: Percentage of words to replace for methods based percentage(default: `0.2`).


### Example Usage:

```bash
python main.py --method top-k-highest --mode BookMIA --target_model meta-llama/Llama-3.1-8B --use_existing all --synonym_method context --context_th 0.8
```

## Watermarking Process

LexiMark follows a two-phase approach:

1. **Watermark Embedding**:
   - Identifies high-entropy words in a sentence.
   - Replaces them with higher-entropy synonyms while preserving meaning.
   - Uses methods like BERT-based lexical substitution, SBERT, and GPT-4o for synonym selection.
2. **Watermark Detection**:
   - Performs Membership Inference Attacks (MIA) to identify watermarked data.
   - Uses Min-K%++ and other MIAs for detection.
   - Achieves superior AUROC scores across diverse datasets.

## Synonym Methods
LexiMark provides multiple synonym substitution methods for embedding the watermark:

- **WordNet (`wordnet`)**: Uses WordNet’s lexical database to find synonyms for the target word. This method does not consider sentence context but ensures direct synonym replacement.
- **GPT-4o (`gpt4o`)**: Uses OpenAI's GPT-4o API to generate context-aware synonyms by understanding the full sentence structure and selecting semantically appropriate substitutions.
- **Contextual (`context`)**: Uses a context-aware synonym finder that selects synonyms based on the sentence structure, ensuring meaningful and natural replacements.
- **SBERT (`sbert`)**: Employs Sentence-BERT embeddings to find synonyms with high contextual similarity, ensuring that the replacement word fits naturally into the given text.
- **Lexical Substitution Dropout (`lexsub_dropout`)**: Applies a masked language model to predict and substitute words while preserving sentence structure. This method replaces words in a controlled manner based on their contextual probability.
- **Lexical Substitution Concatenation (`lexsub_concatenation`)**: Uses a concatenative approach for synonym prediction, improving robustness by selecting words dynamically within a given threshold.

## Watermark Detection Output Structure
LexiMark organizes the watermark detection results in structured directories to ensure traceability and ease of evaluation:

- **Results Directory Structure**:
  The detection results are stored in a folder named based on the model and timestamp:
  ```python
  result_folder = f"{args.output_dir}/{dataset}/M={model}_{current_time}"
  ```
  - This ensures that results from different models and datasets are kept separate.
  
- **Prediction Scores (`preds` file)**:
  - The file `preds_{kind}.csv` contains the prediction scores assigned to each individual record.
  - These scores help assess whether a specific record was likely part of the training data.
  
- **Evaluation Metrics (`metrics` file)**:
  - The metric files store aggregate evaluation results, including:
    - **AUROC (Area Under Receiver Operating Characteristic Curve)**: Measures the overall effectiveness of watermark detection.
    - **TPR@FPR (True Positive Rate at Fixed False Positive Rate)**: Evaluates detection performance under controlled false-positive constraints.

## File Structure

```
├── main.py                     # Main execution script
├── watermarks/                 # Watermarking methods
├── watermark_detection/         # Detection methods
├── utils/                       # Utility functions
├── config.json                  # Configuration file
├── requirements.txt             # Dependencies
```

## Evaluation Results

LexiMark was evaluated on benchmark datasets such as **BookMIA** and **The Pile**, using LLaMA-1 7B, LLaMA-3 8B, Mistral-7B, and Pythia-6.9B models. The method consistently outperformed baseline approaches, achieving AUROC improvements of **4.2% to 25.7%**, with superior performance in **dataset membership detection** while preserving **semantic integrity**.
