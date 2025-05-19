import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import os
from datetime import datetime

def pretrain_model(model_name, data_path, base_path, num_epochs=1, batch_size=2, learning_rate=3e-4):
    """
    Pretrains a causal language model on a CSV dataset.
    Args:
        model_name (str): Name or path of the pre-trained model.
        data_path (str): Path to the training data (CSV with a 'text' column).
        base_path (str): Base directory to save checkpoints and final models.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Training batch size.
        learning_rate (float): Learning rate.
    Returns:
        Tuple[str, transformers.PreTrainedTokenizer]: Path to the saved model directory and tokenizer.
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data
    data_files = {'train': data_path, 'test': data_path}
    dataset = load_dataset('csv', data_files=data_files, verbose=0)
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Output path
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    data_name = data_path.rstrip('/').split('/')[-1].split('.')[0]
    new_model = f"{model_name.split('/')[-1]}_{data_name}_PRETRAINED_{current_time}_epochs_{num_epochs}"
    output_dir = os.path.join(base_path, "Pretrained", new_model)
    os.makedirs(output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        # warmup_steps=20,
        # lr_scheduler_type="cosine",
        save_steps=5000,
        logging_steps=1000,
        # report_to=[],
        logging_strategy="no",
        save_total_limit=3,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_datasets['train'],
        # eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print(f"Starting pretraining, saving to {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Finished pretraining, model saved to {output_dir}")

    return output_dir, tokenizer


def main(model_name, data_path, base_path):
    """
    Main function to pretrain a model on a given dataset.
    Args:
        model_name (str): Name or path of the pre-trained model.
        data_path (str): Path to the training data (CSV with a 'text' column).
        base_path (str): Base directory to save checkpoints and final models.
    """
    new_model, tokenizer = pretrain_model(model_name, data_path, base_path,
                   num_epochs=1, batch_size=2, learning_rate=1e-4)
    return new_model