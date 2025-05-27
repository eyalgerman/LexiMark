import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

from utils import QLora_finetune_LLM


def format_example(example):
    """Format dataset examples for instruction tuning."""
    return {"text": f"### Input: \n{example['prompt']}\n### Output: \n{example['response']}\n"}

def tokenize_texts(tokenizer, dataset, max_length=512):
    """Tokenize dataset texts."""
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_length), batched=True)

def instruction_tune(dataset_name, tokenizer_name, model_name, saved_model_path):
    """Perform instruction tuning on a language model."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    ds = load_dataset(dataset_name, split="train")
    formatted_ds = ds.map(format_example)
    tokenized_ds = tokenize_texts(tokenizer, formatted_ds)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=saved_model_path,
        per_device_train_batch_size=8,  
        num_train_epochs=1,                               
        fp16=True,
        save_strategy="no",
        logging_steps=100,
        learning_rate=1e-5,
        gradient_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(saved_model_path)


def instruction_tune_qlora(dataset_name, tokenizer_name, model_name, saved_model_path, batch_size=2, epochs=1):
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Set pad token
    model.config.pad_token_id = tokenizer.pad_token_id
    target_modules = [
            'k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head"
        ]
    if "pythia" in model_name:
        target_modules = [
            # "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            # "input_layernorm", # no
            # "post_attention_layernorm", # no
            "embed_in",
            "embed_out"
        ]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())

    ds = load_dataset(dataset_name, split="train")
    formatted_ds = ds.map(format_example)
    tokenized_ds = tokenize_texts(tokenizer, formatted_ds)
    print(tokenized_ds[0])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=saved_model_path,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=True,
        save_strategy="no",
        logging_steps=100,
        learning_rate=1e-4,  # LoRA can use higher LR
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # for batch in trainer.get_train_dataloader():
    #     print({k: v.shape for k, v in batch.items()})
    #     print(batch['labels'][0])  # Print first label sequence
    #     print("Min label:", batch['labels'][0].min().item(), "Max label:", batch['labels'][0].max().item())
    #     break
    trainer.train()
    trainer.save_model(saved_model_path)

    output_merged_dir = QLora_finetune_LLM.merge_and_upload_model(model_name, saved_model_path, tokenizer)
    return output_merged_dir



def main(dataset_name, tokenizer_name, model_name, base_path, train_mode ="pretrain"):
    """Main function to execute the instruction tuning."""
    # Define the saved model path
    clean_dataset_name = dataset_name.replace("/", "_")
    new_model = f"{model_name.split('/')[-1]}_{clean_dataset_name}_INSTRUCTION_TUNED"
    saved_model_path = os.path.join(base_path, "INSTRUCTION_TUNED", new_model)

    print(f"Train mode: {train_mode}")
    # Perform instruction tuning
    if train_mode.lower() == "qlora" or train_mode.lower() == "finetune":
        saved_model_path = os.path.join(base_path, "INSTRUCTION_TUNED", "Unmerged", new_model)
        saved_model_path = instruction_tune_qlora(dataset_name, tokenizer_name, model_name, saved_model_path)
    else:
        instruction_tune(dataset_name, tokenizer_name, model_name, saved_model_path)

    print(f"Model saved at: {saved_model_path}")
    return saved_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for instruction tuning.")

    # Define command-line arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for instruction tuning.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer name to use.")
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for training.")
    parser.add_argument("--saved_model_path", type=str, required=True, help="Path to save the trained model.")

    args = parser.parse_args()

    instruction_tune(args.dataset_name, args.tokenizer_name, args.model_name, args.saved_model_path)
