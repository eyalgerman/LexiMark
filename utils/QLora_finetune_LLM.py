import torch
import os

os.environ['DISABLE_TQDM'] = '1'
import sys
import json
from datetime import datetime
from options import config_instance

# import IPython
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments, LlamaConfig,
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login, HfApi, HfFolder


def fine_tune_model(model_name, data_path, base_path, resume_from_checkpoint=None, num_epochs=1):
    """
    Fine-tunes a causal language model using QLoRA and Hugging Face's `transformers` and `trl` libraries.
    Applies 4-bit quantization, PEFT with LoRA, and trains on a CSV dataset.

    Args:
        model_name (str): Name or path of the pre-trained model.
        data_path (str): Path to the training data (CSV with a 'text' column).
        base_path (str): Base directory to save checkpoints and final models.
        resume_from_checkpoint (str, optional): Path to a checkpoint to resume training from.

    Returns:
        Tuple[str, transformers.PreTrainedTokenizer]: Path to the saved (unmerged) model directory and tokenizer.
    """
    login(token=config_instance.hagginface_token) #hagginface token
    device = 'cuda'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # # tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.unk_token_id
    # Initialize tokenizer and set padding token if not defined
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set EOS as pad if no padding token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # model.resize_token_embeddings(len(tokenizer))  # Ensure model compatibility

    compute_dtype = getattr(torch, "float16")
    print(compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_flash_attention_2=False,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model has the correct pad token

    print(model)
    print(f"Model is running on: {next(model.parameters()).device}")

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
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    model = prepare_model_for_kbit_training(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    num_epochs = num_epochs
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    data_name = data_path.rstrip('/').split('/')[-1].split('.')[0]
    # new model name
    new_model = f"{model_name.split('/')[-1]}_{data_name}_QLORA_{current_time}_epochs_{num_epochs}"

    if len(new_model) > 255: # Ensure the model name is not too long for filesystem limits
        short_data_name = "_".join(data_name.split("_")[:3])
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        new_model = f"{model_name.split('/')[-1]}_{short_data_name}_QLORA_{current_time}_epochs_{num_epochs}"

    # training_arguments = TrainingArguments(
    #     output_dir=f"{base_path}checkpoints/results_{new_model}",
    #     evaluation_strategy="epoch",
    #     optim="paged_adamw_8bit",
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=1,
    #     gradient_accumulation_steps=1,
    #     log_level="debug",
    #     save_steps=5000,
    #     logging_steps=1000,
    #     learning_rate=3e-4,
    #     num_train_epochs=num_epochs,
    #     warmup_steps=20,
    #     lr_scheduler_type="cosine",
    #     report_to=[],  # Prevents any integration logs
    #     logging_strategy="no",  # Disables console logging for progress updates
    #     save_total_limit=3,
    #     resume_from_checkpoint=resume_from_checkpoint,
    #     disable_tqdm=False  # Disables tqdm progress bars for both training and evaluation  -TO SHOW progress
    # )
    training_arguments = SFTConfig(
        # output_dir=f"{base_path}checkpoints/results_{new_model}",
        output_dir=os.path.join(base_path, "checkpoints", f"{new_model}"),
        # evaluation_strategy="epoch",
        optim="paged_adamw_8bit",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        log_level="debug",
        save_steps=5000,
        logging_steps=1000,
        learning_rate=3e-4,
        num_train_epochs=num_epochs,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to=[],  # Prevents any integration logs
        logging_strategy="no",
        save_total_limit=3,
        resume_from_checkpoint=resume_from_checkpoint,
        disable_tqdm=False,
        dataset_text_field="text"
    )

    data_files = {'train': data_path, 'test': data_path}
    dataset = load_dataset('csv', data_files=data_files, verbose=0)

    # Ensure the dataset has the 'text' field or update the column name accordingly
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        peft_config=peft_config,
        # dataset_text_field="text",
        # tokenizer=tokenizer,
    )

    print_trainable_parameters(model)
    torch.cuda.empty_cache()
    trainer.evaluate()
    trainer.train()

    # new_model_path = f"{base_path}Unmerged/{new_model}"
    new_model_path = os.path.join(base_path, "Unmerged", new_model)
    trainer.model.save_pretrained(new_model_path)
    return new_model_path, tokenizer


def upload_model_to_huggingface(output_directory, model_id):
    """
    Uploads a fine-tuned model to the Hugging Face Hub.

    Args:
        output_directory (str): Path to the local directory containing the model and tokenizer.
        model_id (str): Name of the target model repo on the Hugging Face Hub.
    """
    # Get token from the HfFolder
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found, please login using `huggingface_cli login`.")

    # Initialize the HfApi to handle the upload
    api = HfApi()

    # Create or use existing repository
    repo_url = api.create_repo(name=model_id, token=token,
                               private=True)  # Change `private` to False if you want the repo to be public
    print(f"Repository created at: {repo_url}")

    # Upload the model files to the repository
    api.upload_folder(
        folder_path=output_directory,
        path_in_repo="",
        repo_id=model_id,
        repo_type="model",
        token=token
    )
    print("Model successfully uploaded to Hugging Face Hub.")


def merge_and_upload_model(model_name, new_model, tokenizer):
    """
    Merges LoRA weights with the base model and saves the resulting model.

    Args:
        model_name (str): The base model's name or path.
        new_model (str): Path to the LoRA fine-tuned model directory.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used with the model.
    Returns:
        str: Path to the merged model directory.
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, new_model)
    merged_model = peft_model.merge_and_unload()
    new_model = new_model.replace("/Unmerged", "")
    output_merged_dir = new_model + "_Merged"  # Adjust this path as necessary

    os.makedirs(output_merged_dir, exist_ok=True)
    merged_model.save_pretrained(output_merged_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_merged_dir)
    return output_merged_dir
    # # Upload the merged model to Hugging Face
    # upload_model_to_huggingface(output_merged_dir, new_model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Function to reload a saved model
def reload_model(model_directory, device='cuda'):
    """
    Loads a saved model from a specified directory.

    Args:
        model_directory (str): Path to the directory where the model is saved.
        device (str): Device to load the model onto, defaults to 'cuda'.

    Returns:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_directory).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    return model, tokenizer


# Function to generate text using the model
def generate_text(model, tokenizer, input_text, max_length=256):
    """
    Generates text using a pre-trained model given an input prompt.

    Args:
        model: The pre-trained model.
        tokenizer: The corresponding tokenizer.
        input_text (str): Input text to generate text from.
        max_length (int): Maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    # Encode the input text to tensor
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    # Generate output from the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def resume_training(model_name, data_path, data_name, checkpoint_dir):
    """
    Resumes training from a given checkpoint.

    Args:
        model_name (str): Model name or path.
        data_path (str): Path to training data.
        data_name (str): Name of the dataset (used for naming outputs).
        checkpoint_dir (str): Path to the checkpoint to resume from.

    Returns:
        Tuple[str, transformers.PreTrainedTokenizer]: Path to the unmerged model and tokenizer.
    """
    return fine_tune_model(model_name, data_path, data_name, resume_from_checkpoint=checkpoint_dir)


def use_fine_tune_model(output_merged_dir):
    """
    Loads a merged fine-tuned model and runs a sample text generation.

    Args:
        output_merged_dir (str): Directory containing the merged model and tokenizer.
    """
    # Load the model and tokenizer
    loaded_model, loaded_tokenizer = reload_model(output_merged_dir)

    # Specify your input text
    input_prompt = "You to your beauteous blessings add a "

    # Generate text
    output_text = generate_text(loaded_model, loaded_tokenizer, input_prompt)
    print("Generated Text:", output_text)


def main(model_name, data_path, base_path, num_epochs=1):
    """
    Main execution flow to fine-tune a model, merge and save it, and test generation.

    Args:
        model_name (str): Name or path of the base model to fine-tune.
        data_path (str): Path to the training dataset (CSV).
        base_path (str): Base path to store checkpoints and outputs.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        str: Path to the merged and saved model directory.
    """
    print("Starting Fine-Tuning Process...")
    print("Model Name:", model_name)
    print("Data Path:", data_path)

    new_model, tokenizer = fine_tune_model(model_name, data_path, base_path, num_epochs=num_epochs)

    # new_model, tokenizer = resume_training(model_name, data_path, data_name, checkpoint_dir)
    output_merged_dir = merge_and_upload_model(model_name, new_model, tokenizer)
    use_fine_tune_model(output_merged_dir)
    return output_merged_dir


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
