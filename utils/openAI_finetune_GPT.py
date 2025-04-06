import os
import openai
import csv
import json
import time

from openai import OpenAI

from options import config_instance


def csv_to_chat_jsonl(csv_path, output_jsonl_path):
    """
    Convert a CSV (with one column 'text') to the new Chat-based fine-tuning format:
    {
      "messages": [{"role": "user", "content": ""}],
      "completion": {"role": "assistant", "content": "<CSV row text>"}
    }
    """
    with open(csv_path, 'r', encoding='utf-8') as fin, \
         open(output_jsonl_path, 'w', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        for row in reader:
            text = row["text"].strip()
            record = {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": text}
                ]
                # ,
                # "completion": {
                #     "role": "assistant",
                #     "content": text
                # }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(model_name, data_path):
    """
    Fine-tune an OpenAI model on a CSV file that has 1 column: 'text'.
    Each row becomes a { "prompt": "", "completion": "<row text>" } pair.
    Then we upload the data and create a fine-tune job with OpenAI.

    :param model_name: (str) e.g. "gpt-3.5-turbo" (GPT-4 not yet available for fine-tuning)
    :param data_path:  (str) path to the CSV file containing a single 'text' column
    """
    # 1) Set OpenAI API key
    # with open("config.json", "r") as file:
    #     config = json.load(file)
    # openai.api_key = config["API_KEY"]
    client = OpenAI(
        # This is the default and can be omitted
        api_key=config_instance.api_key
    )

    # 2) Convert CSV -> JSONL
    #    We'll store the output in the same folder, replacing .csv with .jsonl
    output_jsonl = data_path.rsplit(".", 1)[0] + "_openai.jsonl"
    csv_to_chat_jsonl(data_path, output_jsonl)

    # 3) Upload JSONL to OpenAI
    upload_response = None
    with open(output_jsonl, "rb") as f:
        upload_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )

    file_id = upload_response.id
    print(f"Uploaded JSONL file. File ID: {file_id}")

    # 4) Create the fine-tune job
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model_name
    )

    print("Fine-tune job response:")
    print(fine_tune_response)

    # Optionally, you can poll job status or just instruct the user
    # to check with "openai api fine_tunes.follow -i <JOB_ID>"
    job_id = fine_tune_response.id
    print(f"Fine-tune job created. Job ID: {job_id}")
    print("You can monitor with: openai api fine_tunes.follow -i", job_id)

    # 5) Poll until the job finishes or fails
    while True:
        status = client.fine_tuning.jobs.retrieve(job_id)
        status_state = status.status

        if status_state in ["succeeded", "failed"]:
            break

        print(f"Current fine-tune status: {status_state}. Waiting 30s...")
        time.sleep(30)

    if status_state == "succeeded":
        new_model_name = status.fine_tuned_model
        print(f"Job succeeded! New model name: {new_model_name}")
        return new_model_name
    else:
        print("Fine-tune job failed.")
        return None


if __name__ == "__main__":
    main(model_name="gpt-3.5-turbo", data_path="my_texts.csv")