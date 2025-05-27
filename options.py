import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="huggyllama/llama-7b", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
        self.parser.add_argument('--ref_model', type=str, default="huggyllama/llama-7b")
        self.parser.add_argument('--output_dir', type=str, default="results", help="the output directory")
        self.parser.add_argument('--data', type=str, default="BookMIA", help="the dataset to evaluate")
        self.parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--key_name', type=str, default="snippet", help="the key name corresponding to the input text. Selecting from: input, parapgrase")

        # Watermarking options
        self.parser.add_argument('--method', type=str, default="top-k-higher", help="the method to use for watermarking")
        self.parser.add_argument('--k', type=int, default=5, help="the number of words to replace")
        self.parser.add_argument('--threshold', type=int, default=-3, help="the threshold for replacing words")
        self.parser.add_argument('--mode', type=str, default="BookMIA", help="the mode to use for watermarking: BookMIA, WikiMIA")
        self.parser.add_argument('--synonym_method', type=str, default="context", help="the method to use for finding synonyms: wordnet, gpt, context, sbert")
        self.parser.add_argument('--watermark_non_member', type=str2bool, default=True, help="whether to watermark non-member data")
        self.parser.add_argument('--split', type=int, default=0, help="Specifies whether to split the textual data into smaller parts. If set to a value greater than 0, the data will be split into segments where each segment does not exceed the specified maximum length. ")
        self.parser.add_argument('--watermarks', nargs='+', default=[], help='List of watermarks to apply (optional)')
        self.parser.add_argument('--context_th', type=float, default=0.8, help="the threshold for the context synonym method")
        self.parser.add_argument('--seed', type=int, default=42, help="the seed for the random synonym method")
        self.parser.add_argument('--use_existing', type=str, default='all', help="whether to use existing data and model")
        self.parser.add_argument('--p', type=float, default=0.2, help="the percentage of words to replace")
        self.parser.add_argument('--n', type=int, default=50, help="Ngram size for the n-gram method")

        # Pretraining options
        self.parser.add_argument('--train_mode', type=str, default="finetune", help="the mode to use for training: pretrain, finetune, none")
        self.parser.add_argument('--post_training', type=str, default=None, help="whether to do post training and in which mode/dataset")

class Config:
    def __init__(self, args):
        """
        Initializes configuration settings for the watermarking pipeline.

        Reads from `config.json` to retrieve API keys and data paths, and sets up
        dynamic file naming logic based on command-line arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as file:
            config = json.load(file)

        API_KEY = config["OPENAI_API_KEY"]
        DATA_DIR = config["DATA_DIR"]
        self.hagginface_token = config["HUGGINGFACE_TOKEN"]
        self.data_dir = DATA_DIR
        self.api_key = API_KEY
        self.count = 5000
        self.filter = "Non-member" if args.mode in ["BookMIA", "Arxiv"] else "all"
        self.count_str = f"{self.count // 1000}k" if self.count >= 1000 else str(self.count)
        self.watermark_non_member_str = "" if args.watermark_non_member else "only_member_"
        self.split_str = f"split_{args.split}_" if args.split > 0 else ""
        self.no_member_str = "no_member_" if args.mode in ["BookMIA", "Arxiv"] else ""
        self.suffix = f"syn_{args.synonym_method}_{self.watermark_non_member_str}{self.split_str}{self.count_str}"

    def build_filename(self, args, method_name, params):
        """
        Constructs the filename for the dataset based on arguments and watermark method.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            method_name (str): The watermarking method name.
            params (dict): Parameters used for watermarking, encoded in the filename.

        Returns:
            str: The constructed path to the CSV file for the dataset.
        """
        params = {k: f"{v}".replace('.0', '') if isinstance(v, float) else v for k, v in params.items()}
        params = {k: f"{v}".replace('.', '') for k, v in params.items()}
        params_str = "_".join(f"{k}_{v}" for k, v in params.items())
        datasets_folder = os.path.join(self.data_dir, "Datasets")
        os.makedirs(datasets_folder, exist_ok=True)
        if method_name == "None":
            return datasets_folder + f"/{args.mode}_{self.no_member_str}original_{self.split_str}{self.count_str}.csv"
        else:
            suffix = f"{params_str}_{self.watermark_non_member_str}{self.split_str}{self.count_str}"
        return datasets_folder + f"/{args.mode}_{self.no_member_str}watermark_{method_name}_{suffix}.csv"

args = Options().parser.parse_args()
config_instance = Config(args)