import os

from utils import process_data, QLora_Medium_Finetune_LLM
from watermark_detection import watermark_detection_2
from main import init_watermark, Config
from options import Options


class ConfigMulti(Config):
    def __init__(self, args):
        super(ConfigMulti, self).__init__(args)

    def build_filename(self, args, method_names, params):
        """
        Build a filename based on the mode, method names, and parameters.

        :param args: Argument object containing mode and synonym method.
        :param method_names: List of watermarking methods (or a single method as a string).
        :param params: Dictionary of relevant parameters for each method.
        :return: Filename as a string.
        """
        # Handle the case where a single method name is provided instead of a list
        if isinstance(method_names, str):
            method_names = [method_names]

        # Prepare the parameter string for each watermarking method
        method_suffixes = []
        for method_name in method_names:
            params = params.get(method_name, {})
            params = {k: f"{v}".replace('.0', '') if isinstance(v, float) else v for k, v in params.items()}
            params = {k: f"{v}".replace('.', '') for k, v in params.items()}
            params_str = "_".join(f"{k}_{v}" for k, v in params.items())

            # if method_name == "None":
            #     method_suffixes.append(f"none_{self.split_str}{self.count_str}")
            # else:
            method_suffix = f"{params_str}"
            method_suffixes.append(f"{method_name}_{method_suffix}")

        # Combine all method suffixes to form a single filename
        combined_suffix = "_".join(method_suffixes)
        combined_suffix = combined_suffix + "_"

        # Return the final filename
        datasets_folder = os.path.join(self.data_dir, "Datasets")
        os.makedirs(datasets_folder, exist_ok=True)
        return datasets_folder + f"/{args.mode}_{self.no_member_str}{combined_suffix}{self.watermark_non_member_str}{self.split_str}{self.count_str}.csv"


def load_and_split_data(config, args, method_names, watermark_functions, params):
    """
    Load, clean, and split data, applying watermarking methods.

    :param config: The configuration object for the dataset.
    :param args: The argument object containing mode and other configurations.
    :param method_names: List of watermarking methods.
    :param watermark_functions: List of watermarking functions to apply.
    :param params: Dictionary containing the parameters for each method.
    :return: Tuple of filenames generated.
    """
    # Build the filename based on method names and params
    filename = config.build_filename(args, method_names, params)

    # Load and process the data
    filename1, filename2 = procces_data.load_clean_data_and_split(
        mode=args.mode, from_idx=0, count=config.count, key_name=args.key_name, split=args.split,
        output_file=filename, watermark=watermark_functions, filter=config.filter,
        watermark_non_member=args.watermark_non_member
    )

    return filename1, filename2, filename


if __name__ == '__main__':
    print("Start")
    # Parse arguments
    args = Options()
    args = args.parser.parse_args()

    # Initialize configuration
    config = Config(args)

    # List to store watermark methods and functions
    watermark_methods = args.watermarks
    watermark_functions = []
    method_params = {}

    # Initialize each watermark method and its corresponding parameters
    for method in watermark_methods:
        print(f"Start watermarking the data with method {method}")
        watermark, params = init_watermark(args, method)
        watermark_functions.append(watermark)
        method_params[method] = params

    # Call the load_and_split_data with multiple watermark functions
    filename1, filename2, filename = load_and_split_data(
        config=config, args=args, method_names=watermark_methods,
        watermark_functions=watermark_functions, params=method_params
    )

    print(f"File saved to {filename1}, {filename2}")
    print("Start Fine-tuning the model")
    # Fine-tune the model
    model = args.target_model
    data = filename1
    new_model = QLora_Medium_Finetune_LLM.main(model, data)
    print("Finished Fine-tuning the model: ", new_model)
    args.data = filename.replace(".csv", ".jsonl")
    args.target_model = new_model
    # Use the fine-tuned model
    print("Start detection on the model")
    watermark_detection_2.main(args)

