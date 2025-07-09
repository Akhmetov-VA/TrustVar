from pathlib import Path

from datasets import load_dataset


def load_and_save_dataset_hf(path: str, name: str, split: str, filename: Path) -> None:
    """
    Downloads a dataset from the Hugging Face library and saves it in CSV format.

    Args:
        path (str): The path to the dataset in the Hugging Face library.
        name (str): The name of the dataset configuration.
        split (str): Part of the dataset to download (for example, "train", "test").
        filename (Path): The path to save the file in CSV format.

    Returns:
        None
    """
    try:
        # Load dataset and convert it to Pandas DataFrame
        dataset = load_dataset(path, name=name, split=split)
        df = dataset.to_pandas()

        # Ensure save directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame in CSV
        df.to_csv(filename, index=False)
        print(f"Dataset successfully saved in file: {filename}")

    except Exception as e:
        print(f"Error loading or saving the dataset: {e}")

def main() -> None:
    """
    The main function is to download and save the MERA (ruhatespeech) dataset.
    """
    # Data save path
    data_path = Path("/home/vadim/work/TrustLLM_ru/data/ruhatespeech")

    # Dataset config name
    dataset_name = "ruhatespeech"

    # load and save dataset
    load_and_save_dataset_hf(
        path="MERA-evaluation/MERA",
        name=dataset_name,
        split="test",
        filename=data_path / f"{dataset_name}.csv",
    )


if __name__ == "__main__":
    main()
