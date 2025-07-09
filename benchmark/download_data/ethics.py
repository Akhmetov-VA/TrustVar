from pathlib import Path

from datasets import load_dataset


def load_and_save_dataset_hf(path: str, name: str, split: str, filename: Path) -> None:
    """
    Downloads a dataset from the Hugging Face library and saves it in CSV format.

    Args:
        path (str): The path to the dataset in the Hugging Face library.
        name (str): The name of the dataset.
        split (str): Part of the dataset to download (for example, "train", "test").
        filename (Path): The path to save the file in CSV format.

    Returns:
        None
    """
    try:
        # Load dataset and convert it to Pandas DataFrame
        df = load_dataset(path, name=name, split=split).to_pandas()

        # Ensure save directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame in CSV
        df.to_csv(filename, index=False)
        print(f"Dataset successfully saved in file: {filename}")

    except Exception as e:
        print(f"Error loading or saving the dataset: {e}")


def main() -> None:
    """
    The main function is to download and save the ethics dataset.
    """
    # Data save path
    ethics_path = Path("data/ethics")

    # Dataset list for load
    dataset_names = ["per_ethics", "sit_ethics"]

    # A loop for loading and saving each dataset
    for name in dataset_names:
        load_and_save_dataset_hf(
            path="RussianNLP/tape",
            name=f"{name}.raw",
            split="train",
            filename=ethics_path / f"{name}.csv",
        )


if __name__ == "__main__":
    main()
