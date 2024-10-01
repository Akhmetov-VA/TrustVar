from pathlib import Path

from datasets import load_dataset


def load_and_save_dataset_hf(path, name, split, filename):
    """Load a Hugging Face dataset and save it as a CSV file."""
    try:
        # Load the dataset and convert it to a Pandas DataFrame
        df = load_dataset(path, name=name, split=split).to_pandas()

        # Ensure the directory exists before saving
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Save DataFrame to CSV
        df.to_csv(filename, index=False)
        print(f"Successfully saved dataset to {filename}")

    except Exception as e:
        print(f"Error loading or saving dataset: {e}")


# download dataset по Этичности https://huggingface.co/datasets/RussianNLP/tape
ethics_path = Path("data/ethics")

for name in ["per_ethics", "sit_ethics"]:
    load_and_save_dataset_hf(
        "RussianNLP/tape",
        name=f"{name}.raw",
        split="train",
        filename=ethics_path / f"{name}.csv",
    )
