from pathlib import Path

from datasets import load_dataset


def load_and_save_dataset_hf(path: str, name: str, split: str, filename: Path) -> None:
    """
    Загружает датасет из библиотеки Hugging Face и сохраняет его в формате CSV.

    Args:
        path (str): Путь к датасету в библиотеке Hugging Face.
        name (str): Название датасета.
        split (str): Часть датасета для загрузки (например, "train", "test").
        filename (Path): Путь для сохранения файла в формате CSV.

    Returns:
        None
    """
    try:
        # Загружаем датасет и преобразуем его в Pandas DataFrame
        df = load_dataset(path, name=name, split=split).to_pandas()

        # Убедимся, что директория для сохранения файла существует
        filename.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем DataFrame в формате CSV
        df.to_csv(filename, index=False)
        print(f"Датасет успешно сохранен в файл: {filename}")

    except Exception as e:
        print(f"Ошибка при загрузке или сохранении датасета: {e}")


def main() -> None:
    """
    Основная функция для загрузки и сохранения датасета по этичности.
    """
    # Путь для сохранения данных
    ethics_path = Path("data/ethics")

    # Перечень названий датасетов для загрузки
    dataset_names = ["per_ethics", "sit_ethics"]

    # Цикл для загрузки и сохранения каждого датасета
    for name in dataset_names:
        load_and_save_dataset_hf(
            path="RussianNLP/tape",
            name=f"{name}.raw",
            split="train",
            filename=ethics_path / f"{name}.csv",
        )


if __name__ == "__main__":
    main()
