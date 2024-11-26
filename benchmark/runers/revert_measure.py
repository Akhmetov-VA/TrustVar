import logging

from benchmark.db_client import DBClient

from .processors.dataset_processor import ProcessorMeta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Константы
DATABASE_NAME = "TrustLLM_ru"
COLLECTION_RESULTS = "results_test"
COLLECTION_TOP_QUESTIONS = "top_questions_test"


def revert_all_processors(delete_collections=False):
    """
    Отменяет статус 'measured' на 'completed' для всех процессоров, зарегистрированных в ProcessorMeta.
    При необходимости удаляет коллекции с результатами и топ вопросами.
    """
    db_client = DBClient(db_name=DATABASE_NAME)
    processors = [
        processor(db_client, COLLECTION_RESULTS, COLLECTION_TOP_QUESTIONS)
        for processor in ProcessorMeta.registry
    ]

    for processor in processors:
        try:
            processor.revert_status(delete_collections=delete_collections)
        except Exception as e:
            logging.error(
                f"Ошибка при отмене статуса в процессоре {processor.__class__.__name__}: {e}"
            )


if __name__ == "__main__":
    revert_all_processors(delete_collections=True)
