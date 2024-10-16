import logging
import time

from db_client import DBClient
from processors.dataset_processor import ProcessorMeta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Константы
DATABASE_NAME = "TrustLLM_ru"
DEFAULT_SLEEP_INTERVAL = 600  # in seconds
COLLECTION_RESULTS = "results_test"
COLLECTION_TOP_QUESTIONS = "top_questions_test"


class MetricProcessor:
    """
    Класс для запуска обработки всех датасетов.
    """

    def __init__(self):
        self.db_client = DBClient(db_name=DATABASE_NAME)
        self.processors = self.get_processors()

    def get_processors(self):
        """
        Инициализирует все доступные процессоры датасетов.
        """
        return [
            processor(self.db_client, COLLECTION_RESULTS, COLLECTION_TOP_QUESTIONS)
            for processor in ProcessorMeta.registry
        ]

    def run(self):
        """
        Запускает обработку всех датасетов в цикле.
        """
        while True:
            for processor in self.processors:
                try:
                    processor.process_dataset()
                except Exception as e:
                    logging.error(
                        f"Ошибка при обработке процессора {processor.__class__.__name__}: {e}"
                    )
            logging.info("Все метрики обновлены. Ожидание перед следующим запуском...")
            time.sleep(DEFAULT_SLEEP_INTERVAL)


if __name__ == "__main__":
    metric_processor = MetricProcessor()
    metric_processor.run()
