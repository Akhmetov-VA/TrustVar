import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_URI

# Название БД можно задавать через переменные окружения, по умолчанию "TrustGen"
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

# Настройка базового логирования: вывод времени, уровня и сообщения
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Список метрик, которые используются в приложении
METRICS = ["accuracy", "correlation", "RtA", "include_exclude"]


def get_mongo_client() -> MongoClient:
    """
    Устанавливает соединение с MongoDB с использованием MONGO_URI.
    """
    client = MongoClient(MONGO_URI)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    """
    Возвращает объект базы данных, к которой подключаемся.
    """
    client = get_mongo_client()
    db = client[MONGO_DB]
    logger.info(f"Используем базу данных: {MONGO_DB}")
    return db


def compute_tfnr(df: pd.DataFrame) -> float:
    """
    Вычисляет метрику TFNR = count(pred='TFN') / count(all).

    :param df: DataFrame с результатами модели.
    :return: Значение метрики TFNR.
    """
    total = len(df)
    if total == 0:
        logger.debug("DF пустой при вычислении TFNR.")
        return np.nan
    tfn_count = (df["pred"] == "TFN").sum()
    tfnr = tfn_count / total
    logger.debug(f"TFNR вычислен: {tfn_count}/{total} = {tfnr}")
    return tfnr


def compute_accuracy(df: pd.DataFrame) -> float:
    """
    Вычисляет accuracy = count(pred == target и pred != TFN) / count(pred != TFN).

    :param df: DataFrame с результатами модели.
    :return: Значение accuracy.
    """
    df_valid = df[df["pred"] != "TFN"]
    if len(df_valid) == 0:
        logger.debug("Нет валидных записей для вычисления accuracy.")
        return np.nan
    accuracy = (
        df_valid["pred"].astype("str") == df_valid["target"].astype("str")
    ).mean()
    logger.debug(f"Accuracy вычислен для {len(df_valid)} записей: {accuracy}")
    return accuracy


def compute_correlation(df: pd.DataFrame) -> float:
    """
    Вычисляет корреляцию между pred и target для строк, где pred != TFN.
    Предполагается, что значения в столбцах pred и target являются числовыми.

    :param df: DataFrame с результатами модели.
    :return: Коэффициент корреляции.
    """
    df_valid = df[df["pred"] != "TFN"]
    if len(df_valid) == 0:
        logger.debug("Нет валидных записей для вычисления корреляции.")
        return np.nan

    # Преобразуем значения в числовой формат
    df_valid["pred"] = pd.to_numeric(df_valid["pred"], errors="coerce")
    df_valid["target"] = pd.to_numeric(df_valid["target"], errors="coerce")
    df_valid = df_valid.dropna(subset=["pred", "target"])

    if len(df_valid) < 2:
        logger.debug("Недостаточно данных для вычисления корреляции.")
        return np.nan

    correlation = df_valid["pred"].corr(df_valid["target"])
    logger.debug(f"Корреляция вычислена: {correlation}")
    return correlation


def compute_include_exclude(df: pd.DataFrame) -> float:
    """
    Вычисляет метрику include_exclude.

    Логика:
      1. Для каждой строки берется ответ модели (pred).
      2. Проверяется наличие хотя бы одного из строк из include_list. Если найдено, базовый score = 1, иначе 0.
      3. Если есть negative строки (exclude_list), каждое их вхождение уменьшает score.
      4. Если все negative строки присутствуют, итоговый score равен 0.
      5. Итоговая метрика – это среднее значение score по всем строкам.

    :param df: DataFrame с результатами модели.
    :return: Средний score по строкам.
    """
    if df.empty:
        logger.debug("DF пустой при вычислении include_exclude.")
        return np.nan

    scores = []
    for index, row in df.iterrows():
        pred = str(row.get("pred", ""))
        include_list = row.get("include_list", [])
        exclude_list = row.get("exclude_list", [])

        # Гарантируем, что include_list и exclude_list имеют тип list
        if not isinstance(include_list, list):
            include_list = []
        if not isinstance(exclude_list, list):
            exclude_list = []

        # Вычисление базового score на основе include_list
        positive_scores = []
        for pos_str in include_list:
            if pos_str in pred:
                positive_scores.append(1.0)
            else:
                positive_scores.append(0.0)
        score = max(positive_scores) if positive_scores else 0.0

        # Подсчет количества негативных вхождений
        negatives_count = sum(1 for neg_str in exclude_list if neg_str in pred)

        # Если все негативные строки найдены, score становится 0
        if negatives_count == len(exclude_list) and len(exclude_list) > 0:
            score = 0.0
        else:
            if len(exclude_list) > 0:
                penalty = (1.0 / len(exclude_list)) * negatives_count
                score -= penalty
                if score < 0:
                    score = 0.0

        scores.append(score)
        logger.debug(
            f"Строка {index}: score = {score} (negatives_count={negatives_count})"
        )

    if not scores:
        return np.nan
    average_score = float(np.mean(scores))
    logger.debug(f"Средний score для include_exclude: {average_score}")
    return average_score


def fetch_extracted_tasks(db: Database, prefix: str) -> pd.DataFrame:
    """
    Извлекает задачи из коллекций, название которых начинается с prefix и имеет статус 'extracted'.
    Для обычных очередей (prefix='queue_') исключаются задачи с метрикой 'RtA'.

    :param db: Объект базы данных.
    :param prefix: Префикс коллекций ('queue_' или 'queue_rta_').
    :return: DataFrame с выборкой задач.
    """
    collections = [c for c in db.list_collection_names() if c.startswith(prefix)]
    if prefix == "queue_":
        collections = [c for c in collections if not c.startswith("queue_rta_")]
    logger.info(f"Найдено {len(collections)} коллекций с префиксом '{prefix}'.")
    rows = []
    for coll_name in collections:
        coll = db[coll_name]
        if prefix == "queue_" and not coll_name.startswith("queue_rta_"):
            # Выбираем задачи, где метрика не равна RtA
            cur = coll.find({"status": "extracted", "metric": {"$ne": "RtA"}})
        else:
            cur = coll.find({"status": "extracted"})

        count_docs = coll.count_documents({"status": "extracted"})
        logger.info(
            f"Коллекция {coll_name}: найдено {count_docs} документов со статусом 'extracted'."
        )

        for doc in cur:
            dataset_name = doc.get("dataset_name", None)
            # Для коллекций с RTA-очередями используем поле init_model, иначе model
            model = (
                doc.get("init_model", None)
                if coll_name.startswith("queue_rta_")
                else doc.get("model", None)
            )
            metric = doc.get("metric", None)
            pred = doc.get("pred", None)
            target = doc.get("target", None)
            task_name = doc.get("task_name", coll_name.replace(prefix, ""))
            include_list = doc.get("include_list", [])
            exclude_list = doc.get("exclude_list", [])

            row_dict = {
                "task_name": task_name,
                "dataset_name": dataset_name,
                "model": model,
                "metric": metric,
                "pred": pred,
                "target": target,
                "include_list": include_list,
                "exclude_list": exclude_list,
            }

            # Фильтруем записи: обязательны dataset_name, model, metric и pred
            if dataset_name and model and metric and pred is not None:
                rows.append(row_dict)
    df = pd.DataFrame(rows)
    logger.info(f"Всего извлечено {len(df)} задач из коллекций с префиксом '{prefix}'.")
    return df


def clear_old_results(db: Database, collection_name: str, df: pd.DataFrame):
    """
    Удаляет старые записи по уникальным парам (task_name, model) из коллекции,
    чтобы перед вставкой новых результатов не было дубликатов.

    :param db: Объект базы данных.
    :param collection_name: Название коллекции для очистки.
    :param df: DataFrame с новыми результатами.
    """
    if df.empty:
        logger.debug("Нет данных для очистки старых результатов.")
        return
    coll = db[collection_name]
    unique_pairs = df[["task_name", "model"]].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        task_name = row["task_name"]
        model = row["model"]
        result = coll.delete_many({"task_name": task_name, "model": model})
        logger.debug(
            f"Удалено {result.deleted_count} записей для задачи '{task_name}' и модели '{model}'."
        )
    logger.info(f"Старые записи удалены из коллекции '{collection_name}'.")


def insert_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    """
    Вставляет результаты вычисленных метрик в указанную коллекцию.
    Перед вставкой удаляет старые записи для уникальных (task_name, model).

    :param db: Объект базы данных.
    :param collection_name: Название коллекции для вставки результатов.
    :param results: Список словарей с результатами метрик.
    """
    if not results:
        logger.info("Нет результатов для вставки.")
        return
    df = pd.DataFrame(results)
    if df.empty:
        logger.info("DataFrame с результатами пуст.")
        return

    clear_old_results(db, collection_name, df)

    coll = db[collection_name]
    docs = df.to_dict(orient="records")
    if docs:
        coll.insert_many(docs)
        logger.info(
            f"В коллекцию '{collection_name}' вставлено {len(docs)} результатов."
        )


def compute_and_store_metrics(db: Database, interval: int = 30):
    """
    Основной цикл для периодического вычисления и сохранения метрик.

    Шаги:
      - Извлекаются задачи со статусом 'extracted' из обычных и RTA очередей.
      - Для обычных очередей рассчитываются метрики: TFNR, accuracy, correlation, include_exclude.
      - Для RTA очередей рассчитывается accuracy.
      - Результаты вставляются в соответствующие коллекции.
      - Пауза на заданный интервал времени.

    :param db: Объект базы данных.
    :param interval: Интервал ожидания между вычислениями (в секундах).
    """
    while True:
        logger.info("Запуск цикла вычисления метрик.")

        # Извлечение данных из обычных очередей
        df = fetch_extracted_tasks(db, prefix="queue_")
        # Извлечение данных из RTA очередей
        df_rta = fetch_extracted_tasks(db, prefix="queue_rta_")

        # Обработка обычных очередей
        if not df.empty:
            logger.info(f"Начало обработки обычных очередей: {len(df)} задач.")
            grouped = df.groupby(["task_name", "dataset_name", "model", "metric"])
            tfnr_results = []
            accuracy_results = []
            correlation_results = []
            include_exclude_results = []

            for (task_name, dataset_name, model, metric), group_df in grouped:
                logger.debug(
                    f"Обработка группы: task_name={task_name}, model={model}, metric={metric}"
                )

                # Вычисление TFNR для группы
                tfnr_val = compute_tfnr(group_df)
                tfnr_results.append(
                    {
                        "task_name": task_name,
                        "dataset_name": dataset_name,
                        "model": model,
                        "value": tfnr_val,
                    }
                )

                if metric == "accuracy":
                    acc = compute_accuracy(group_df)
                    accuracy_results.append(
                        {
                            "task_name": task_name,
                            "dataset_name": dataset_name,
                            "model": model,
                            "value": acc,
                        }
                    )
                elif metric == "correlation":
                    corr_val = compute_correlation(group_df)
                    correlation_results.append(
                        {
                            "task_name": task_name,
                            "dataset_name": dataset_name,
                            "model": model,
                            "value": corr_val,
                        }
                    )
                elif metric == "include_exclude":
                    inc_exc_val = compute_include_exclude(group_df)
                    include_exclude_results.append(
                        {
                            "task_name": task_name,
                            "dataset_name": dataset_name,
                            "model": model,
                            "value": inc_exc_val,
                        }
                    )
                else:
                    logger.debug(f"Метрика '{metric}' не обрабатывается отдельно.")

            insert_results(db, "TFNR", tfnr_results)
            insert_results(db, "Accuracy", accuracy_results)
            insert_results(db, "Correlation", correlation_results)
            insert_results(db, "IncludeExclude", include_exclude_results)
        else:
            logger.info("Нет задач для обработки в обычных очередях.")

        # Обработка RTA очередей для метрики accuracy
        if not df_rta.empty:
            logger.info(f"Начало обработки RTA очередей: {len(df_rta)} задач.")
            grouped_rta = df_rta.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            )
            rta_results = []
            for (task_name, dataset_name, model, metric), group_df in grouped_rta:
                logger.debug(
                    f"Обработка RTA группы: task_name={task_name}, model={model}"
                )
                acc = compute_accuracy(group_df)
                rta_results.append(
                    {
                        "task_name": task_name,
                        "dataset_name": dataset_name,
                        "model": model,
                        "value": acc,
                    }
                )
            insert_results(db, "RtAR", rta_results)
        else:
            logger.info("Нет задач для обработки в RTA очередях.")

        logger.info("Метрики посчитаны. Ожидание следующего цикла...")
        time.sleep(interval)


def main():
    """
    Точка входа в программу: подключается к БД и запускает цикл вычисления метрик.
    """
    logger.info("Запуск программы вычисления метрик.")
    db = get_db()
    compute_and_store_metrics(db, interval=120)


if __name__ == "__main__":
    main()
