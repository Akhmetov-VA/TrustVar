import logging
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.constants import MODELS
from utils.db_client import MongoDBClient, MongoDBConfig
from utils.sync_task import sync_task_once

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация клиента БД (конфигурация берется из переменных окружения)
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

DEFAULT_REGEX = r"(?:^\W*([01]).*)|(?:.*([01])\W*$)"


def generate_prompt_hint(var_cols: List[str]) -> Tuple[str, str]:
    placeholders = ", ".join("{" + c + "}" for c in var_cols)
    hint = f"Вы можете использовать любые выбранные колонки в фигурных скобках: {placeholders}."
    return hint, placeholders


def display_task_summary(df_tasks: pd.DataFrame):
    total_tasks = len(df_tasks)
    unique_datasets = df_tasks["dataset_name"].nunique()
    unique_metrics = df_tasks["metric"].nunique()
    unique_groups = df_tasks["group"].nunique()
    unique_prompts = df_tasks["prompt"].nunique()

    # Если в колонке models содержатся списки, извлекаем все модели
    all_models = [
        model
        for sublist in df_tasks["models"]
        if isinstance(sublist, list)
        for model in sublist
    ]
    unique_models = len(set(all_models))

    rta_count = (
        df_tasks["rta_prompt"].notna().sum() if "rta_prompt" in df_tasks.columns else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего задач", total_tasks)
    col2.metric("Уникальных датасетов", unique_datasets)
    col3.metric("Уникальных метрик", unique_metrics)

    col4, col5, col6 = st.columns(3)
    col4.metric("Уникальных групп", unique_groups)
    col5.metric("Уникальных промптов", unique_prompts)
    col6.metric("Уникальных моделей", unique_models)

    col7, _ = st.columns([1, 1])
    col7.metric("Используется RTA промптов", rta_count)


def filter_tasks_by_group(df_tasks: pd.DataFrame) -> pd.DataFrame:
    if df_tasks.empty:
        return df_tasks
    groups = df_tasks["group"].unique().tolist()
    if len(groups) > 1:
        selected_group = st.selectbox(
            "Выберите группу для отображения:", ["Все"] + groups, key=str(uuid.uuid4())
        )
        if selected_group != "Все":
            df_tasks = df_tasks[df_tasks["group"] == selected_group]
    return df_tasks


def render_update_task():
    with st.expander("Изменить модели задачи", expanded=False):
        st.header("Изменить модели задачи")
        df_tasks = db_client.get_all_tasks()
        if df_tasks.empty:
            st.write("Нет задач для обновления.")
            return

        df_tasks = filter_tasks_by_group(df_tasks)
        if df_tasks.empty:
            st.write("Нет задач в выбранной группе для обновления.")
            return

        # Выбираем задачу для обновления
        task_names = df_tasks["task_name"].unique().tolist()
        selected_task_name = st.selectbox(
            "Выберите задачу:", task_names, key="update_task_selectbox"
        )
        task_to_update = df_tasks[df_tasks["task_name"] == selected_task_name].iloc[0]

        # Только обновление списка моделей
        current_models = task_to_update.get("models", [])
        selected_models = st.multiselect(
            "Выберите модели для задачи:", options=MODELS, default=current_models
        )

        if st.button("Обновить модели"):
            update_data = {"models": selected_models}
            task_id = task_to_update["_id"]
            db_client.update_task(task_id, update_data)
            updated_task = db_client.get_task(task_id)
            sync_task_once(db_client.db, updated_task)
            st.success("Список моделей успешно обновлён!")


def highlight_status(s: str) -> str:
    if s == "Ошибка":
        return "background-color: red; color: white;"
    elif s == "В процессе":
        return "background-color: orange; color: white;"
    elif s == "Завершено":
        return "background-color: green; color: white;"
    else:
        return ""


def restart_stopped_error_tasks(collection_name: str) -> int:
    count_stopped = db_client.update_tasks_status(collection_name, "stopped", "pending")
    count_error = db_client.update_tasks_status(collection_name, "error", "pending")
    return count_stopped + count_error


def stop_pending_tasks(collection_name: str) -> int:
    return db_client.update_tasks_status(collection_name, "pending", "stopped")


def load_data_for_dashboard(collections: List[str]) -> pd.DataFrame:
    data = []
    exclude = ["delete_me", "test"]
    collections = [c for c in collections if c not in exclude]

    for collection_name in collections:
        total_tasks = db_client.count_total_tasks(collection_name)
        statuses = [
            "pending",
            "completed",
            "stopped",
            "error",
            "extracted",
            "processing",
            "transfered_to_rta",
        ]
        status_counts = {
            status: db_client.count_tasks_by_status(collection_name, status)
            for status in statuses
        }

        error_count = status_counts["stopped"] + status_counts["error"]
        if error_count > 0:
            overall_status = "Ошибка"
        elif status_counts["pending"] > 0 or status_counts["processing"] > 0:
            overall_status = "В процессе"
        else:
            overall_status = "Завершено"

        data_row = {
            "Коллекция": collection_name,
            "Всего задач": total_tasks,
            "В ожидании": status_counts["pending"],
            "Выполнено": status_counts["completed"]
            + status_counts["transfered_to_rta"],
            "Измерено": status_counts["extracted"],
            "С ошибками": error_count,
            "Статус": overall_status,
        }
        data.append(data_row)

    return pd.DataFrame(data)


def show_errors(collections: List[str]):
    st.header("Уникальные сообщения об ошибках")
    with st.expander("Показать ошибки", expanded=False):
        # Словарь для хранения информации о коллекциях с ошибками
        collections_with_errors = {}

        # Собираем информацию о задачах с ошибками
        for collection_name in collections:
            stopped_tasks = db_client.get_tasks_by_status(collection_name, "stopped")
            error_tasks = db_client.get_tasks_by_status(collection_name, "error")
            failed_tasks = stopped_tasks + error_tasks

            if failed_tasks:
                collections_with_errors[collection_name] = failed_tasks

                # Отображаем ошибки для коллекции
                error_messages = [
                    task.get("error", "Нет информации об ошибке")
                    for task in failed_tasks
                ]
                error_counts = Counter(error_messages)
                st.subheader(f"Коллекция: {collection_name}")

                # Группируем ошибки по моделям
                models_errors = {}
                for task in failed_tasks:
                    model = task.get("model", "Неизвестная модель")
                    error = task.get("error", "Нет информации об ошибке")
                    if model not in models_errors:
                        models_errors[model] = Counter()
                    models_errors[model][error] += 1

                # Отображаем ошибки по моделям
                for model, errors in models_errors.items():
                    st.write(f"**Модель:** {model}")
                    for i, (error_message, count) in enumerate(errors.items()):
                        st.write(
                            f"- **Ошибка:** {error_message} | **Количество:** {count}"
                        )
                        if i > 5:
                            break
                st.write("---")

        if collections_with_errors:
            # Получаем список коллекций с ошибками
            collections_list = list(collections_with_errors.keys())

            # Добавляем опцию "Все коллекции"
            options = [None, "Все коллекции"] + collections_list

            # Выбор коллекции для перезапуска
            selected_collection = st.selectbox(
                "Выберите коллекцию для перезапуска задач:",
                options,
                index=0,
                key="collection_to_restart",
            )

            total_restarted = 0

            if selected_collection:
                if selected_collection == "Все коллекции":
                    # Перезапускаем задачи во всех коллекциях с ошибками
                    for collection_name in collections_with_errors.keys():
                        modified_count = restart_stopped_error_tasks(collection_name)
                        if modified_count > 0:
                            total_restarted += modified_count
                            st.write(
                                f"В коллекции '{collection_name}' перезапущено {modified_count} задач."
                            )
                else:
                    # Перезапускаем задачи только в выбранной коллекции
                    modified_count = restart_stopped_error_tasks(selected_collection)
                    if modified_count > 0:
                        total_restarted += modified_count
                        st.write(
                            f"В коллекции '{selected_collection}' перезапущено {modified_count} задач."
                        )

                if total_restarted > 0:
                    st.success(f"Всего перезапущено {total_restarted} задач.")
                else:
                    st.info("Не найдено задач для перезапуска.")
        else:
            st.info("Нет задач с ошибками для перезапуска.")


def render_progressbar():
    st.header("Мониторинг очередей")
    if st.checkbox("Загрузить мониторинг очередей", value=False, key="load_monitoring"):
        collections_to_process = sorted(
            [col for col in db_client.list_collections() if col.startswith("queue_")]
        )
        df = load_data_for_dashboard(collections_to_process)

        if df.empty:
            st.info("Нет данных для отображения в очередях.")
        else:
            df = df.sort_values("Коллекция").reset_index(drop=True)
            df_style = df.style.applymap(highlight_status, subset=["Статус"])
            st.write(df_style)

            pending_queues = df.loc[df["В ожидании"] > 0, "Коллекция"].tolist()
            if pending_queues:
                selected_queue = st.selectbox(
                    "Выберите очередь для остановки:",
                    pending_queues,
                    index=None,
                    key="fail_pending_selectbox",
                )
                if selected_queue:
                    st.write(f"Start stopping {selected_queue}")
                    count_stopped = stop_pending_tasks(selected_queue)
                    st.success(
                        f"В коллекции '{selected_queue}' остановлено {count_stopped} задач."
                    )

            if df["С ошибками"].sum() > 0:
                show_errors(collections_to_process)
    else:
        st.info("Нажмите кнопку выше, чтобы загрузить мониторинг очередей.")

    st.header("Просмотр данных коллекции")
    collections_to_process = sorted(
        [col for col in db_client.list_collections() if col.startswith("queue_")]
    )
    if collections_to_process:
        selected_collection = st.selectbox(
            "Выберите коллекцию",
            collections_to_process,
            key="dashboard_select_collection",
        )
        if selected_collection:
            if st.button(
                "Показать данные коллекции", key=f"show_data_{selected_collection}"
            ):
                st.session_state[f"data_loaded_{selected_collection}"] = True

            if st.session_state.get(f"data_loaded_{selected_collection}", False):
                collection = db_client.get_collection(selected_collection)
                data = list(collection.find())
                df_collection = pd.DataFrame(data)
                if not df_collection.empty and "_id" in df_collection.columns:
                    df_collection = df_collection.drop(columns=["_id"])

                if not df_collection.empty and "model" in df_collection.columns:
                    models_in_data = df_collection["model"].unique()
                    filter_models = st.multiselect(
                        "Фильтровать по моделям",
                        options=models_in_data,
                        default=list(models_in_data),
                        key=f"dashboard_filter_models_{selected_collection}",
                    )
                    df_filtered = df_collection[
                        df_collection["model"].isin(filter_models)
                    ]
                else:
                    df_filtered = df_collection

                st.dataframe(df_filtered)
                if not df_filtered.empty:
                    csv = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать результаты в CSV",
                        data=csv,
                        file_name=f"{selected_collection}_results.csv",
                        mime="text/csv",
                        key=f"download_csv_{selected_collection}",
                    )
    else:
        st.info("Нет доступных коллекций для просмотра.")


def render_tasks_visualization_tab():
    st.header("Визуализация по задачам")
    df_tasks = db_client.get_all_tasks()
    df_tasks = filter_tasks_by_group(df_tasks)
    if df_tasks.empty:
        st.write("Нет задач в базе для выбранной группы или вообще.")
    else:
        display_task_summary(df_tasks)
        st.dataframe(
            df_tasks[
                [
                    "task_name",
                    "dataset_name",
                    "group",
                    "metric",
                    "models",
                    "regexp",
                    "prompt",
                    "rta_prompt",
                ]
            ]
        )
        render_update_task()
        render_progressbar()
