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
    with st.expander("Изменить задачу", expanded=False):
        st.header("Изменить задачу")
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

        # Обновление моделей
        current_models = task_to_update.get("models", [])
        selected_models = st.multiselect(
            "Выберите модели для задачи:", options=MODELS, default=current_models
        )

        # Обновление prompt
        current_prompt = task_to_update.get("prompt", "")
        new_prompt = st.text_area(
            "Обновить prompt задачи:", value=current_prompt, height=150
        )

        # Обновление regexp и target
        current_regexp = task_to_update.get("regexp", "")
        new_regexp = st.text_input(
            "Обновить регулярное выражение для задачи:", value=current_regexp
        )
        current_target = task_to_update.get("target", "")
        new_target = st.text_input("Обновить target для задачи:", value=current_target)

        # Проверка наличия плейсхолдеров для переменных
        var_cols = task_to_update.get("variables_cols", [])
        if var_cols:
            missing_placeholders = [
                col for col in var_cols if f"{{{col}}}" not in new_prompt
            ]
            if missing_placeholders:
                st.warning(
                    "В prompt отсутствуют плейсхолдеры: "
                    + ", ".join(missing_placeholders)
                )

        # Обновление include/exclude колонок для метрики "include_exclude"
        if task_to_update.get("metric") == "include_exclude":
            current_include = task_to_update.get("include_column", "")
            new_include = st.text_input(
                "Обновить include_column для задачи:", value=current_include
            )
            current_exclude = task_to_update.get("exclude_column", "")
            new_exclude = st.text_input(
                "Обновить exclude_column для задачи:", value=current_exclude
            )
        else:
            new_include = None
            new_exclude = None

        # Обновление RTA prompt и модели для метрики "RtA"
        update_rta = False
        if task_to_update.get("metric") == "RtA":
            update_rta = True
            current_rta_prompt = task_to_update.get("rta_prompt", "")
            new_rta_prompt = st.text_area(
                "Обновить RTA prompt задачи:", value=current_rta_prompt, height=150
            )
            missing_placeholders = [
                col for col in ["input", "answer"] if f"{{{col}}}" not in new_rta_prompt
            ]
            if missing_placeholders:
                st.warning(
                    "В RTA prompt отсутствуют плейсхолдеры: "
                    + ", ".join(missing_placeholders)
                )
            current_rta_model = task_to_update.get("rta_model", "")
            new_rta_model = st.selectbox(
                "Обновить модель для RTA:",
                options=MODELS,
                index=MODELS.index(current_rta_model)
                if current_rta_model in MODELS
                else 0,
                key="update_rta_model_selectbox",
            )
        else:
            new_rta_prompt = None
            new_rta_model = None

        if st.button("Обновить задачу"):
            if not new_prompt:
                st.error("Prompt не может быть пустым!")
                return
            if update_rta and not new_rta_prompt:
                st.error("RTA prompt не может быть пустым для задач с метрикой RtA!")
                return

            # Формирование данных для обновления, синхронизированных с логикой создания задачи
            update_data = {
                "models": selected_models,
                "prompt": new_prompt,
                "regexp": new_regexp,
                "target": new_target,
            }
            if var_cols:
                update_data["variables_cols"] = var_cols
            if task_to_update.get("metric") == "include_exclude":
                update_data["include_column"] = new_include
                update_data["exclude_column"] = new_exclude
            if update_rta:
                update_data["rta_prompt"] = new_rta_prompt
                update_data["rta_model"] = new_rta_model

            task_id = task_to_update["_id"]
            db_client.update_task(task_id, update_data)
            # Получаем обновленную задачу (предполагается, что db_client имеет метод get_task)
            updated_task = db_client.get_task(task_id)
            # Вызываем одноразовую синхронизацию, используя объект базы данных из db_client
            sync_task_once(db_client.db, updated_task)
            st.success("Задача успешно обновлена и синхронизирована!")


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
                    for error_message, count in errors.items()[:5]:
                        st.write(
                            f"- **Ошибка:** {error_message} | **Количество:** {count}"
                        )
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
