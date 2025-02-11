import logging
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from monitoring.src import load_file_any_format
from utils.constants import METRICS, MODELS, RTA_MODEL, STATUSES
from utils.db_client import MongoDBClient, MongoDBConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация клиента БД
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

st.set_page_config(page_title="Trust LLM Dashboard", layout="wide")

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
    with st.expander("Обновить задачу", expanded=False):
        st.header("Обновить задачу")
        df_tasks = db_client.get_all_tasks()
        if df_tasks.empty:
            st.write("Нет задач для обновления.")
            return

        df_tasks = filter_tasks_by_group(df_tasks)
        if df_tasks.empty:
            st.write("Нет задач в выбранной группе для обновления.")
            return

        task_names = df_tasks["task_name"].unique().tolist()
        selected_task_name = st.selectbox(
            "Выберите задачу:", task_names, key="update_task_selectbox"
        )
        task_to_update = df_tasks[df_tasks["task_name"] == selected_task_name].iloc[0]

        current_models = task_to_update.get("models", [])
        selected_models = st.multiselect(
            "Выберите модели для задачи:", options=MODELS, default=current_models
        )

        current_prompt = task_to_update.get("prompt", "")
        var_cols = task_to_update.get("variables_cols", [])
        new_prompt = st.text_area(
            "Обновить prompt задачи:", value=current_prompt, height=150
        )

        if var_cols:
            missing_placeholders = [
                col for col in var_cols if f"{{{col}}}" not in new_prompt
            ]
            if missing_placeholders:
                st.warning(
                    "В prompt отсутствуют плейсхолдеры: "
                    + ", ".join(missing_placeholders)
                )

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
                    "В prompt отсутствуют плейсхолдеры: "
                    + ", ".join(missing_placeholders)
                )

        if st.button("Обновить задачу"):
            if not new_prompt:
                st.error("Prompt не может быть пустым!")
                return
            if update_rta and not new_rta_prompt:
                st.error("RTA prompt не может быть пустым для задач с метрикой RtA!")
                return

            update_data = {"models": selected_models, "prompt": new_prompt}
            if update_rta:
                update_data["rta_prompt"] = new_rta_prompt

            task_id = task_to_update["_id"]
            db_client.update_task(task_id, update_data)
            st.success("Задача успешно обновлена!")


def highlight_status(s: str) -> str:
    if s == "Ошибка":
        return "background-color: red; color: white;"
    elif s == "В процессе":
        return "background-color: orange; color: white;"
    elif s == "Завершено":
        return "background-color: green; color: white;"
    else:
        return ""


def restart_failed_tasks(db_client: MongoDBClient, collection_name: str) -> int:
    count = db_client.update_tasks_status(collection_name, "failed", "pending")
    return count


def fail_pending_tasks(db_client: MongoDBClient, collection_name: str) -> int:
    """
    Обновляет задачи со статусом 'pending' на 'failed'
    """
    count = db_client.update_tasks_status(collection_name, "pending", "failed")
    return count


def load_data_for_dashboard(
    db_client: MongoDBClient, collections: List[str]
) -> pd.DataFrame:
    data = []
    exclude = ["delete_me", "test"]
    collections = [c for c in collections if c not in exclude]

    for collection_name in collections:
        total_tasks = db_client.count_total_tasks(collection_name)
        statuses = [
            "pending",
            "completed",
            "failed",
            "extracted",
            "processing",
            "transfered_to_rta",
        ]
        status_counts = {
            status: db_client.count_tasks_by_status(collection_name, status)
            for status in statuses
        }

        if status_counts["failed"] > 0:
            status = "Ошибка"
        elif status_counts["pending"] > 0 or status_counts["processing"] > 0:
            status = "В процессе"
        else:
            status = "Завершено"

        data_row = {
            "Коллекция": collection_name,
            "Всего задач": total_tasks,
            "В ожидании": status_counts["pending"],
            "Выполнено": status_counts["completed"]
            + status_counts["transfered_to_rta"],
            "Измерено": status_counts["extracted"],
            "С ошибками": status_counts["failed"],
            "Статус": status,
        }
        data.append(data_row)

    return pd.DataFrame(data)


def show_errors(db_client: MongoDBClient, collections: List[str]):
    st.header("Уникальные сообщения об ошибках")
    with st.expander("Показать ошибки", expanded=False):
        for collection_name in collections:
            failed_tasks = db_client.get_tasks_by_status(collection_name, "failed")
            if failed_tasks:
                error_messages = [
                    task.get("error", "Нет информации об ошибке")
                    for task in failed_tasks
                ]
                error_counts = Counter(error_messages)
                st.subheader(f"Коллекция: {collection_name}")
                for error_message, count in error_counts.items():
                    st.write(f"**Ошибка:** {error_message} | **Количество:** {count}")
                st.write("---")

        if st.button("Перезапустить задачи с ошибками", key="restart_failed_tasks"):
            for collection_name in collections:
                modified_count = restart_failed_tasks(db_client, collection_name)
                if modified_count > 0:
                    st.write(
                        f"В коллекции '{collection_name}' перезапущено {modified_count} задач."
                    )
        else:
            st.write("Нажмите кнопку выше, чтобы перезапустить все задачи с ошибками.")


def render_progressbar():
    st.header("Мониторинг очередей")
    collections_to_process = sorted(
        [col for col in db_client.list_collections() if col.startswith("queue_")]
    )
    df = load_data_for_dashboard(db_client, collections_to_process)

    if st.button("Обновить таблицу", key="refresh_dashboard"):
        df = load_data_for_dashboard(db_client, collections_to_process)

    if not df.empty:
        df = df.sort_values("Коллекция").reset_index(drop=True)
        df_style = df.style.applymap(highlight_status, subset=["Статус"])
        st.write(df_style)

        # Новый экспандер: Перевод задач из pending в failed
        # Определяем коллекции, в которых есть задачи со статусом pending
        pending_queues = [
            col
            for col in collections_to_process
            if db_client.count_tasks_by_status(col, "pending") > 0
        ]
        if pending_queues:
            with st.expander("Перевести задачи в failed", expanded=False):
                selected_queue = st.selectbox(
                    "Выберите очередь (collection):",
                    pending_queues,
                    key="fail_pending_selectbox",
                )
                if st.button(
                    "Поменять статус задач на 'failed'",
                    key="fail_pending_button",
                ):
                    count_failed = fail_pending_tasks(db_client, selected_queue)
                    st.success(
                        f"В коллекции '{selected_queue}' обновлено {count_failed} задач."
                    )

        if df["С ошибками"].sum() > 0:
            show_errors(db_client, collections_to_process)
    else:
        st.info("Нет данных для отображения в очередях.")

    st.header("Просмотр данных коллекции")
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
            df_tasks[["task_name", "dataset_name", "group", "metric", "models"]]
        )
        render_update_task()
        render_progressbar()


def render_dataset_registry_section():
    st.subheader("Содержимое dataset_regestry")
    coll = db_client.get_collection("dataset_regestry")
    docs = list(coll.find({}))
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        st.dataframe(df)
    else:
        st.write("dataset_regestry пуст.")


def render_dataset_upload_section() -> Optional[str]:
    with st.expander("Добавить новый датасет", expanded=False):
        st.write("Вы можете загрузить CSV, Excel, JSON или Parquet файл.")
        uploaded_file = st.file_uploader(
            "Загрузите CSV, Excel, JSON или Parquet файл",
            type=["csv", "xlsx", "json", "parquet"],
            key="file_uploader_experiments",
        )

        if uploaded_file is not None:
            dataset_name_input = st.text_input(
                "Введите имя нового датасета (латиницей):",
                value=uploaded_file.name.split(".")[0],
            )
        if uploaded_file is not None and dataset_name_input:
            df_uploaded = load_file_any_format(uploaded_file)
            if df_uploaded is not None and not df_uploaded.empty:
                st.write("Некоторые строки загруженного датасета (случайные 10 строк):")
                st.dataframe(df_uploaded.sample(min(10, len(df_uploaded))))
                chosen_metric = st.selectbox(
                    "Выберите метрику для этого датасета:",
                    METRICS,
                    key="dataset_upload_selectbox",
                )
                st.write(
                    "Выберите колонки, которые будут использоваться как переменные для промпта:"
                )
                var_cols = st.multiselect(
                    "Переменные для промпта:", list(df_uploaded.columns)
                )

                target_column = None
                include_col = None
                exclude_col = None

                if chosen_metric == "include_exclude":
                    st.write(
                        "Для метрики 'include_exclude' необходимо указать:\n"
                        "1) Колонку, где хранится список строк, которые должны присутствовать в ответе.\n"
                        "2) Опционально — колонку, где хранится список строк, которые не должны присутствовать."
                    )
                    potential_cols = [
                        c for c in df_uploaded.columns if c not in var_cols
                    ]
                    include_col = st.selectbox(
                        "Колонка со строками, которые должны присутствовать (include):",
                        potential_cols,
                        key="dataset_upload_include_selectbox",
                    )
                    exclude_col = st.selectbox(
                        "Колонка со строками, которые не должны присутствовать (exclude) (необязательно):",
                        [None] + potential_cols,
                        index=0,
                        key="dataset_upload_exclude_selectbox",
                    )
                else:
                    if chosen_metric != "RtA":
                        potential_targets = [
                            c for c in df_uploaded.columns if c not in var_cols
                        ]
                        if not potential_targets:
                            target_column = st.text_input(
                                "Введите название колонки с таргетом:"
                            )
                        else:
                            target_column = st.selectbox(
                                "Выберите колонку с таргетом:",
                                potential_targets,
                                key="dataset_upload_target_selectbox",
                            )
                st.subheader("Предпросмотр записи для сохранения:")
                record_preview = {
                    "dataset_name": dataset_name_input,
                    "var_cols": var_cols,
                    "metric": chosen_metric,
                    "target_column": target_column,
                    "include_column": include_col,
                    "exclude_column": exclude_col,
                }
                st.json(record_preview)
                if st.button("Сохранить датасет в БД"):
                    db_client.insert_dataset_records(dataset_name_input, df_uploaded)
                    db_client.insert_dataset_into_registry(record_preview)
                    st.success(
                        f"Датасет '{dataset_name_input}' загружен и зарегистрирован!"
                    )
                    return dataset_name_input
            else:
                st.error("Загруженный файл пуст или не может быть прочитан.")
    return None


def render_dataset_management_tab():
    st.header("Управление датасетами")
    render_dataset_registry_section()
    render_dataset_upload_section()


def render_dataset_varcols_section(
    dataset_name: str,
) -> Tuple[
    Optional[List[str]], Optional[str], Optional[str], Optional[str], Optional[str]
]:
    registry_info = db_client.get_dataset_registry_info(dataset_name)
    if not registry_info:
        st.write("Для этого датасета нет сохраненных var_cols, метрики или таргета.")
        return None, None, None, None, None
    else:
        var_cols = registry_info["var_cols"]
        chosen_metric = registry_info.get("metric", METRICS[0])
        target_column = registry_info.get("target_column", None)
        include_column = registry_info.get("include_column", None)
        exclude_column = registry_info.get("exclude_column", None)
        st.write(f"**Переменные для промпта (var_cols):** {var_cols}")
        st.write(f"**Метрика:** {chosen_metric}")
        st.write(f"**Таргет колонка:** {target_column}")
        st.write(f"**Колонка для include:** {include_column}")
        st.write(f"**Колонка для exclude:** {exclude_column}")
        return var_cols, chosen_metric, target_column, include_column, exclude_column


def show_all_prompts():
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        st.write("Нет промптов в хранилище.")
        return
    coll = db_client.get_collection(coll_name)
    prompts = list(coll.find({}))
    if prompts:
        df = pd.DataFrame(prompts)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        st.write("Существующие промпты (name, prompt):")
        st.dataframe(df)
    else:
        st.write("Нет промптов в хранилище.")


def get_all_prompts() -> List[Dict[str, Any]]:
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        return []
    coll = db_client.get_collection(coll_name)
    return list(coll.find({}))


def prompt_exists(name: str) -> bool:
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        return False
    coll = db_client.get_collection(coll_name)
    return coll.find_one({"name": name}) is not None


def insert_prompt_global(name: str, prompt: str):
    coll_name = "prompt_storage"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "prompt": prompt})


def show_all_rta_prompts():
    show_all_prompts()


def render_prompt_creation_section(var_cols: List[str]) -> Optional[str]:
    hint, placeholders = generate_prompt_hint(var_cols)
    st.write(hint)
    prompt_name = st.text_input("Введите имя нового промпта:")
    user_prompt = st.text_area("Введите свой промпт:", value=placeholders)
    if user_prompt and prompt_name:
        missing_cols = [c for c in var_cols if f"{{{c}}}" not in user_prompt]
        if missing_cols:
            st.error("Отсутствуют плейсхолдеры: " + ", ".join(missing_cols))
        else:
            if prompt_exists(prompt_name):
                st.warning(
                    f"Промпт с именем '{prompt_name}' уже существует. Вы можете использовать его."
                )
                if st.button("Использовать существующий промпт"):
                    for p in get_all_prompts():
                        if p["name"] == prompt_name:
                            return p["prompt"]
            else:
                if st.button("Добавить промпт в базу"):
                    insert_prompt_global(prompt_name, user_prompt)
                    st.success("Промпт добавлен!")
                    return user_prompt
    return None


def render_prompt_selection_section(var_cols: List[str]) -> Optional[str]:
    with st.expander("Выбор или создание промпта", expanded=False):
        show_all_prompts()
        use_existing_prompt = st.radio("Промпт:", ("Выбрать из базы", "Ввести свой"))
        selected_prompt = None
        all_prompt_docs = get_all_prompts()
        if use_existing_prompt == "Выбрать из базы":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox(
                    "Выберите промпт по имени:", names, key="prompt_selectbox"
                )
                for p in all_prompt_docs:
                    if p["name"] == selected_name:
                        selected_prompt = p["prompt"]
                        break
                if selected_prompt:
                    for c in var_cols:
                        if f"{{{c}}}" not in selected_prompt:
                            st.warning(
                                f"В промпте не найден плейсхолдер для колонки {c}"
                            )
            else:
                st.write("Нет доступных промптов. Введите свой.")
        else:
            selected_prompt = render_prompt_creation_section(var_cols)
        return selected_prompt


def show_existing_regexp(metric: str):
    coll_name = f"regexp_{metric}"
    if coll_name not in db_client.list_collections():
        st.write("Нет регулярок для данной метрики.")
        return
    coll = db_client.get_collection(coll_name)
    docs = list(coll.find({}))
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        st.write("Существующие регулярки (name, pattern, metric):")
        st.dataframe(df)
    else:
        st.write("Нет регулярок для данной метрики.")


def get_all_regexps_for_metric(metric: str) -> List[Dict[str, Any]]:
    coll_name = f"regexp_{metric}"
    if coll_name not in db_client.list_collections():
        return []
    coll = db_client.get_collection(coll_name)
    return list(coll.find({}))


def insert_regexp_global(name: str, pattern: str, metric: str):
    coll_name = f"regexp_{metric}"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "pattern": pattern, "metric": metric})


def render_regexp_section(metric: str) -> Optional[str]:
    with st.expander("Выбор или создание регулярки для метрики", expanded=False):
        show_existing_regexp(metric)
        use_existing_regexp = st.radio("Регулярка:", ("Существующая", "Своя"))
        selected_regexp = None
        if use_existing_regexp == "Существующая":
            regexps = get_all_regexps_for_metric(metric)
            if regexps:
                names = [r["name"] for r in regexps]
                selected_name = st.selectbox(
                    "Выберите регулярку по имени:", names, key="regexp_selectbox"
                )
                for r in regexps:
                    if r["name"] == selected_name:
                        selected_regexp = r["pattern"]
                        break
            else:
                st.write("Нет доступных регулярок для этой метрики.")
        else:
            st.write(f"По умолчанию предлагаем: {DEFAULT_REGEX}")
            custom_regexp = st.text_input(
                "Введите свою регулярку:", value=DEFAULT_REGEX
            )
            if custom_regexp:
                if db_client.validate_regex(custom_regexp):
                    regexp_name = st.text_input("Введите имя для этой регулярки:")
                    if regexp_name and st.button("Добавить регулярку в базу"):
                        insert_regexp_global(regexp_name, custom_regexp, metric)
                        st.success("Регулярка добавлена!")
                        selected_regexp = custom_regexp
                else:
                    st.error("Неверное регулярное выражение!")
        return selected_regexp


def render_rta_prompt_section() -> Tuple[Optional[str], Optional[str], Any]:
    with st.expander("Выбор или создание RTA промпта", expanded=False):
        show_all_rta_prompts()
        st.write("Метрика RtA выбрана. Необходим RTA промпт.")
        use_rta_existing = st.radio("RTA промпт:", ("Выбрать из базы", "Ввести свой"))
        rta_prompt_selected = None
        all_prompt_docs = get_all_prompts()
        if use_rta_existing == "Выбрать из базы":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox(
                    "Выберите RTA промпт по имени:", names, key="rta_prompt_selectbox"
                )
                for rp in all_prompt_docs:
                    if rp["name"] == selected_name:
                        rta_prompt_selected = rp["prompt"]
                        break
            else:
                st.write("Нет доступных RTA промптов. Введите свой.")
        else:
            rta_prompt_selected = render_prompt_creation_section(var_cols=[])
        rta_target = st.text_input("Целевое значение для RtA:", value="1")
        rta_model = st.selectbox(
            "Модель для RTA:",
            MODELS,
            index=MODELS.index(RTA_MODEL) if RTA_MODEL in MODELS else 0,
            key="rta_model_selectbox",
        )
        return rta_prompt_selected, rta_model, rta_target


def render_models_section() -> List[str]:
    with st.expander("Выбор моделей для задачи", expanded=False):
        selected_models = st.multiselect("Выберите модели:", MODELS)
        return selected_models


def render_preview_and_save_task(
    dataset_name: str,
    var_cols: List[str],
    selected_prompt: str,
    selected_regexp: str,
    target_value: Any,
    selected_models: List[str],
    metric: str,
    rta_prompt_selected: Optional[str],
    rta_model: Optional[str],
    include_column: Optional[str],
    exclude_column: Optional[str],
):
    with st.expander("Предпросмотр и сохранение задачи", expanded=False):
        if (
            selected_prompt
            and selected_regexp
            and selected_models
            and (target_value or metric in ["RtA", "include_exclude"])
        ):
            group_name = st.text_input("Группа задачи (group):", value="default")
            task_name = st.text_input("Имя задачи:", value=f"{dataset_name}")
            st.subheader("Предпросмотр 5 случайных примеров:")
            df_head = db_client.get_dataset_head(dataset_name, limit=100)
            if not df_head.empty:
                sample_size = min(5, len(df_head))
                preview_data = (
                    df_head[var_cols].sample(n=sample_size).to_dict(orient="records")
                )
                for i, row in enumerate(preview_data):
                    filled_prompt = selected_prompt
                    for k, v in row.items():
                        filled_prompt = filled_prompt.replace(f"{{{k}}}", str(v))
                    st.write(f"**Пример {i + 1}:** {filled_prompt}")
            st.write("**Структура записи задачи в БД:**")
            task_data = {
                "task_name": task_name,
                "dataset_name": dataset_name,
                "prompt": selected_prompt,
                "variables_cols": var_cols,
                "models": selected_models,
                "metric": metric,
                "regexp": selected_regexp,
                "group": group_name,
            }
            if metric == "RtA":
                task_data["rta_prompt"] = rta_prompt_selected
                task_data["rta_model"] = rta_model
                task_data["target"] = target_value
            elif metric == "include_exclude":
                task_data["include_column"] = include_column
                task_data["exclude_column"] = exclude_column
            else:
                task_data["target"] = target_value

            st.json(task_data, expanded=False)
            if st.button("Загрузить задачу в базу"):
                db_client.insert_task(task_data)
                st.success("Задача успешно добавлена!")


def render_create_task_tab():
    st.header("Создать новую задачу")
    all_datasets = db_client.get_all_datasets()
    if "regestry" in all_datasets:
        all_datasets.remove("regestry")
    selected_dataset = st.selectbox(
        "Выберите датасет:", sorted(all_datasets), key="create_task_selectbox"
    )
    if selected_dataset:
        var_cols, metric, target_column, include_column, exclude_column = (
            render_dataset_varcols_section(selected_dataset)
        )
        if var_cols and metric is not None:
            selected_prompt = render_prompt_selection_section(var_cols)
            if selected_prompt:
                if metric != "include_exclude":
                    selected_regexp = render_regexp_section(metric)
                else:
                    selected_regexp = "Метрика include_exclude не использует regexp."
                if selected_regexp:
                    rta_prompt_selected = None
                    rta_model = None
                    rta_target_value = None
                    if metric == "RtA":
                        rta_prompt_selected, rta_model, rta_target_value = (
                            render_rta_prompt_section()
                        )
                    selected_models = render_models_section()
                    final_target = (
                        rta_target_value if metric == "RtA" else target_column
                    )
                    render_preview_and_save_task(
                        dataset_name=selected_dataset,
                        var_cols=var_cols,
                        selected_prompt=selected_prompt,
                        selected_regexp=selected_regexp,
                        target_value=final_target,
                        selected_models=selected_models,
                        metric=metric,
                        rta_prompt_selected=rta_prompt_selected,
                        rta_model=rta_model,
                        include_column=include_column,
                        exclude_column=exclude_column,
                    )


def render_metrics_tab():
    st.header("Метрики моделей")
    results_collections = ["RtAR", "TFNR", "Accuracy", "Correlation", "IncludeExclude"]
    if results_collections:
        selected_results_collection = st.selectbox(
            "Выберите коллекцию с метриками",
            options=results_collections,
            key="metrics_collection_selection",
        )
        results_collection = db_client.get_collection(selected_results_collection)
        results_data = list(results_collection.find())
        if results_data:
            visualize_metrics(results_data, selected_results_collection)
        else:
            st.info(f"Данные в коллекции '{selected_results_collection}' отсутствуют.")
    else:
        st.info("Нет доступных коллекций с метриками.")


def visualize_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    required_cols = {"dataset_name", "model", "value"}
    if not required_cols.issubset(results_df.columns):
        st.error("В данных отсутствуют необходимые поля (dataset_name, model, value).")
        return
    datasets = results_df["dataset_name"].unique()
    models = results_df["model"].unique()
    selected_datasets = st.multiselect(
        "Выберите датасеты",
        options=datasets,
        default=list(datasets),
        key=f"metrics_datasets_{collection_name}",
    )
    selected_models = st.multiselect(
        "Выберите модели",
        options=models,
        default=list(models),
        key=f"metrics_models_{collection_name}",
    )
    filtered_df = results_df[
        (results_df["dataset_name"].isin(selected_datasets))
        & (results_df["model"].isin(selected_models))
    ]
    if filtered_df.empty:
        st.info("Нет данных для отображения с выбранными фильтрами.")
        return
    pivot_table = filtered_df.pivot_table(
        index="model", columns="dataset_name", values="value", aggfunc="mean"
    )
    st.subheader("Таблица метрик по датасетам и моделям")
    st.dataframe(pivot_table)
    st.subheader("Визуализация метрик")
    st.bar_chart(pivot_table)


# Основной интерфейс
tabs = st.tabs(
    [
        "Визуализация по задачам",
        "Управление датасетами",
        "Создать задачу",
        "Метрики моделей",
    ]
)

with tabs[0]:
    render_tasks_visualization_tab()

with tabs[1]:
    render_dataset_management_tab()

with tabs[2]:
    render_create_task_tab()

with tabs[3]:
    render_metrics_tab()
