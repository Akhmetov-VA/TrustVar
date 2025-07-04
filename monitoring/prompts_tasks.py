# prompts_jobs.py
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dataset_management import render_dataset_varcols_section
from utils.constants import MODELS, RTA_MODEL, AUGMENTATIONS, TASKS
from utils.db_client import MongoDBClient, MongoDBConfig

# Инициализация клиента БД
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

DEFAULT_REGEX = r"(?:^\W*([01]).*)|(?:.*([01])\W*$)"

def show_all_prompts() -> None:
    """Показать все промпты из базы."""
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

def insert_prompt_global(name: str, prompt: str) -> None:
    coll_name = "prompt_storage"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "prompt": prompt})

def show_all_rta_prompts() -> None:
    show_all_prompts()

def render_prompt_creation_section(var_cols: List[str]) -> Optional[str]:
    """UI для создания нового промпта."""
    hint = f"Вы можете использовать любые выбранные колонки: {', '.join('{' + c + '}' for c in var_cols)}."
    st.write(hint)
    prompt_name = st.text_input("Введите имя нового промпта:")
    user_prompt = st.text_area("Введите свой промпт:", value=hint)
    if user_prompt and prompt_name:
        missing_cols = [c for c in var_cols if f"{{{c}}}" not in user_prompt]
        if missing_cols:
            st.error("Отсутствуют плейсхолдеры: " + ", ".join(missing_cols))
        else:
            if prompt_exists(prompt_name):
                st.warning(f"Промпт с именем '{prompt_name}' уже существует. Вы можете использовать его.")
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
    """UI для выбора или создания промпта."""
    selected_prompt = None
    with st.expander("Выбор или создание промпта", expanded=False):
        show_all_prompts()
        use_existing_prompt = st.radio("Промпт:", ("Выбрать из базы", "Ввести свой"))
        all_prompt_docs = get_all_prompts()
        if use_existing_prompt == "Выбрать из базы":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox("Выберите промпт по имени:", names, key="prompt_selectbox")
                for p in all_prompt_docs:
                    if p["name"] == selected_name:
                        selected_prompt = p["prompt"]
                        break
                if selected_prompt:
                    for c in var_cols:
                        if f"{{{c}}}" not in selected_prompt:
                            st.warning(f"В промпте не найден плейсхолдер для колонки {c}")
            else:
                st.write("Нет доступных промптов. Введите свой.")
        else:
            selected_prompt = render_prompt_creation_section(var_cols)
    return selected_prompt

def show_existing_regexp(metric: str) -> None:
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

def insert_regexp_global(name: str, pattern: str, metric: str) -> None:
    coll_name = f"regexp_{metric}"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "pattern": pattern, "metric": metric})

def render_regexp_section(metric: str) -> Optional[str]:
    selected_regexp = None
    with st.expander("Выбор или создание регулярки для метрики", expanded=False):
        show_existing_regexp(metric)
        use_existing_regexp = st.radio("Регулярка:", ("Существующая", "Своя"))
        if use_existing_regexp == "Существующая":
            regexps = get_all_regexps_for_metric(metric)
            if regexps:
                names = [r["name"] for r in regexps]
                selected_name = st.selectbox("Выберите регулярку по имени:", names, key="regexp_selectbox")
                for r in regexps:
                    if r["name"] == selected_name:
                        selected_regexp = r["pattern"]
                        break
            else:
                st.write("Нет доступных регулярок для этой метрики.")
        else:
            st.write(f"По умолчанию предлагаем: {DEFAULT_REGEX}")
            custom_regexp = st.text_input("Введите свою регулярку:", value=DEFAULT_REGEX)
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
    rta_prompt_selected = None
    rta_model = None
    rta_target = None
    with st.expander("Выбор или создание RTA промпта", expanded=False):
        show_all_rta_prompts()
        st.write("Метрика RtA выбрана. Необходим RTA промпт.")
        use_rta_existing = st.radio("RTA промпт:", ("Выбрать из базы", "Ввести свой"))
        all_prompt_docs = get_all_prompts()
        if use_rta_existing == "Выбрать из базы":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox("Выберите RTA промпт по имени:", names, key="rta_prompt_selectbox")
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
        return st.multiselect("Выберите модели:", MODELS)

def render_dynamic_variations() -> List[str]:
    with st.expander("Динамическая аугментация датасета [AUG]", expanded=False):
        return st.multiselect("Выберите метод аугментации:", AUGMENTATIONS)

def build_task_data(
    task_type: str,
    task_name: str,
    dataset_name: str,
    var_cols: List[str],
    prompt: str,
    regexp: str,
    target_value: Any,
    models: List[str],
    metric: str,
    variations: Optional[List[str]],
    rta_prompt: Optional[str],
    rta_model: Optional[str],
    include_column: Optional[str],
    exclude_column: Optional[str],
    group_name: str = "default",
) -> Dict[str, Any]:
    data = {
        "task_type": task_type,
        "task_name": task_name,
        "dataset_name": dataset_name,
        "prompt": prompt,
        "variables_cols": var_cols,
        "models": models,
        "metric": metric,
        "regexp": regexp,
        "group": group_name,
    }
    if metric == "RtA":
        data["rta_prompt"] = rta_prompt
        data["rta_model"] = rta_model
        data["target"] = target_value
    elif metric == "include_exclude":
        data["include_column"] = include_column
        data["exclude_column"] = exclude_column
    else:
        data["target"] = target_value
    data["dynamic_augments"] = variations if variations else []
    return data

def render_preview_and_save_task(
    task_type: str,
    dataset_name: str,
    var_cols: List[str],
    selected_prompt: str,
    selected_regexp: str,
    target_value: Any,
    selected_models: List[str],
    metric: str,
    selected_variations: Optional[List[str]],
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
            task_name = st.text_input("Task Name:", value=f"{dataset_name}")
            group_name = st.text_input("Task Group:", value="default")
            st.subheader("5 Random Sample Preview:")
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
                    st.write(f"**Example {i + 1}:** {filled_prompt}")
            if selected_variations:
                st.write(f"**Методы динамической аугментации:** {' | '.join(selected_variations)}")
            st.write("**DB Record Structure:**")
            task_data = build_task_data(
                task_type=task_type,
                task_name=task_name,
                dataset_name=dataset_name,
                var_cols=var_cols,
                prompt=selected_prompt,
                regexp=selected_regexp,
                target_value=target_value,
                models=selected_models,
                metric=metric,
                variations=selected_variations,
                rta_prompt=rta_prompt_selected,
                rta_model=rta_model,
                include_column=include_column,
                exclude_column=exclude_column,
                group_name=group_name,
            )
            st.json(task_data, expanded=False)
            if st.button("Upload task"):
                db_client.insert_task(task_data)
                st.success("Task was uploaded successfully!")

def render_create_task_tab():
    st.header("Create new task")
    all_datasets = db_client.get_all_datasets()
    if "regestry" in all_datasets:
        all_datasets.remove("regestry")
    task_type = st.selectbox("Task type:", TASKS, key="select_task_type_selectbox")
    selected_dataset = st.selectbox("Select dataset:", sorted(all_datasets), key="select_ds_selectbox")
    if task_type and selected_dataset:
        var_cols, metric, target_column, include_column, exclude_column = render_dataset_varcols_section(selected_dataset)
        if task_type.lower() == "compare model behaviour":
            selected_variations = render_dynamic_variations()
        else:
            selected_variations = None
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
                        rta_prompt_selected, rta_model, rta_target_value = render_rta_prompt_section()
                    selected_models = render_models_section()
                    final_target = rta_target_value if metric == "RtA" else target_column
                    render_preview_and_save_task(
                        task_type=task_type,
                        dataset_name=selected_dataset,
                        var_cols=var_cols,
                        selected_prompt=selected_prompt,
                        selected_regexp=selected_regexp,
                        target_value=final_target,
                        selected_models=selected_models,
                        metric=metric,
                        selected_variations=selected_variations,
                        rta_prompt_selected=rta_prompt_selected,
                        rta_model=rta_model,
                        include_column=include_column,
                        exclude_column=exclude_column,
                    )
