import json
import requests
import streamlit as st
import os
import utils.constants
from dotenv import load_dotenv
import logging
import pandas as pd

load_dotenv()
OLLAMA_PULL_URL = os.getenv('OLLAMA_BASE_URL') + "/api/pull"
MODELS = utils.constants.MODELS
# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_model_management_tab():
    st.set_page_config(page_title="Model Loader", page_icon="🧩")

    if "results" not in st.session_state:
        st.session_state.results = {"success": [], "failed": {}}
    
    if "MODELS" not in st.session_state:
        st.session_state.MODELS = MODELS 

    st.title("🧩 Model Loader for Ollama")

    # ========= Expander: View available models =========
    with st.expander("View available models", expanded=False):
        df = pd.DataFrame({"Model": MODELS})
        df.index = range(1, len(df) + 1)
        df.index.name = "№"

        st.caption(f"Models num: {len(MODELS)}")
        st.dataframe(df, use_container_width=True)

    # ========= Вспомогательные функции =========
    def human_bytes(n: int) -> str:
        if n is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    def pull_model(model: str, progress, status_placeholder):
        """
        Тянет модель через Ollama /api/pull со streaming-прогрессом.
        Обновляет progress и status_placeholder.
        Возвращает (ok: bool, error_message: str|None)
        """
        try:
            with requests.post(
                OLLAMA_PULL_URL,
                json={"model": model, "stream": True},
                stream=True,
                timeout=600,  # увеличьте при медленном интернете
            ) as r:
                r.raise_for_status()

                ok = False
                last_status = "Начало загрузки..."
                total = None
                completed = 0
                progress.progress(0)
                status_placeholder.info(last_status)

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Сообщение об ошибке от сервера
                    if "error" in evt:
                        msg = evt.get("error") or "Unknown error"
                        status_placeholder.error(msg)
                        progress.progress(0)
                        return False, msg

                    # Обновляем текст статуса
                    if "status" in evt and evt["status"]:
                        last_status = evt["status"]

                    # Обновляем прогресс по байтам, если есть completed/total
                    if isinstance(evt.get("total"), int) and isinstance(evt.get("completed"), int):
                        total = evt["total"] or total
                        completed = evt["completed"]
                        if total and total > 0:
                            frac = max(0.0, min(1.0, completed / total))
                            progress.progress(int(frac * 100))
                            status_placeholder.info(f"{last_status} — {human_bytes(completed)} / {human_bytes(total)}")
                        else:
                            status_placeholder.info(last_status)
                    else:
                        status_placeholder.info(last_status)

                    # Признак финального успеха
                    
                    if evt.get("status") == 'success':
                        ok = True
                logger.info(evt)
                if ok:
                    progress.progress(100)
                    status_placeholder.success("Готово ✅")
                    return True, None
                else:
                    msg = "Поток завершился без признака успеха"
                    status_placeholder.error(msg)
                    return False, msg

        except requests.exceptions.RequestException as e:
            msg = f"HTTP/сетевой сбой: {e}"
            status_placeholder.error(msg)
            progress.progress(0)
            return False, msg

    # ========= Expander: Load new models =========
    with st.expander("Load new models", expanded=False):
        user_input = st.text_area(
            "Input ollama model names (one per line)",
            height=160,
            placeholder="llama3\nmistral:instruct\nphi3",
        )
        load = st.button("LOAD", type="primary", use_container_width=True)

        if load:
            items = [ln.strip() for ln in user_input.splitlines() if ln.strip()]
            items = list(dict.fromkeys(items))  # deduplicate

            if not items:
                st.warning("Please specify at least one model.")
            else:
                st.session_state.results = {"success": [], "failed": {}}
                st.write(f"Loading {len(items)} model(s):")

                for model in items:
                    block = st.container()
                    with block:
                        st.markdown(f"**{model}**")
                        progress = st.progress(0)
                        status_placeholder = st.empty()

                    ok, err = pull_model(model, progress, status_placeholder)
                    if ok:
                        st.session_state.results["success"].append(model)
                        MODELS.append(model)
                        st.session_state.MODELS = MODELS
                        utils.constants.MODELS[:] = MODELS # for changing in other places
                        st.rerun()
                    else:
                        st.session_state.results["failed"][model] = err or "unknown error"

    # ========= Результаты (если есть) =========
    if st.session_state.results["success"] or st.session_state.results["failed"]:
        st.subheader("Loading results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("✅ Successfully loaded")
            if st.session_state.results["success"]:
                for m in st.session_state.results["success"]:
                    st.success(m, icon="✅")
            else:
                st.info("Empty")

        with col2:
            st.markdown("❌ Loading failed")
            if st.session_state.results["failed"]:
                for m, err in st.session_state.results["failed"].items():
                    st.error(f"{m} — {err}", icon="❌")
            else:
                st.info("Empty")