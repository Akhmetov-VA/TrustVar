FROM python:3.11.0 as base
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install "uv>=0.5.29"

#langchain-backend
FROM base as back
RUN uv pip install -r pyproject.toml --extra backend --system
COPY ./langchain_back /app/langchain_back/

#bd_filler
FROM base as db_filler
RUN uv pip install -r pyproject.toml --extra db --system
COPY ./utils /app/utils/
COPY ./data /app/data/

#streamlit-frontend
FROM base as front
RUN uv pip install -r pyproject.toml --extra frontend --system
COPY ./utils /app/utils/
COPY ./monitoring /app/monitoring/

#runners
FROM base as runners
RUN uv pip install -r pyproject.toml --extra db --system
COPY ./benchmark/runners /app/runners/
COPY ./utils /app/utils/
COPY ./data /app/data/