FROM python:3.11.0 as base
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml /app/
COPY README.md /app/README.md
RUN apt update && apt install -y iproute2 iputils-ping curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install "uv>=0.5.29" && uv pip install -r pyproject.toml --system

#langchain-backend
FROM base as back
RUN uv pip install -r pyproject.toml --extra backend --system
COPY ./langchain_back /app/langchain_back/
COPY ./utils /app/utils/
COPY ./data /app/data/

#streamlit-frontend
FROM base as front
RUN uv pip install -r pyproject.toml --extra frontend --system
COPY ./utils /app/utils/
COPY ./monitoring /app/monitoring/

#runners
FROM base as runners
RUN uv pip install -r pyproject.toml --extra runners --system
COPY ./benchmark/runners /app/runners/
COPY ./utils /app/utils/
