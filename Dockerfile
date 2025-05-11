FROM python:3.11.0 as base
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml /app/
COPY README.md /app/README.md
COPY requirements.txt /app/requirements.txt
RUN apt update && apt install -y iproute2 iputils-ping curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install "uv>=0.5.29"

#langchain-backend
FROM base as back
RUN uv pip install -r requirements.txt --system
COPY ./langchain_back /app/langchain_back/

#streamlit-frontend
FROM base as front
RUN uv pip install -r requirements.txt --system
COPY ./utils /app/utils/
COPY ./monitoring /app/monitoring/

#runners
FROM base as runners
RUN uv pip install -r requirements.txt --system
COPY ./benchmark/runners /app/runners/
COPY ./utils /app/utils/
COPY ./data /app/data/
