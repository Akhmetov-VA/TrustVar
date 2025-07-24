# Развертывание TrustVar

## Обзор развертывания

TrustVar поддерживает несколько способов развертывания:
- **Docker Compose** (рекомендуется для продакшена)
- **Локальная разработка** с Poetry
- **Kubernetes** (для масштабируемых развертываний)

## 1. Развертывание с Docker Compose

### Предварительные требования

- Docker Engine 20.10+
- Docker Compose 2.0+
- Минимум 8GB RAM
- 50GB свободного места на диске
- NVIDIA GPU (опционально, для локальных моделей)

### Шаг 1: Подготовка окружения

1. **Клонируйте репозиторий:**
   ```bash
   git clone <repository-url>
   cd TrustVar
   ```

2. **Создайте файл `.env`:**
   ```bash
   cp .env.example .env
   ```

3. **Настройте переменные окружения в `.env`:**
   ```env
   # MongoDB
   MONGO_INITDB_ROOT_USERNAME=admin
   MONGO_INITDB_ROOT_PASSWORD=secure_password
   MONGO_INITDB_ROOT_PORT=27017
   
   # API Keys
   YANDEX_API_KEY=your_yandex_api_key
   YANDEX_MODEL_URI=your_yandex_model_uri
   YANDEX_BASE_URL=https://llm.api.cloud.yandex.net
   
   OPENAI_KEY=your_openai_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1
   
   # URLs
   API_URL=http://localhost:45321/generate
   OLLAMA_BASE_URL=http://localhost:12345
   
   # User permissions for Docker
   CURRENT_UID=1000
   CURRENT_GID=1000
   ```

### Шаг 2: Настройка GPU (опционально)

Если у вас есть NVIDIA GPU и вы хотите использовать локальные модели:

1. **Установите NVIDIA Container Toolkit:**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Проверьте установку:**
   ```bash
   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Настройте путь к моделям в `docker-compose.yml`:**
   ```yaml
   ollama:
     volumes:
       - /path/to/your/models:/models:ro  # Измените на ваш путь
   ```

### Шаг 3: Запуск системы

1. **Соберите и запустите все сервисы:**
   ```bash
   docker-compose up -d --build
   ```

2. **Проверьте статус сервисов:**
   ```bash
   docker-compose ps
   ```

3. **Просмотрите логи:**
   ```bash
   # Все сервисы
   docker-compose logs -f
   
   # Конкретный сервис
   docker-compose logs -f langchain_backend
   docker-compose logs -f streamlit_frontend
   ```

### Шаг 4: Проверка работоспособности

1. **Веб-интерфейс мониторинга:**
   - URL: http://localhost:27366
   - Логин/пароль: см. `monitoring/config.yaml`

2. **MongoDB Express:**
   - URL: http://localhost:8081
   - Логин/пароль: из переменных окружения

3. **API Backend:**
   - URL: http://localhost:45321
   - Health check: http://localhost:45321/health

### Шаг 5: Остановка системы

```bash
# Остановить все сервисы
docker-compose down

# Остановить и удалить volumes (данные)
docker-compose down -v

# Остановить и удалить images
docker-compose down --rmi all
```

## 2. Локальная разработка

### Предварительные требования

- Python 3.11+
- Poetry
- MongoDB (локально или в Docker)
- Node.js 18+ (для некоторых frontend компонентов)

### Шаг 1: Установка зависимостей

1. **Установите Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Установите зависимости проекта:**
   ```bash
   poetry install
   ```

3. **Активируйте виртуальное окружение:**
   ```bash
   poetry shell
   ```

### Шаг 2: Настройка MongoDB

**Вариант A: Локальная установка MongoDB**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Запуск
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

**Вариант B: MongoDB в Docker**
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:latest
```

### Шаг 3: Настройка переменных окружения

Создайте файл `.env` в корне проекта:
```env
MONGO_INITDB_ROOT_USERNAME=admin
MONGO_INITDB_ROOT_PASSWORD=password
MONGO_HOST=localhost
MONGO_PORT=27017
API_URL=http://localhost:45321/generate
OLLAMA_BASE_URL=http://localhost:12345
```

### Шаг 4: Запуск компонентов

1. **Запустите Langchain Backend:**
   ```bash
   cd langchain_back
   python main.py
   ```

2. **Запустите Streamlit Frontend:**
   ```bash
   cd monitoring
   streamlit run app_main.py --server.port 27366
   ```

3. **Запустите Task Runners (в отдельных терминалах):**
   ```bash
   # Основной обработчик задач
   python -m runners.run
   
   # Процессор задач
   python -m runners.task_processor
   
   # Обработчик метрик
   python -m runners.run_metrics
   
   # RtA обработчик
   python -m runners.run_rta_queuer
   
   # Извлечение данных
   python -m runners.run_regexp
   ```

## 3. Развертывание в Kubernetes

### Предварительные требования

- Kubernetes кластер 1.24+
- Helm 3.0+
- kubectl
- Ingress контроллер

### Шаг 1: Подготовка Helm charts

Создайте структуру Helm charts для каждого компонента:

```bash
mkdir -p k8s/charts
cd k8s/charts

# Создайте charts для каждого компонента
helm create trustvar-backend
helm create trustvar-frontend
helm create trustvar-runners
helm create trustvar-mongodb
```

### Шаг 2: Настройка ConfigMaps и Secrets

```yaml
# k8s/configmaps.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trustvar-config
data:
  MONGO_HOST: "mongodb-service"
  MONGO_PORT: "27017"
  API_URL: "http://backend-service:45321/generate"
  OLLAMA_BASE_URL: "http://ollama-service:12345"
```

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trustvar-secrets
type: Opaque
data:
  MONGO_INITDB_ROOT_USERNAME: YWRtaW4=  # admin
  MONGO_INITDB_ROOT_PASSWORD: cGFzc3dvcmQ=  # password
  YANDEX_API_KEY: <base64-encoded-key>
  OPENAI_KEY: <base64-encoded-key>
```

### Шаг 3: Развертывание

```bash
# Примените конфигурации
kubectl apply -f k8s/configmaps.yaml
kubectl apply -f k8s/secrets.yaml

# Разверните компоненты
helm install trustvar-backend ./charts/trustvar-backend
helm install trustvar-frontend ./charts/trustvar-frontend
helm install trustvar-runners ./charts/trustvar-runners
helm install trustvar-mongodb ./charts/trustvar-mongodb
```

## 4. Мониторинг и обслуживание

### Логирование

**Docker Compose:**
```bash
# Просмотр логов всех сервисов
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f langchain_backend

# Логи с временными метками
docker-compose logs -f --timestamps
```

**Kubernetes:**
```bash
# Логи подов
kubectl logs -f deployment/trustvar-backend

# Логи с временными метками
kubectl logs -f deployment/trustvar-backend --timestamps=true
```

### Резервное копирование

**MongoDB:**
```bash
# Создание бэкапа
docker exec mongodb mongodump --out /backup/$(date +%Y%m%d_%H%M%S)

# Восстановление
docker exec mongodb mongorestore /backup/20240101_120000/
```

**Volumes:**
```bash
# Бэкап volumes
docker run --rm -v trustvar_mongo-volume-data:/data -v $(pwd):/backup alpine tar czf /backup/mongodb_backup.tar.gz -C /data .

# Восстановление volumes
docker run --rm -v trustvar_mongo-volume-data:/data -v $(pwd):/backup alpine tar xzf /backup/mongodb_backup.tar.gz -C /data
```

### Обновление системы

**Docker Compose:**
```bash
# Остановка сервисов
docker-compose down

# Обновление кода
git pull

# Пересборка и запуск
docker-compose up -d --build
```

**Kubernetes:**
```bash
# Обновление deployment
kubectl set image deployment/trustvar-backend backend=new-image:tag

# Откат к предыдущей версии
kubectl rollout undo deployment/trustvar-backend
```

### Масштабирование

**Docker Compose:**
```bash
# Масштабирование runners
docker-compose up -d --scale runner-main=3 --scale runner-metrics=2
```

**Kubernetes:**
```bash
# Масштабирование deployment
kubectl scale deployment trustvar-backend --replicas=3

# Автоматическое масштабирование
kubectl autoscale deployment trustvar-backend --min=2 --max=10 --cpu-percent=80
```

## 5. Устранение неполадок

### Частые проблемы

**1. MongoDB не подключается**
```bash
# Проверьте статус MongoDB
docker-compose ps mongodb

# Проверьте логи
docker-compose logs mongodb

# Проверьте подключение
docker exec -it mongodb mongosh --username admin --password password
```

**2. Backend не отвечает**
```bash
# Проверьте статус
docker-compose ps langchain_backend

# Проверьте логи
docker-compose logs langchain_backend

# Проверьте health endpoint
curl http://localhost:45321/health
```

**3. Frontend не загружается**
```bash
# Проверьте статус
docker-compose ps streamlit_frontend

# Проверьте логи
docker-compose logs streamlit_frontend

# Проверьте порт
netstat -tlnp | grep 27366
```

**4. Runners не обрабатывают задачи**
```bash
# Проверьте статус всех runners
docker-compose ps | grep runner

# Проверьте логи конкретного runner
docker-compose logs runner-main

# Проверьте подключение к MongoDB
docker exec -it runner-main python -c "from utils.db_client import get_mongo_client; print(get_mongo_client())"
```

### Диагностика

**Проверка здоровья системы:**
```bash
#!/bin/bash
# health_check.sh

echo "Checking MongoDB..."
curl -f http://localhost:27017/ || echo "MongoDB is down"

echo "Checking Backend..."
curl -f http://localhost:45321/health || echo "Backend is down"

echo "Checking Frontend..."
curl -f http://localhost:27366/ || echo "Frontend is down"

echo "Checking Ollama..."
curl -f http://localhost:12345/api/tags || echo "Ollama is down"
```

**Мониторинг ресурсов:**
```bash
# Использование CPU и памяти
docker stats

# Использование диска
df -h

# Использование сети
docker network ls
docker network inspect trustvar_default
```

## 6. Безопасность

### Рекомендации по безопасности

1. **Измените пароли по умолчанию**
2. **Используйте HTTPS в продакшене**
3. **Настройте firewall**
4. **Регулярно обновляйте зависимости**
5. **Используйте secrets management**
6. **Настройте мониторинг безопасности**

### Настройка SSL/TLS

**Nginx reverse proxy:**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:27366;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Аутентификация и авторизация

Настройте аутентификацию в `monitoring/config.yaml`:
```yaml
credentials:
  usernames:
    admin:
      email: admin@example.com
      name: Administrator
      password: $2b$12$hashed_password
    user:
      email: user@example.com
      name: Regular User
      password: $2b$12$hashed_password
``` 