# API TrustVar

## Обзор API

TrustVar предоставляет REST API для взаимодействия с системой оценки языковых моделей. API построен на FastAPI и поддерживает асинхронные запросы.

## Базовый URL

```
http://localhost:45321
```

## Аутентификация

В текущей версии API не требует аутентификации для внутренних запросов. Для внешних запросов рекомендуется настроить API ключи или токены.

## Endpoints

### 1. Генерация ответов

#### POST /generate

Основной endpoint для генерации ответов от языковых моделей.

**Запрос:**
```json
{
  "model": "string",
  "prompt": "string",
  "variables": {
    "key1": "value1",
    "key2": "value2"
  },
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Параметры:**
- `model` (string, обязательный) - Название модели для использования
- `prompt` (string, обязательный) - Текст промпта
- `variables` (object, опциональный) - Переменные для подстановки в промпт
- `stream` (boolean, опциональный) - Включить потоковую передачу ответа
- `temperature` (float, опциональный) - Температура генерации (0.0-2.0)
- `max_tokens` (integer, опциональный) - Максимальное количество токенов

**Ответ:**
```json
{
  "response": "string",
  "model": "string",
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  },
  "finish_reason": "stop"
}
```

**Пример запроса:**
```bash
curl -X POST "http://localhost:45321/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api/gpt-4o",
    "prompt": "Привет, как дела?",
    "variables": {},
    "stream": false
  }'
```

### 2. Проверка здоровья

#### GET /health

Проверка состояния сервиса.

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### 3. Список моделей

#### GET /models

Получение списка доступных моделей.

**Ответ:**
```json
{
  "models": [
    {
      "id": "api/gpt-4o",
      "name": "GPT-4 Omni",
      "provider": "openai",
      "type": "api"
    },
    {
      "id": "qwen2.5:7b-instruct-q4_0",
      "name": "Qwen 2.5 7B Instruct",
      "provider": "ollama",
      "type": "local"
    }
  ]
}
```

### 4. Информация о модели

#### GET /models/{model_id}

Получение детальной информации о конкретной модели.

**Параметры:**
- `model_id` (string) - Идентификатор модели

**Ответ:**
```json
{
  "id": "api/gpt-4o",
  "name": "GPT-4 Omni",
  "provider": "openai",
  "type": "api",
  "capabilities": ["text-generation", "vision"],
  "max_tokens": 128000,
  "supported_formats": ["text", "json"]
}
```

### 5. Потоковая генерация

#### POST /generate/stream

Потоковая генерация ответов в реальном времени.

**Запрос:**
```json
{
  "model": "string",
  "prompt": "string",
  "variables": {},
  "stream": true
}
```

**Ответ (Server-Sent Events):**
```
data: {"chunk": "Привет", "finish_reason": null}

data: {"chunk": ", как", "finish_reason": null}

data: {"chunk": " дела?", "finish_reason": "stop"}

data: [DONE]
```

## Обработка ошибок

### Коды ошибок

| Код | Описание |
|-----|----------|
| 400 | Bad Request - Неверный запрос |
| 404 | Not Found - Ресурс не найден |
| 422 | Validation Error - Ошибка валидации |
| 500 | Internal Server Error - Внутренняя ошибка сервера |
| 503 | Service Unavailable - Сервис недоступен |

### Формат ошибки

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid-model' not found",
    "details": {
      "model": "invalid-model",
      "available_models": ["api/gpt-4o", "qwen2.5:7b-instruct-q4_0"]
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Типы ошибок

#### MODEL_NOT_FOUND
Модель не найдена или недоступна.

#### INVALID_PROMPT
Промпт содержит недопустимые символы или превышает лимиты.

#### PROVIDER_ERROR
Ошибка провайдера модели (API недоступен, превышен лимит и т.д.).

#### VALIDATION_ERROR
Ошибка валидации входных данных.

## Примеры использования

### Python клиент

```python
import requests
import json

class TrustVarClient:
    def __init__(self, base_url="http://localhost:45321"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate(self, model, prompt, variables=None, stream=False):
        """Генерация ответа от модели"""
        url = f"{self.base_url}/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "variables": variables or {},
            "stream": stream
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_models(self):
        """Получение списка доступных моделей"""
        url = f"{self.base_url}/models"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Проверка здоровья сервиса"""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Пример использования
client = TrustVarClient()

# Генерация ответа
response = client.generate(
    model="api/gpt-4o",
    prompt="Объясни квантовую физику простыми словами",
    variables={"level": "простой"}
)
print(response["response"])

# Получение списка моделей
models = client.get_models()
for model in models["models"]:
    print(f"{model['name']} ({model['provider']})")
```

### JavaScript клиент

```javascript
class TrustVarClient {
    constructor(baseUrl = 'http://localhost:45321') {
        this.baseUrl = baseUrl;
    }
    
    async generate(model, prompt, variables = {}, stream = false) {
        const url = `${this.baseUrl}/generate`;
        const data = {
            model,
            prompt,
            variables,
            stream
        };
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getModels() {
        const url = `${this.baseUrl}/models`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async healthCheck() {
        const url = `${this.baseUrl}/health`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Пример использования
const client = new TrustVarClient();

// Генерация ответа
client.generate('api/gpt-4o', 'Привет, мир!')
    .then(response => console.log(response.response))
    .catch(error => console.error('Error:', error));

// Получение списка моделей
client.getModels()
    .then(data => {
        data.models.forEach(model => {
            console.log(`${model.name} (${model.provider})`);
        });
    })
    .catch(error => console.error('Error:', error));
```

### cURL примеры

#### Базовая генерация
```bash
curl -X POST "http://localhost:45321/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api/gpt-4o",
    "prompt": "Напиши короткое стихотворение о программировании",
    "variables": {},
    "stream": false
  }'
```

#### Генерация с переменными
```bash
curl -X POST "http://localhost:45321/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct-q4_0",
    "prompt": "Переведи на {language}: {text}",
    "variables": {
      "language": "английский",
      "text": "Привет, как дела?"
    },
    "stream": false
  }'
```

#### Потоковая генерация
```bash
curl -X POST "http://localhost:45321/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api/gpt-4o",
    "prompt": "Расскажи историю о космическом путешествии",
    "variables": {},
    "stream": true
  }'
```

## Ограничения и лимиты

### Лимиты запросов
- **Максимальный размер промпта**: 128,000 токенов
- **Максимальный размер ответа**: 128,000 токенов
- **Таймаут запроса**: 300 секунд
- **Частота запросов**: 100 запросов в минуту на IP

### Поддерживаемые форматы
- **Входные данные**: JSON
- **Выходные данные**: JSON, Server-Sent Events (для потоковой передачи)
- **Кодировка**: UTF-8

## Мониторинг и метрики

### Метрики API
- Количество запросов в секунду
- Время ответа
- Количество ошибок
- Использование токенов

### Логирование
Все API запросы логируются с информацией о:
- Времени запроса
- IP адресе клиента
- Используемой модели
- Размере промпта и ответа
- Времени обработки
- Статусе ответа

## Безопасность

### Рекомендации
1. Используйте HTTPS в продакшене
2. Настройте rate limiting
3. Валидируйте входные данные
4. Логируйте подозрительную активность
5. Регулярно обновляйте зависимости

### Заголовки безопасности
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

## Версионирование

API использует семантическое версионирование. Текущая версия: `v1.0.0`

### Изменения в версиях
- **v1.0.0**: Первоначальная версия API
- **v1.1.0**: Добавлена поддержка потоковой передачи
- **v1.2.0**: Добавлены метрики использования токенов

### Обратная совместимость
API гарантирует обратную совместимость в рамках мажорной версии. Критические изменения будут объявлены заранее. 