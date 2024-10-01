# LangchainBack

Тут живет код для бекенд приложения на лангчейне которое позволяет оставить способ взаимодействия таким-же как он был в TrustLLM

## Запуск бекенда
1) ```screen -S langchain_back```
2) ```source .venv/bin/activate```
3) ```python main.py```
4) ```screen -r langchain_back```

## Проверка:
Сервис можно запустить запустив main.py
после запуска можно отправить тестовый запрос
```curl -X POST "http://127.0.0.1:8000/generate/yandexgpt" -H "Content-Type: application/json" -d '{"prompt": "Translate this to Russian: Hello", "variables": {}}'```

или 

```curl -X POST "http://127.0.0.1:8000/generate/ollama" -H "Content-Type: application/json" -d '{"prompt": "Translate this to Russian: Hello", "variables": {}, "model": "gemma:7b-instruct-v1.1-q4_0"}'  ```