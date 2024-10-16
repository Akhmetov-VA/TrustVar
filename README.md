

Архитектура инференса
1) MongoDB
2) LangchainBackend


## Запуск бекенда
1) ```screen -S langchain_back```
2) ```source .venv/bin/activate```
3) ```python main.py```
4) ```screen -r langchain_back```

## Запуск обработчика задач
1) Создаем сессию ```screen -S my_session```
2) В новом терминале запускаем наш скрипт ```/home/vadim/work/TrustLLM_ru/.venv/bin/python /home/vadim/work/TrustLLM_ru/benchmark/runers/run.py```
    теперь можно закрыть терминал 
3) Для подключения к созданной сессии ```screen -r my_session```

## Запуск мониторинга состояния экспериментов
1) Создаем сессию ```screen -S monitoring```
2) В новом терминале запускаем наш скрипт ```/home/vadim/work/TrustLLM_ru/.venv/bin/python -m streamlit run /home/vadim/work/TrustLLM_ru/monitoring/app.py --server.port 27365```
    теперь можно закрыть терминал 
3) Для подключения к созданной сессии ```screen -r monitoring```