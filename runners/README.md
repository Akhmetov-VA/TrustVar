# run.py
реализует непрерывный процесс обработки задач, хранящихся в базе данных MongoDB. Его основная цель — опрашивать определённые коллекции в базах данных, выбирать задачи со статусом `"pending"`, отправлять соответствующие запросы к внешнему API для обработки и затем обновлять записи в MongoDB с результатами выполнения или информацией об ошибках. Ниже приведено подробное описание его работы:

1. **Настройка логирования и загрузка переменных окружения:**
   - Функция `configure_logging()` настраивает логирование: задаётся уровень логов (INFO) и формат сообщений (с указанием времени, уровня и текста сообщения). Все сообщения выводятся в консоль. Это позволяет отслеживать ход выполнения скрипта.
   - В начале работы выводится сообщение об успешной настройке логирования.

2. **Подключение к MongoDB:**
   - Функция `get_mongo_client()` формирует URI для подключения к MongoDB, используя параметры (хост, порт, имя пользователя, пароль), импортированные из модуля с константами.
   - Скрипт пытается установить соединение с MongoDB, выводит сообщения о попытке и об успешном подключении, или, в случае ошибки, регистрирует исключение.

3. **Отправка запросов к API:**
   - Функция `make_request()` принимает следующие аргументы:
     - `model` — название модели, которая будет использоваться для обработки запроса;
     - `prompt` — текст запроса;
     - `variables` — дополнительные переменные для запроса;
     - `session` — объект сессии из библиотеки `requests` для повторного использования HTTP-соединений.
   - Внутри функции формируется POST-запрос к API (используется URL, заданный константой `API_URL`), в теле которого передаются модель, промпт, переменные и флаг `"stream": False`.
   - Если запрос проходит успешно, возвращается JSON-ответ API. В случае ошибки выполняется логирование и генерируется исключение.

4. **Обработка отдельной задачи:**
   - Функция `process_task()` принимает:
     - `task` — документ задачи из MongoDB (включает такие поля, как `_id`, `prompt`, `model` и, возможно, `variables`);
     - `collection` — коллекция MongoDB, где находится данная задача;
     - `session` — сессия для HTTP-запросов.
   - Сначала извлекаются необходимые данные задачи (идентификатор, промпт, модель, переменные).
   - Затем с помощью `make_request()` отправляется запрос к API для обработки задачи.
   - Если запрос выполнен успешно, документ задачи в базе данных обновляется: поле `status` меняется на `"completed"`, а в поле `response` записывается ответ от API.
   - Если при обработке возникает ошибка, статус задачи обновляется на `"failed"`, а в поле `error` записывается информация об ошибке. В обоих случаях происходят соответствующие логирования.

5. **Обработка коллекций задач:**
   - Функция `process_collection()` отвечает за обработку всех задач внутри конкретной коллекции базы данных.
   - Она принимает:
     - `db` — объект базы данных MongoDB,
     - `collection_name` — имя коллекции,
     - `session` — сессия для HTTP-запросов.
   - Сначала функция получает коллекцию по имени и определяет список уникальных моделей (`distinct("model")`), для которых существуют задачи.
   - Для каждой найденной модели запускается цикл, в котором последовательно выбираются задачи со статусом `"pending"` для данной модели. Для этого используется метод `find_one_and_update()`, который одновременно находит задачу и меняет её статус на `"processing"`, что предотвращает повторную обработку одной и той же задачи.
   - Если задача найдена, она передается в `process_task()` для дальнейшей обработки. Если задач для модели больше нет, цикл завершается, и переходим к следующей модели.

6. **Основной цикл обработки:**
   - Функция `run_processing_loop()` запускает цикл обработки задач для всех подходящих коллекций в базе данных.
   - Вначале создаётся сессия `requests.Session()` для оптимизации HTTP-запросов.
   - Скрипт получает список всех коллекций в базе данных, исключая коллекции с именами `"delete_me"` и `"test"`, которые не предназначены для обработки.
   - Для каждой оставшейся коллекции вызывается функция `process_collection()`, которая обрабатывает задачи внутри коллекции.
   - После обработки всех коллекций скрипт делает паузу (5 секунд), чтобы дать время появлению новых задач, и продолжает цикл.

7. **Запуск скрипта и работа с несколькими базами данных:**
   - Функция `main()` является точкой входа в скрипт.
   - В ней происходит настройка логирования и установка подключения к MongoDB.
   - Затем запускается бесконечный цикл, в котором последовательно обрабатываются две базы данных: `"TrustLLM_ru"` и `"TrustGen"`.
   - Для каждой базы данных вызывается `run_processing_loop()`, после чего скрипт делает паузу (10 секунд) перед повторным циклом, обеспечивая непрерывный мониторинг и обработку новых задач.

Таким образом, скрипт организует автоматизированную обработку задач, хранящихся в нескольких коллекциях MongoDB, посредством следующих шагов:

- **Выбор задачи:** Извлечение задачи со статусом `"pending"` из коллекции, при этом обновление статуса на `"processing"` для предотвращения повторной обработки.
- **Обработка задачи:** Отправка запроса к API с использованием заданной модели, промпта и переменных.
- **Обновление результата:** После получения ответа API, обновление записи задачи в MongoDB с установкой статуса `"completed"` и сохранением ответа, либо установка статуса `"failed"` с информацией об ошибке при возникновении проблем.
- **Непрерывный цикл:** Постоянное опрашивание баз данных и коллекций, с заданными интервалами ожидания, что позволяет обрабатывать задачи в режиме реального времени.

Скрипт логгирует все ключевые события (подключение к базе, отправку запроса, получение ответа, ошибки) для удобства мониторинга и отладки работы системы.


# 2 task_processor.py
Этот скрипт предназначен для автоматизации создания записей в очереди для последующей обработки задач, определённых в базе данных MongoDB. Основная логика скрипта заключается в том, чтобы периодически считывать задачи из коллекции `tasks`, извлекать для каждой задачи связанный датасет, а затем для каждой строки этого датасета и для каждой указанной модели создавать (если такая запись ещё не существует) новую запись в специальной коллекции очереди. Ниже приведено подробное описание работы скрипта:

---

### 1. Инициализация и настройки

- **Импорты и константы:**  
  Скрипт использует стандартные модули Python (`logging`, `os`, `time`), библиотеку для работы с данными — `pandas`, а также клиент для MongoDB из пакета `pymongo`. Параметры подключения к MongoDB (хост, порт, имя пользователя, пароль) импортируются из модуля с константами.  
  Также определяется имя базы данных `MONGO_DB`, которое берётся из переменной окружения `MONGO_DB` (если переменная не задана, используется значение `"TrustGen"`).

- **Настройка логирования:**  
  С помощью `logging.basicConfig` настраивается логирование на уровне `INFO`. Все сообщения форматируются с указанием времени, уровня логирования и самого сообщения. Создаётся объект логгера `logger` для дальнейшего использования в скрипте.

---

### 2. Подключение к MongoDB

- **Функция `get_mongo_client()`:**  
  Эта функция формирует URI для подключения к MongoDB, используя заданные параметры (имя пользователя, пароль, хост и порт). Создаётся экземпляр `MongoClient`, и при успешном подключении выводится сообщение в лог.  
  _Возвращаемое значение:_ объект `MongoClient`.

- **Функция `get_db()`:**  
  Вызывает `get_mongo_client()` и возвращает объект базы данных, используя имя, указанное в `MONGO_DB`.

---

### 3. Получение задач и датасета

- **Функция `fetch_tasks(db)`:**  
  Из базы данных извлекается коллекция `tasks` и с неё считываются все документы. Полученные задачи возвращаются в виде списка словарей.  
  _Назначение:_ получить перечень всех задач, зарегистрированных в системе.

- **Функция `get_dataset_head(db, dataset_name)`:**  
  Для конкретной задачи определяется имя коллекции с датасетом в формате `dataset_{dataset_name}`. Из этой коллекции извлекаются все документы, на основе которых формируется объект `pandas.DataFrame`. Если в DataFrame присутствует столбец `_id`, он удаляется, чтобы оставить только полезные данные.  
  _Возвращаемое значение:_ DataFrame с данными датасета или пустой DataFrame, если документов нет.

---

### 4. Формирование записей очереди для задач

- **Функция `insert_queue_entries_for_task(db, task)`:**  
  Это ключевая функция, которая для каждой задачи из коллекции `tasks` выполняет следующие действия:
  
  1. **Извлечение параметров задачи:**  
     Из документа задачи извлекаются такие поля, как:
     - `task_name` — имя задачи.
     - `dataset_name` — имя датасета, с которым связана задача.
     - `prompt` — текст запроса, который будет использоваться при обработке.
     - `variables_cols` — список названий столбцов, значения из которых будут использованы как переменные.
     - `models` — список моделей, для которых нужно создать записи.
     - `metric` — тип метрики, которая определяет особенности обработки.
     - Дополнительные параметры, такие как `target`, `regexp`, `include_column`, `exclude_column`, `rta_prompt` и `rta_model`.

  2. **Получение датасета:**  
     С помощью функции `get_dataset_head` извлекается DataFrame для указанного датасета. Если датасет пустой, функция логгирует предупреждение и прекращает дальнейшую обработку для данной задачи.

  3. **Определение коллекции очереди:**  
     Имя коллекции очереди формируется по схеме `queue_{task_name}`. Именно в этой коллекции будут сохраняться записи для каждой строки датасета и для каждой модели.

  4. **Анализ уже существующих записей:**  
     Чтобы избежать дублирования, из коллекции очереди извлекаются существующие записи, и для каждой из них формируется пара ключей `(line_index, model)`. Эти ключи собираются в множество `existing_keys`. Таким образом, при создании новых записей проверяется, не существует ли уже запись для конкретной строки датасета и модели.

  5. **Формирование новых записей:**  
     - Датасет переводится в список словарей, где каждый словарь соответствует строке.
     - Для каждой строки (с индексом `i`) и для каждого элемента из списка `models` проверяется, присутствует ли пара `(i, model)` в `existing_keys`.
     - Если такой ключ отсутствует, формируется новый документ с полями:
       - `"task_name"`, `"line_index"`, `"dataset_name"`, `"prompt"`, `"variables"` (словарь, составленный из значений столбцов, указанных в `variables_cols`), `"model"`, `"metric"`, `"regexp"`, а также полями `"status"` (устанавливается в `"pending"`) и `"response"` (изначально `None`).
     - Далее, в зависимости от значения поля `metric`, документ дополняется:
       - **Если `metric` равен `"RtA"`:**  
         Если заданы `rta_prompt` и `rta_model`, они добавляются в документ. Также поле `"target"` устанавливается либо в значение `target` (если это строка), либо в значение `metric`.
       - **Если `metric` равен `"include_exclude"`:**  
         Если в задаче указаны поля `include_column` и `exclude_column` и они присутствуют в строке датасета, создаются списки `include_list` и `exclude_list` соответственно. Поле `"target"` также устанавливается аналогично.
       - **В остальных случаях:**  
         Если существует поле `target` и оно присутствует в строке датасета, его значение берётся из строки; если нет, то `"target"` устанавливается в `None`.

  6. **Вставка новых записей в коллекцию очереди:**  
     Если сформированы новые документы, они вставляются в коллекцию с помощью метода `insert_many` (с опцией `ordered=False`, что позволяет выполнять вставку независимо от порядка). После успешной вставки логируется количество добавленных документов. В случае возникновения ошибки производится логирование сообщения об ошибке.

---

### 5. Основной цикл выполнения

- **Функция `main()`:**  
  Эта функция организует бесконечный цикл, в рамках которого происходит:
  
  1. **Подключение к базе данных:**  
     Сначала вызывается функция `get_db()` для получения объекта базы данных.

  2. **Периодическое выполнение:**  
     Задаётся интервал ожидания (10 секунд). В бесконечном цикле:
     - Вызывается функция `fetch_tasks(db)` для получения всех задач из коллекции `tasks`.
     - Если задачи найдены, для каждой задачи производится:
       - Логирование начала обработки задачи (с указанием `task_name`).
       - Вызов функции `insert_queue_entries_for_task(db, task)`, которая отвечает за создание соответствующих записей в очереди.
     - Если задач нет, логируется соответствующее сообщение.
     - После обработки всех задач цикл делает паузу на 10 секунд перед следующей итерацией.
  
  3. **Завершение работы:**  
     Если во время работы пользователь прерывает выполнение (например, через сочетание клавиш Ctrl+C), перехватывается исключение `KeyboardInterrupt`, и в лог выводится сообщение об остановке процесса.

---

### Итоговое назначение скрипта

Скрипт автоматизирует процесс формирования очередей для обработки задач. Каждый документ из коллекции `tasks` содержит информацию о том, какой датасет и какие модели использовать, а также дополнительные параметры (например, параметры метрики). Скрипт извлекает датасет из соответствующей коллекции, а затем для каждой строки датасета и для каждой модели создаёт уникальную запись в очереди (коллекция с именем `queue_{task_name}`). Это позволяет в дальнейшем другим процессам или сервисам считывать записи очереди, выполнять заданную обработку (например, отправку запросов к API, анализ данных и т.д.) и обновлять статусы выполненных задач.

Таким образом, данный скрипт является связующим звеном между определением задач (коллекция `tasks`) и формированием очереди заданий для дальнейшей обработки, обеспечивая автоматизированное и периодическое обновление очередей в базе данных MongoDB.


# 3 run_rta_queuer.py
Данный скрипт реализует автоматизированный процесс переноса задач с метрикой **"RtA"** из обычных очередей в специальные **RTA-очереди** в базе данных MongoDB. Его основная цель — находить завершённые задачи (со статусом `"completed"`) из коллекций, имена которых начинаются с префикса `"queue_"`, проверять наличие необходимых для переноса полей, преобразовывать данные и создавать для них новую запись в соответствующей RTA-очереди (коллекция с именем, начинающимся с `"queue_rta_"`). Ниже приведено подробное описание работы скрипта.

---

### 1. Настройка окружения и подключение к базе данных

- **Импорт модулей и настройка переменных:**  
  Скрипт импортирует необходимые библиотеки:
  - `logging`, `os`, `time` для логирования, работы с операционной системой и задержками.
  - `pandas` для работы с табличными данными (хотя в данном скрипте она не используется напрямую).
  - `dotenv` для загрузки переменных окружения (предполагается, что переменные уже установлены).
  - `pymongo` для работы с MongoDB.
  
  Из модуля `utils.constants` импортируются константы подключения: `MONGO_HOST`, `MONGO_PASSWORD`, `MONGO_PORT` и `MONGO_USERNAME`.  
  Имя базы данных (`MONGO_DB`) определяется из переменной окружения, по умолчанию используется значение `"TrustGen"`.

- **Настройка логирования:**  
  Логирование настроено на уровне `INFO` с выводом сообщений в формате, содержащем время, уровень логирования и само сообщение. Это позволяет отслеживать все ключевые этапы выполнения скрипта.

- **Подключение к MongoDB:**  
  Функция `get_mongo_client()` формирует URI для подключения к MongoDB, используя заданные константы, и создаёт экземпляр `MongoClient`. При успешном подключении выводится информационное сообщение в лог.  
  Функция `get_db()` получает объект базы данных, используя имя, указанное в переменной `MONGO_DB`.

---

### 2. Поиск задач с метрикой "RtA"

- **Функция `fetch_rta_tasks(db: Database)`:**  
  Это генератор, который:
  - Перебирает все коллекции в базе данных, имена которых начинаются с `"queue_"` (это обычные очереди задач).
  - В каждой из таких коллекций ищет документы (задачи), у которых поле `"metric"` равно `"RtA"` и статус (`"status"`) равен `"completed"`.
  - Для каждой найденной задачи функция выдаёт кортеж из имени коллекции (`coll_name`) и самого документа задачи.
  
  Таким образом, на каждом шаге цикла будут доступны задачи, готовые к переносу в RTA-очередь.

---

### 3. Перенос задачи в RTA-очередь

Функция `create_rta_queue_entry(db: Database, coll_name: str, task: Dict[str, Any])` выполняет перенос одной задачи из обычной очереди в специальную RTA-очередь. Рассмотрим её логику по шагам:

1. **Определение целевой коллекции RTA-очереди:**  
   - Из имени исходной коллекции (например, `"queue_myTask"`) извлекается имя задачи (`task_name`), удаляя префикс `"queue_"`.
   - Формируется имя целевой коллекции для RTA-задач: `"queue_rta_{task_name}"`.

2. **Извлечение и проверка обязательных полей:**  
   - Из исходной задачи извлекается оригинальная модель (`original_model`) и исходный prompt (`original_prompt`).
   - Из задачи извлекаются поля `rta_model` и `rta_prompt`. Если одно из них отсутствует, задача не может быть перенесена:
     - В этом случае в исходной записи обновляется статус на `"error"`, а в поле `"error"` записывается соответствующее сообщение (например, "Задача RtA без rta_model" или "Задача RtA без rta_prompt").
   - Также проверяется наличие поля `response`. Если ответа нет, задача помечается как ошибочная аналогичным образом.

3. **Формирование нового поля `variables`:**  
   - Исходный prompt, содержащий заполнители (например, `{name}`, `{value}`), форматируется с использованием словаря `variables` из исходной задачи. Результатом является строка `filled_input` — исходный prompt с подставленными значениями.
   - Если при форматировании возникает ошибка (например, из-за отсутствующих переменных), задача обновляется с ошибкой `"Ошибка форматирования prompt"`.
   - Создаётся новый словарь `new_variables`, который содержит:
     - `"input"` — заполненный prompt (`filled_input`);
     - `"answer"` — значение поля `response` из исходной задачи.

4. **Проверка на дублирование:**  
   - В целевой RTA-очереди проверяется, существует ли уже запись с такими же значениями:
     - Поле `model` должно совпадать с новым `rta_model`.
     - Поля `variables.input` и `variables.answer` должны совпадать с `filled_input` и `response` соответственно.
   - Если такая запись уже есть, в исходной задаче выставляется статус `"error"` с сообщением `"Дубликат в rta_queue"`, и перенос не выполняется.

5. **Формирование новой записи для RTA-очереди:**  
   Если все проверки пройдены, формируется новый документ для вставки в целевую коллекцию:
   - **Поля копирования и преобразования:**
     - `"task_name"` и `"dataset_name"` берутся из исходной задачи.
     - `"init_prompt"` и `"init_model"` сохраняют исходный prompt и модель.
     - `"prompt"` устанавливается равным `rta_prompt` (т.е. специализированному prompt для RtA).
     - `"model"` заменяется на `rta_model`.
     - `"variables"` принимает сформированный ранее словарь с полями `"input"` и `"answer"`.
     - Дополнительно копируется поле `"regexp"` и `"target"`, если они есть.
   - **Значения по умолчанию:**  
     - Статус новой записи устанавливается в `"pending"`, что означает, что задача ожидает дальнейшей обработки.
     - Поле `"metric"` жестко задаётся как `"accuracy"` (в соответствии с условием задачи).
   - **Синхронизация с исходной записью:**  
     - Добавляется поле `"source_id"`, в котором хранится идентификатор исходной задачи. Это позволяет отслеживать, из какой записи произошёл перенос.

6. **Вставка и обновление статусов:**  
   - Новый документ вставляется в целевую RTA-очередь (`rta_coll.insert_one(doc)`).
   - После успешной вставки логируется сообщение об успешном переносе.
   - Исходная задача в обычной очереди обновляется: её статус меняется на `"transfered_to_rta"`, что свидетельствует о том, что задача была успешно перенесена.

---

### 4. Бесконечный цикл переноса задач

- **Функция `run_rta_transfer_loop(db: Database, interval: int = 10)`:**  
  Эта функция реализует основной цикл работы скрипта:
  - На каждом шаге цикла с помощью генератора `fetch_rta_tasks(db)` перебираются все задачи с метрикой `"RtA"` и статусом `"completed"` из обычных очередей.
  - Для каждой найденной задачи вызывается функция `create_rta_queue_entry`, которая пытается перенести её в соответствующую RTA-очередь.
  - Если в текущей итерации не найдено ни одной задачи для переноса, выводится информационное сообщение `"Нет задач RtA для переноса. Ожидание..."`.
  - После обработки всех найденных задач цикл засыпает на заданный интервал (по умолчанию 10 секунд) и затем повторяет проверку.

---

### 5. Точка входа в скрипт

- **Функция `main()`:**  
  - Получает объект базы данных через `get_db()`.
  - Запускает бесконечный цикл переноса задач, вызвав `run_rta_transfer_loop(db, interval=10)`.

- **Запуск скрипта:**  
  Если скрипт выполняется напрямую (не импортируется как модуль), вызывается функция `main()`, и процесс переноса задач запускается немедленно.

---

### Итоговая схема работы

1. **Подключение к базе данных:**  
   Скрипт устанавливает соединение с MongoDB, используя заданные параметры подключения.

2. **Мониторинг очередей:**  
   Он перебирает все коллекции, имена которых начинаются с `"queue_"`, и выбирает из них задачи с метрикой `"RtA"` и статусом `"completed"`.

3. **Проверка и перенос:**  
   Для каждой найденной задачи выполняется проверка наличия обязательных полей (`rta_model`, `rta_prompt`, `response`) и корректность форматирования исходного prompt с подстановкой значений. При отсутствии необходимых данных задача помечается как ошибочная. Если все в порядке и дубликата в целевой коллекции нет, создаётся новая запись в RTA-очереди (`queue_rta_{task_name}`) с преобразованными данными и ссылкой на исходную запись.

4. **Обновление исходной задачи:**  
   После успешного переноса статус исходной задачи обновляется на `"transfered_to_rta"`, что предотвращает повторный перенос.

5. **Непрерывный цикл:**  
   Скрипт работает в бесконечном цикле, периодически (каждые 10 секунд) проверяя наличие новых задач для переноса, что позволяет обеспечить своевременную синхронизацию данных между обычными очередями и RTA-очередями.

Таким образом, данный скрипт служит для автоматизированного переноса и преобразования задач с метрикой **"RtA"** из обычных очередей в специализированные RTA-очереди, обеспечивая при этом контроль корректности данных, предотвращение дублирования и синхронизацию информации между различными коллекциями в MongoDB.


