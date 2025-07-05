#!/usr/bin/env python3
"""
Тестовый скрипт для проверки логики аугментации с новой архитектурой.
"""

def extract_text_from_response(response):
    """
    Извлекает текст из ответа API.
    
    Args:
        response: Ответ от API.
        
    Returns:
        str: Извлеченный текст или None, если не удалось извлечь.
    """
    if isinstance(response, dict):
        # Ищем стандартные ключи с текстом
        for key in ["response", "text", "content", "result", "output"]:
            if key in response and isinstance(response[key], str):
                return response[key]
        
        # Если не нашли стандартные ключи, берем первый строковый ключ
        for key, value in response.items():
            if isinstance(value, str):
                return value
        
        print(f"Не удалось извлечь текст из ответа: {response}")
        return None
    elif isinstance(response, str):
        return response
    else:
        print(f"Неожиданный формат ответа: {type(response)}")
        return None


def format_prompt_with_variables(prompt, variables):
    """
    Форматирует промпт с переменными.
    
    Args:
        prompt (str): Промпт с плейсхолдерами.
        variables (dict): Переменные для подстановки.
        
    Returns:
        str: Промпт с подставленными переменными.
    """
    try:
        return prompt.format(**variables)
    except KeyError as e:
        print(f"Переменная {e} не найдена в промпте, используем исходный промпт")
        return prompt


def test_augmentation_response_parsing():
    """Тестирует парсинг ответов от аугментатора."""
    
    # Тестовые ответы от API
    test_responses = [
        {"response": "Это аугментированный текст с {variable}."},
        {"text": "Другой формат ответа с {variable}."},
        {"content": "Третий формат с {variable}."},
        {"result": "Нестандартный ключ с {variable}."},
        {"output": "Ключ output с {variable}."},
        "Просто строка с {variable}.",
        {"error": "Ошибка", "data": "Данные"}
    ]
    
    print("Тестирование парсинга ответов аугментатора:")
    print("=" * 50)
    
    for i, resp in enumerate(test_responses):
        result = extract_text_from_response(resp)
        print(f"Тест {i+1}: {resp}")
        print(f"Результат: {result}")
        print("-" * 30)
    
    # Тестирование подстановки переменных
    print("\nТестирование подстановки переменных:")
    print("=" * 50)
    
    variables = {"variable": "тестовое_значение"}
    
    test_texts = [
        "Текст с {variable}.",
        "Текст без переменных.",
        "Текст с {variable} и {missing_var}."
    ]
    
    for text in test_texts:
        print(f"Исходный текст: {text}")
        result = format_prompt_with_variables(text, variables)
        print(f"Результат: {result}")
        print("-" * 30)


def test_augmentation_flow():
    """Тестирует полный поток аугментации с новой архитектурой."""
    
    # Имитируем данные задачи
    task_data = {
        "prompt": "Прочитайте текст: {text} и ответьте на вопрос: {question}",
        "variables": {
            "text": "Москва - столица России.",
            "question": "Какая столица России?"
        },
        "dynamic_augments": ["Synonymy", "Paraphrasing"]
    }
    
    # Имитируем ответы аугментатора
    augment_responses = [
        {"response": "Изучите текст: {text} и дайте ответ на вопрос: {question}"},  # Synonymy
        {"response": "Проанализируйте следующий текст: {text} и решите задачу: {question}"}  # Paraphrasing
    ]
    
    print("Тестирование полного потока аугментации (новая архитектура):")
    print("=" * 60)
    
    prompt = task_data["prompt"]
    variables = task_data["variables"]
    
    print(f"Исходный промпт: {prompt}")
    print(f"Переменные: {variables}")
    print()
    
    # Шаг 1: Форматируем исходный промпт с переменными
    formatted_prompt = format_prompt_with_variables(prompt, variables)
    print(f"1. Исходный промпт с переменными: {formatted_prompt}")
    print()
    
    for i, augment_resp in enumerate(augment_responses):
        technique = task_data["dynamic_augments"][i]
        print(f"Аугментация {i+1} ({technique}):")
        
        # Шаг 2: Извлекаем аугментированный текст
        augmented_text = extract_text_from_response(augment_resp)
        print(f"  2. Ответ аугментатора: {augment_resp}")
        print(f"  3. Извлеченный текст: {augmented_text}")
        
        if augmented_text:
            # Шаг 3: Подставляем переменные в аугментированный текст
            augmented_prompt_with_vars = format_prompt_with_variables(augmented_text, variables)
            print(f"  4. Финальный промпт: {augmented_prompt_with_vars}")
            
            # Шаг 4: Имитируем отправку в основную модель
            print(f"  5. Отправляем в модель: make_request(model, '{augmented_prompt_with_vars}', session)")
        
        print("-" * 40)


def test_ordinary_task_flow():
    """Тестирует поток обычной задачи."""
    
    task_data = {
        "prompt": "Прочитайте текст: {text} и ответьте на вопрос: {question}",
        "variables": {
            "text": "Москва - столица России.",
            "question": "Какая столица России?"
        }
    }
    
    print("Тестирование потока обычной задачи:")
    print("=" * 50)
    
    prompt = task_data["prompt"]
    variables = task_data["variables"]
    
    print(f"Исходный промпт: {prompt}")
    print(f"Переменные: {variables}")
    
    # Шаг 1: Форматируем промпт с переменными
    formatted_prompt = format_prompt_with_variables(prompt, variables)
    print(f"1. Промпт с переменными: {formatted_prompt}")
    
    # Шаг 2: Имитируем отправку в модель
    print(f"2. Отправляем в модель: make_request(model, '{formatted_prompt}', session)")


if __name__ == "__main__":
    test_augmentation_response_parsing()
    print("\n" + "="*60 + "\n")
    test_augmentation_flow()
    print("\n" + "="*60 + "\n")
    test_ordinary_task_flow() 