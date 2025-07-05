#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности группированных метрик.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Имитируем данные из коллекции Accuracy_Groups
test_grouped_data = [
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "value": 0.75,
        "errors": []
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "value": 0.85,
        "errors": []
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Paraphrasing", "Increase sentence length"],
        "value": 0.65,
        "errors": []
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "llama3.1:8b-instruct",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "value": 0.80,
        "errors": []
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "llama3.1:8b-instruct",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "value": 0.90,
        "errors": []
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "llama3.1:8b-instruct",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Paraphrasing", "Increase sentence length"],
        "value": 0.70,
        "errors": []
    },
    {
        "task_name": "test_correlation_task",
        "dataset_name": "test_correlation_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "value": 0.85,
        "errors": []
    },
    {
        "task_name": "test_correlation_task",
        "dataset_name": "test_correlation_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "value": 0.92,
        "errors": []
    },
    {
        "task_name": "test_correlation_task",
        "dataset_name": "test_correlation_1405",
        "model": "llama3.1:8b-instruct",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "value": 0.88,
        "errors": []
    },
    {
        "task_name": "test_correlation_task",
        "dataset_name": "test_correlation_1405",
        "model": "llama3.1:8b-instruct",
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "value": 0.95,
        "errors": []
    }
]

def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Вычисляет коэффициент вариации (CV = std/mean * 100%)."""
    if not values or len(values) < 2:
        return np.nan
    mean_val = np.mean(values)
    if mean_val == 0:
        return np.nan
    std_val = np.std(values)
    return (std_val / mean_val) * 100

def test_grouped_metrics_analysis():
    """Тестирует анализ группированных метрик."""
    df = pd.DataFrame(test_grouped_data)
    
    print("Исходные данные:")
    print(df[["task_name", "model", "dynamic_augments", "value"]].to_string())
    print("\n" + "="*80 + "\n")
    
    # Фильтруем только задачи типа "Compare model behaviour"
    compare_df = df[df["task_type"] == "Compare model behaviour"].copy()
    
    if compare_df.empty:
        print("Нет данных для задач типа 'Compare model behaviour'.")
        return
    
    # Преобразуем списки dynamic_augments в строки для удобства отображения
    compare_df["augments_str"] = compare_df["dynamic_augments"].apply(
        lambda x: " + ".join(sorted(x)) if isinstance(x, list) else str(x)
    )
    
    print("Данные с преобразованными аугментациями:")
    print(compare_df[["task_name", "model", "augments_str", "value"]].to_string())
    print("\n" + "="*80 + "\n")
    
    # 1. Таблица метрик по аугментациям
    print("1. Таблица метрик по аугментациям:")
    pivot_augments = compare_df.pivot_table(
        index=["model", "task_name"], 
        columns="augments_str", 
        values="value", 
        aggfunc="mean"
    )
    print(pivot_augments.round(3))
    print("\n" + "="*80 + "\n")
    
    # 2. Коэффициент вариации для оценки устойчивости
    print("2. Коэффициент вариации (устойчивость к аугментациям):")
    
    # Вычисляем CV для каждой модели и задачи
    cv_data = []
    for (model, task), group in compare_df.groupby(["model", "task_name"]):
        values = group["value"].tolist()
        cv = calculate_coefficient_of_variation(values)
        cv_data.append({
            "model": model,
            "task_name": task,
            "cv": cv,
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
            "num_augments": len(values)
        })
    
    cv_df = pd.DataFrame(cv_data)
    
    if not cv_df.empty:
        # Сортируем по CV (меньше CV = более устойчивая модель)
        cv_df = cv_df.sort_values("cv")
        
        print("Интерпретация CV:")
        print("- CV < 10%: очень устойчивая модель")
        print("- CV 10-20%: устойчивая модель") 
        print("- CV 20-30%: умеренно устойчивая модель")
        print("- CV > 30%: неустойчивая модель")
        print()
        
        print("Результаты анализа устойчивости:")
        print(cv_df.round(3).to_string())
        print("\n" + "="*80 + "\n")
        
        # 3. Детальный анализ по каждой аугментации
        print("3. Детальный анализ по аугментациям:")
        augments = compare_df["augments_str"].unique()
        
        for augment in augments:
            print(f"\nАугментация: {augment}")
            augment_data = compare_df[compare_df["augments_str"] == augment]
            
            if not augment_data.empty:
                # Таблица значений
                pivot_augment = augment_data.pivot_table(
                    index="model", 
                    columns="task_name", 
                    values="value", 
                    aggfunc="mean"
                )
                print(pivot_augment.round(3).to_string())

if __name__ == "__main__":
    test_grouped_metrics_analysis() 