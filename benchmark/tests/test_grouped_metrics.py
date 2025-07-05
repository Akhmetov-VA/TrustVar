#!/usr/bin/env python3
"""
Тестовый скрипт для проверки логики группировки метрик по task_type и dynamic_augments.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Имитируем данные из базы
test_data = [
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "metric": "accuracy",
        "pred": "4",
        "target": 4,
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "input": "test input 1"
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "metric": "accuracy",
        "pred": "3",
        "target": 4,
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Synonymy", "Stylistic change"],
        "input": "test input 2"
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "metric": "accuracy",
        "pred": "4",
        "target": 4,
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "input": "test input 3"
    },
    {
        "task_name": "test_accuracy_small_generateion",
        "dataset_name": "test_accuracy_small_1405",
        "model": "qwen2.5:7b-instruct-q4_0",
        "metric": "accuracy",
        "pred": "TFN",
        "target": 4,
        "task_type": "Compare model behaviour",
        "dynamic_augments": ["Reorder words/phrases", "Shorten sentence length"],
        "input": "test input 4"
    }
]

def compute_accuracy(df: pd.DataFrame) -> tuple[float, List[Dict[str, Any]]]:
    """Вычисляет accuracy для группы данных."""
    df_valid = df[df["pred"] != "TFN"]
    if df_valid.empty:
        return np.nan, []
    cond = df_valid["pred"].astype(str) != df_valid["target"].astype(str)
    value = (~cond).mean()
    errors = []
    for idx, row in df_valid[cond].iterrows():
        errors.append({
            "input": row["input"],
            "pred": row["pred"],
            "target": row["target"]
        })
    return value, errors

def compute_tfnr(df: pd.DataFrame) -> tuple[float, List[Dict[str, Any]]]:
    """Вычисляет TFNR для группы данных."""
    total = len(df)
    if total == 0:
        return np.nan, []
    cond = df["pred"] == "TFN"
    value = cond.sum() / total
    errors = []
    for idx, row in df[cond].iterrows():
        errors.append({
            "input": row["input"],
            "pred": row["pred"],
            "target": row["target"]
        })
    return value, errors

def test_grouped_metrics():
    """Тестирует логику группировки метрик."""
    df = pd.DataFrame(test_data)
    
    print("Исходные данные:")
    print(df[["task_name", "model", "task_type", "dynamic_augments", "pred", "target"]])
    print("\n" + "="*80 + "\n")
    
    # Фильтруем только записи с task_type и dynamic_augments
    df_with_groups = df[
        (df["task_type"].notna()) & 
        (df["task_type"] != "") & 
        (df["dynamic_augments"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
    ]
    
    print(f"Записей с группировкой: {len(df_with_groups)}")
    
    if not df_with_groups.empty:
        # Преобразуем списки dynamic_augments в строки для группировки
        df_with_groups = df_with_groups.copy()
        df_with_groups["dynamic_augments_str"] = df_with_groups["dynamic_augments"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )
        
        grouped_acc_res = []
        grouped_tfnr_res = []
        
        # Группируем по task_type, dynamic_augments_str, task_name, dataset_name, model, metric
        for (task_type, augments_str, task, ds, model, metric), g in df_with_groups.groupby(
            ["task_type", "dynamic_augments_str", "task_name", "dataset_name", "model", "metric"]
        ):
            # Восстанавливаем оригинальный список dynamic_augments
            augments = g["dynamic_augments"].iloc[0]
            
            print(f"\nГруппа: {task_type} - {augments}")
            print(f"Записей в группе: {len(g)}")
            print(g[["pred", "target"]].to_string())
            
            # Вычисляем метрики
            val_acc, errs_acc = compute_accuracy(g)
            val_tfnr, errs_tfnr = compute_tfnr(g)
            
            print(f"Accuracy: {val_acc:.3f}")
            print(f"TFNR: {val_tfnr:.3f}")
            
            grouped_acc_res.append({
                "task_name": task,
                "dataset_name": ds,
                "model": model,
                "task_type": task_type,
                "dynamic_augments": augments,
                "value": val_acc,
                "errors": errs_acc,
            })
            
            grouped_tfnr_res.append({
                "task_name": task,
                "dataset_name": ds,
                "model": model,
                "task_type": task_type,
                "dynamic_augments": augments,
                "value": val_tfnr,
                "errors": errs_tfnr,
            })
        
        print("\n" + "="*80)
        print("Результаты группированных метрик:")
        print("\nAccuracy Groups:")
        for res in grouped_acc_res:
            print(f"  {res['task_type']} - {res['dynamic_augments']}: {res['value']:.3f}")
        
        print("\nTFNR Groups:")
        for res in grouped_tfnr_res:
            print(f"  {res['task_type']} - {res['dynamic_augments']}: {res['value']:.3f}")

if __name__ == "__main__":
    test_grouped_metrics() 