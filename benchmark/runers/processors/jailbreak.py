from collections import defaultdict

import pandas as pd
from dataset_processor import DatasetProcessor


class JailbreakProcessor(DatasetProcessor):
    COLLECTION_NAME = "jailbreak"

    def extract_results(self, df):
        res = defaultdict(list)
        for model, group_df in df.groupby("model"):
            for _, row in group_df.iterrows():
                res[model].append(
                    {
                        "prompt": row["prompt"],
                        "res": row["response"],
                        "label": row["label"],
                    }
                )

        from trustllm import config
        from trustllm.task import safety

        config.device = "cuda:0"  # Или 'cpu', если нет GPU
        evaluator = safety.SafetyEval()

        out_data = []
        for model, data in res.items():
            data_processed, _ = evaluator.jailbreak_eval(
                data, eval_type="total", return_data=True
            )
            model_df = df[df["model"] == model].copy()
            model_df["res"] = [item["jailbreak"] for item in data_processed]
            out_data.append(model_df)

        if out_data:
            df = pd.concat(out_data)
        return df

    def calculate_metrics(self, df):
        metrics = df.groupby("model")["res"].mean().reset_index().to_dict("records")
        for record in metrics:
            record.update({"dataset": "jailbreak", "value": record.pop("res")})
        return metrics
