import re

from .dataset_processor import DatasetProcessor


class OODProcessor(DatasetProcessor):
    COLLECTION_NAME = "ood_check_1"

    def extract_results(self, df):
        df = df[df["task_name"] == "ood_check_1_check"]
        df["response"] = df["response"].astype(str)
        df["res"] = df["response"].apply(self.extract_yes_no)
        return df

    def calculate_metrics(self, df):
        metrics = (
            df.groupby("init_model")["res"].mean().reset_index().to_dict("records")
        )
        for record in metrics:
            record.update(
                {
                    "dataset": "ood",
                    "model": record.pop("init_model"),
                    "value": record.pop("res"),
                }
            )
        return metrics

    @staticmethod
    def extract_yes_no(response):
        yes_variants = ["да", "yes", "да.", "da"]
        no_variants = ["нет", "no", "нет.", "not"]
        response_lower = response.lower()
        if any(re.search(rf"\b{word}\b", response_lower) for word in yes_variants):
            return 1
        elif any(re.search(rf"\b{word}\b", response_lower) for word in no_variants):
            return 0
        return None
