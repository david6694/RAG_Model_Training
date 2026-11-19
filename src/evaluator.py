"""Evaluator module for batch search generation and Recall@k benchmarking."""

import json
import os
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.search_engine import BM25SearchEngine


def is_source_retrieved(
    gt_source: MinimalSource,
    retrieved_sources: List[MinimalSource],
    iou_threshold: float = 0.05,
) -> bool:
    """Check if ground truth source matches retrieved source with IoU > threshold."""
    for ret in retrieved_sources:
        if gt_source.file_path != ret.file_path:
            continue
        inter_start = max(
            gt_source.first_character_index, ret.first_character_index
        )
        inter_end = min(
            gt_source.last_character_index, ret.last_character_index
        )
        if inter_start < inter_end:
            intersection = inter_end - inter_start
            union = (
                max(gt_source.last_character_index, ret.last_character_index)
                - min(
                    gt_source.first_character_index,
                    ret.first_character_index,
                )
            )
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                return True
    return False


class Evaluator:
    """Batch search runner and recall metric calculator."""

    def __init__(self, search_engine: BM25SearchEngine) -> None:
        """Initialize with search engine instance."""
        self.engine = search_engine

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> StudentSearchResults:
        """Run search on a dataset of questions and save results."""
        os.makedirs(save_directory, exist_ok=True)
        results: List[MinimalSearchResults] = []

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path '{dataset_path}' does not exist.")
            empty_res = StudentSearchResults(search_results=[], k=k)
            self._save_results(empty_res, save_directory)
            return empty_res

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            dataset = RagDataset.model_validate(raw_data)
        except Exception as e:
            print(f"Error loading dataset '{dataset_path}': {e}")
            empty_res = StudentSearchResults(search_results=[], k=k)
            self._save_results(empty_res, save_directory)
            return empty_res

        target_type: Optional[str] = None
        dpath_lower = dataset_path.lower()
        if "docs" in dpath_lower:
            target_type = "docs"
        elif "code" in dpath_lower:
            target_type = "code"

        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            search_res = self.engine.search(
                question_id=q.question_id,
                question=q.question,
                k=k,
                target_type=target_type,
            )
            results.append(search_res)

        filename = (
            os.path.basename(dataset_path)
            if dataset_path
            else "student_search_results.json"
        )
        student_res = StudentSearchResults(search_results=results, k=k)
        self._save_results(student_res, save_directory, filename=filename)
        return student_res

    def _save_results(
        self,
        student_res: StudentSearchResults,
        save_directory: str,
        filename: str = "student_search_results.json",
    ) -> str:
        """Save search results to JSON file."""
        if save_directory.endswith(".json"):
            out_file = save_directory
            dir_name = os.path.dirname(out_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        else:
            os.makedirs(save_directory, exist_ok=True)
            out_file = os.path.join(save_directory, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(student_res.model_dump_json(indent=2))
        print(f"Saved search results to '{out_file}'.")
        return out_file

    @staticmethod
    def evaluate(
        student_search_results_path: str, dataset_path: str
    ) -> Dict[str, Any]:
        """Compute Recall@k for k in [1, 3, 5, 10] with decimal and percentage."""
        if not os.path.exists(student_search_results_path) or not os.path.exists(
            dataset_path
        ):
            print("Error: Missing search results or dataset file for evaluation.")
            return {"overall_recall": 0.0, "docs_recall": 0.0, "code_recall": 0.0}

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            student_res = StudentSearchResults.model_validate(s_data)

            with open(dataset_path, "r", encoding="utf-8") as f:
                d_data = json.load(f)
            dataset = RagDataset.model_validate(d_data)
        except Exception as e:
            print(f"Error loading files for evaluation: {e}")
            return {"overall_recall": 0.0, "docs_recall": 0.0, "code_recall": 0.0}

        res_map = {
            r.question_id: r.retrieved_sources for r in student_res.search_results
        }

        k_levels = [1, 3, 5, 10]
        k_levels_to_eval = [
            kv for kv in k_levels if kv <= max(student_res.k, 1)
        ]
        if not k_levels_to_eval:
            k_levels_to_eval = [student_res.k]

        results_by_k: Dict[int, Dict[str, float]] = {}

        for k_val in k_levels_to_eval:
            docs_recalls: List[float] = []
            code_recalls: List[float] = []
            overall_recalls: List[float] = []

            for q in dataset.rag_questions:
                if not isinstance(q, AnsweredQuestion):
                    continue
                retrieved = res_map.get(q.question_id, [])[:k_val]
                if not q.sources:
                    continue

                hits = sum(
                    1 for gt in q.sources if is_source_retrieved(gt, retrieved)
                )
                q_recall = hits / len(q.sources)
                overall_recalls.append(q_recall)

                is_code = any(s.file_path.endswith(".py") for s in q.sources)
                if is_code:
                    code_recalls.append(q_recall)
                else:
                    docs_recalls.append(q_recall)

            avg_overall = (
                sum(overall_recalls) / len(overall_recalls)
                if overall_recalls
                else 0.0
            )
            avg_docs = (
                sum(docs_recalls) / len(docs_recalls) if docs_recalls else 0.0
            )
            avg_code = (
                sum(code_recalls) / len(code_recalls) if code_recalls else 0.0
            )

            results_by_k[k_val] = {
                "overall": avg_overall,
                "docs": avg_docs,
                "code": avg_code,
            }

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k_val in k_levels_to_eval:
            ov = results_by_k[k_val]["overall"]
            dc = results_by_k[k_val]["docs"]
            cd = results_by_k[k_val]["code"]

            extra_info = ""
            if k_val == 5:
                if dc > 0 and cd == 0:
                    extra_info = "  (Docs Goal >= 80.00%)"
                elif cd > 0 and dc == 0:
                    extra_info = "  (Code Goal >= 50.00%)"
                elif dc > 0 and cd > 0:
                    extra_info = "  (Docs >= 80%, Code >= 50%)"

            print(
                f"Recall@{k_val:<2}: {ov:.3f} ({ov * 100:.2f}%){extra_info}"
            )
        print("=" * 60 + "\n")

        rec_5 = results_by_k.get(5, results_by_k.get(student_res.k, {}))
        return {
            "overall_recall": rec_5.get("overall", 0.0),
            "docs_recall": rec_5.get("docs", 0.0),
            "code_recall": rec_5.get("code", 0.0),
            "k": student_res.k,
            "results_by_k": results_by_k,
        }
