"""CLI entry point for RAG_Model_training using Python Fire."""

import uuid
import fire

from src.evaluator import Evaluator
from src.generator import AnswerGenerator
from src.indexer import Indexer
from src.search_engine import BM25SearchEngine


class RagCLI:
    """RAG_Model_training CLI Interface."""

    def __init__(self) -> None:
        """Initialize CLI with lazy loading for search engine."""
        self._engine: BM25SearchEngine | None = None

    def _get_engine(self) -> BM25SearchEngine:
        if self._engine is None:
            self._engine = BM25SearchEngine.load_index()
            if not self._engine.chunks:
                chunks = Indexer.load_chunks()
                self._engine = BM25SearchEngine(chunks)
        return self._engine

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest raw corpus and create processed search index."""
        indexer = Indexer(max_chunk_size=max_chunk_size)
        indexer.run()

    def search(self, query: str = "", k: int = 10) -> None:
        """Search indexed corpus for a given query."""
        engine = self._get_engine()
        res = engine.search(question_id=str(uuid.uuid4()), question=query, k=k)
        print(res.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Perform batch search on dataset questions."""
        engine = self._get_engine()
        evaluator = Evaluator(engine)
        evaluator.search_dataset(
            dataset_path=dataset_path, k=k, save_directory=save_directory
        )

    def answer(self, query: str = "", k: int = 10) -> None:
        """Retrieve sources and generate an answer for a single query."""
        engine = self._get_engine()
        search_res = engine.search(
            question_id=str(uuid.uuid4()), question=query, k=k
        )
        chunks = Indexer.load_chunks()
        chunks_map = {
            (
                f"{c.file_path}:"
                f"{c.first_character_index}:"
                f"{c.last_character_index}"
            ): c
            for c in chunks
        }
        generator = AnswerGenerator()
        ans = generator.answer_search_results(
            search_res, chunks_map=chunks_map
        )
        print(ans.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a batch search results JSON file."""
        chunks = Indexer.load_chunks()
        chunks_map = {
            (
                f"{c.file_path}:"
                f"{c.first_character_index}:"
                f"{c.last_character_index}"
            ): c
            for c in chunks
        }
        generator = AnswerGenerator()
        generator.batch_answer_dataset(
            student_search_results_path=student_search_results_path,
            save_directory=save_directory,
            chunks_map=chunks_map,
        )

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
    ) -> None:
        """Evaluate search recall metrics against ground truth dataset."""
        Evaluator.evaluate(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path,
        )


def main() -> None:
    """Run CLI application."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
