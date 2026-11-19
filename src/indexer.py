"""Corpus ingestion and indexing pipeline."""

import json
import os
import time
from typing import List
from tqdm import tqdm

from src.chunker import CodeChunker, Chunk


class Indexer:
    """Orchestrates corpus scanning, chunking, and persistence."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        max_chunk_size: int = 1500,
    ) -> None:
        """Initialize indexer paths and chunker configuration."""
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.max_chunk_size = min(max_chunk_size, 1500)
        self.chunker = CodeChunker(
            max_chunk_size=self.max_chunk_size,
            overlap=min(300, self.max_chunk_size // 4),
        )

    def run(self) -> List[Chunk]:
        """Scan raw directory, chunk files, and save index artifacts."""
        os.makedirs(self.processed_dir, exist_ok=True)
        if not os.path.exists(self.raw_dir):
            print(f"Warning: Raw directory '{self.raw_dir}' does not exist.")
            self._save_chunks([])
            return []

        valid_exts = {
            ".py", ".md", ".markdown", ".rst", ".txt", ".sh",
            ".yaml", ".yml", ".toml",
        }
        valid_names = {"cmakelists.txt", "dockerfile", "license", "dco", "readme"}

        file_paths: List[str] = []
        for root, dirs, files in os.walk(self.raw_dir):
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d != "__pycache__"
            ]
            for file in files:
                if file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                fname_lower = file.lower()
                if ext in valid_exts or fname_lower in valid_names:
                    file_paths.append(os.path.join(root, file))

        all_chunks: List[Chunk] = []
        start_time = time.time()

        for f_path in tqdm(file_paths, desc="Chunking corpus files"):
            chunks = self.chunker.chunk_file(f_path, root_dir=None)
            all_chunks.extend(chunks)

        elapsed = time.time() - start_time
        print(
            f"Successfully indexed {len(file_paths)} files into "
            f"{len(all_chunks)} chunks in {elapsed:.2f} seconds."
        )

        self._save_chunks(all_chunks)
        from src.search_engine import BM25SearchEngine
        engine = BM25SearchEngine(all_chunks)
        engine.save_index(self.processed_dir)
        return all_chunks

    def _save_chunks(self, chunks: List[Chunk]) -> None:
        """Persist chunks to JSON file in processed directory."""
        chunks_file = os.path.join(self.processed_dir, "chunks.json")
        data = [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "first_character_index": c.first_character_index,
                "last_character_index": c.last_character_index,
                "text": c.text,
                "file_type": c.file_type,
            }
            for c in chunks
        ]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_chunks(processed_dir: str = "data/processed") -> List[Chunk]:
        """Load persisted chunks from processed directory."""
        chunks_file = os.path.join(processed_dir, "chunks.json")
        if not os.path.exists(chunks_file):
            return []
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [
                Chunk(
                    chunk_id=item["chunk_id"],
                    file_path=item["file_path"],
                    first_character_index=item["first_character_index"],
                    last_character_index=item["last_character_index"],
                    text=item["text"],
                    file_type=item.get("file_type", "text"),
                )
                for item in data
            ]
        except Exception:
            return []
