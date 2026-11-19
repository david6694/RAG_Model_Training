"""Bimodal chunking module for Python source code and Markdown documentation."""

import ast
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from src.models import MinimalSource


@dataclass
class Chunk:
    """Internal chunk model with attribution and content."""

    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_type: str

    def to_minimal_source(self) -> MinimalSource:
        """Convert to MinimalSource Pydantic model."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )


class CodeChunker:
    """Bimodal chunker supporting Python AST and Markdown section splitting."""

    def __init__(self, max_chunk_size: int = 1500, overlap: int = 300) -> None:
        """Initialize chunker parameters."""
        self.max_chunk_size = max(100, max_chunk_size)
        self.overlap = min(overlap, self.max_chunk_size // 2)

    def chunk_file(
        self, file_path: str, root_dir: Optional[str] = None
    ) -> List[Chunk]:
        """Ingest and chunk a single file based on extension.

        Gracefully returns an empty list if file reading fails.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        if not content.strip():
            return []

        rel_path = (
            os.path.relpath(file_path, root_dir)
            if root_dir
            else os.path.normpath(file_path)
        ).replace("\\", "/")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".py":
            return self._chunk_python(content, rel_path)
        elif ext in (".md", ".markdown", ".rst", ".txt"):
            return self._chunk_markdown(content, rel_path)
        else:
            return self._chunk_text(content, rel_path)

    def _get_line_offsets(self, content: str) -> List[int]:
        """Compute cumulative character offset for start of each line."""
        lines = content.splitlines(keepends=True)
        offsets = [0]
        curr = 0
        for line in lines:
            curr += len(line)
            offsets.append(curr)
        return offsets

    def _chunk_python(self, content: str, rel_path: str) -> List[Chunk]:
        """Chunk Python code using AST parsing with method-level granularity."""
        line_offsets = self._get_line_offsets(content)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        total_len = len(content)

        if total_len <= self.max_chunk_size:
            return [
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    file_path=rel_path,
                    first_character_index=0,
                    last_character_index=total_len,
                    text=content,
                    file_type="python",
                )
            ]

        chunks: List[Chunk] = []
        try:
            tree = ast.parse(content)
            blocks: List[tuple[int, int]] = []

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_l = node.lineno - 1
                    end_l = getattr(node, "end_lineno", total_lines)
                    blocks.append((start_l, end_l))
                elif isinstance(node, ast.ClassDef):
                    c_start = node.lineno - 1
                    c_end = getattr(node, "end_lineno", total_lines)
                    c_len = line_offsets[c_end] - line_offsets[c_start]
                    if c_len <= self.max_chunk_size:
                        blocks.append((c_start, c_end))
                    else:
                        method_found = False
                        for subnode in node.body:
                            if isinstance(
                                subnode,
                                (ast.FunctionDef, ast.AsyncFunctionDef),
                            ):
                                m_start = subnode.lineno - 1
                                m_end = getattr(
                                    subnode, "end_lineno", total_lines
                                )
                                blocks.append((m_start, m_end))
                                method_found = True
                        if not method_found:
                            blocks.append((c_start, c_end))

            # Non-overlapping sorted blocks
            blocks.sort(key=lambda b: b[0])
            merged_blocks: List[tuple[int, int]] = []
            for b in blocks:
                if not merged_blocks or b[0] >= merged_blocks[-1][1]:
                    merged_blocks.append(b)
                else:
                    merged_blocks[-1] = (
                        merged_blocks[-1][0],
                        max(merged_blocks[-1][1], b[1]),
                    )

            curr_start = 0
            for start_l, end_l in merged_blocks:
                if start_l > curr_start:
                    gap_chunks = self._sliding_window_lines(
                        content,
                        lines,
                        line_offsets,
                        curr_start,
                        start_l,
                        rel_path,
                        "python",
                    )
                    chunks.extend(gap_chunks)
                block_chunks = self._sliding_window_lines(
                    content,
                    lines,
                    line_offsets,
                    start_l,
                    end_l,
                    rel_path,
                    "python",
                )
                chunks.extend(block_chunks)
                curr_start = max(curr_start, end_l)

            if curr_start < total_lines:
                gap_chunks = self._sliding_window_lines(
                    content,
                    lines,
                    line_offsets,
                    curr_start,
                    total_lines,
                    rel_path,
                    "python",
                )
                chunks.extend(gap_chunks)

            seen: set[tuple[int, int]] = set()
            unique_chunks: List[Chunk] = []
            for c in chunks:
                key = (c.first_character_index, c.last_character_index)
                if key not in seen and c.text.strip():
                    seen.add(key)
                    unique_chunks.append(c)
            if unique_chunks:
                return unique_chunks
        except Exception:
            pass

        return self._sliding_window_lines(
            content, lines, line_offsets, 0, total_lines, rel_path, "python"
        )

    def _chunk_markdown(self, content: str, rel_path: str) -> List[Chunk]:
        """Chunk Markdown text by headings and sections with merging."""
        line_offsets = self._get_line_offsets(content)
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        total_len = len(content)

        if total_len <= self.max_chunk_size:
            return [
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    file_path=rel_path,
                    first_character_index=0,
                    last_character_index=total_len,
                    text=content,
                    file_type="markdown",
                )
            ]

        header_lines: List[int] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                header_lines.append(i)

        if not header_lines or header_lines[0] != 0:
            header_lines.insert(0, 0)
        if header_lines[-1] != total_lines:
            header_lines.append(total_lines)

        sections: List[tuple[int, int]] = []
        i = 0
        while i < len(header_lines) - 1:
            s_start = header_lines[i]
            s_end = header_lines[i + 1]

            while (
                i + 2 < len(header_lines)
                and (line_offsets[header_lines[i + 2]] - line_offsets[s_start])
                <= self.max_chunk_size
            ):
                i += 1
                s_end = header_lines[i + 1]

            sections.append((s_start, s_end))
            i += 1

        chunks: List[Chunk] = []
        for start_l, end_l in sections:
            chunks.extend(
                self._sliding_window_lines(
                    content,
                    lines,
                    line_offsets,
                    start_l,
                    end_l,
                    rel_path,
                    "markdown",
                )
            )

        seen: set[tuple[int, int]] = set()
        unique_chunks: List[Chunk] = []
        for c in chunks:
            key = (c.first_character_index, c.last_character_index)
            if key not in seen and c.text.strip():
                seen.add(key)
                unique_chunks.append(c)
        return unique_chunks

    def _chunk_text(self, content: str, rel_path: str) -> List[Chunk]:
        """Fallback chunker for plain text, configuration files, and scripts."""
        line_offsets = self._get_line_offsets(content)
        lines = content.splitlines(keepends=True)
        return self._sliding_window_lines(
            content, lines, line_offsets, 0, len(lines), rel_path, "text"
        )

    def _sliding_window_lines(
        self,
        content: str,
        lines: List[str],
        line_offsets: List[int],
        start_line: int,
        end_line: int,
        rel_path: str,
        file_type: str,
    ) -> List[Chunk]:
        """Chunk a range of lines with overlapping sliding windows."""
        if start_line >= end_line:
            return []

        start_char = line_offsets[start_line]
        end_char = line_offsets[end_line]
        range_len = end_char - start_char

        if range_len <= self.max_chunk_size:
            text = content[start_char:end_char]
            if text.strip():
                return [
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        file_path=rel_path,
                        first_character_index=start_char,
                        last_character_index=end_char,
                        text=text,
                        file_type=file_type,
                    )
                ]
            return []

        chunks: List[Chunk] = []
        curr_line = start_line
        while curr_line < end_line:
            c_start = line_offsets[curr_line]
            if end_char - c_start <= self.max_chunk_size:
                text = content[c_start:end_char]
                if text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            file_path=rel_path,
                            first_character_index=c_start,
                            last_character_index=end_char,
                            text=text,
                            file_type=file_type,
                        )
                    )
                break

            c_end_line = curr_line
            while (
                c_end_line < end_line
                and (line_offsets[c_end_line + 1] - c_start)
                <= self.max_chunk_size
            ):
                c_end_line += 1

            if c_end_line == curr_line:
                c_end_char = min(c_start + self.max_chunk_size, end_char)
                text = content[c_start:c_end_char]
                if text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            file_path=rel_path,
                            first_character_index=c_start,
                            last_character_index=c_end_char,
                            text=text,
                            file_type=file_type,
                        )
                    )
                next_char = c_start + max(
                    50, self.max_chunk_size - self.overlap
                )
                while (
                    curr_line < end_line and line_offsets[curr_line] < next_char
                ):
                    curr_line += 1
            else:
                c_end_char = line_offsets[c_end_line]
                text = content[c_start:c_end_char]
                if text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            file_path=rel_path,
                            first_character_index=c_start,
                            last_character_index=c_end_char,
                            text=text,
                            file_type=file_type,
                        )
                    )

                if c_end_line >= end_line:
                    break

                target_char = max(c_start + 1, c_end_char - self.overlap)
                next_line = curr_line + 1
                while (
                    next_line < c_end_line
                    and line_offsets[next_line] < target_char
                ):
                    next_line += 1
                curr_line = next_line

        return chunks
