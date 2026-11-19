# RAG_Model_training 🤖⚡

High-performance, crash-proof RAG (Retrieval-Augmented Generation) system built for indexing, searching, and answering technical queries over Python source code and Markdown documentation from the vLLM corpus.

Developed strictly in accordance with 42 School subject specifications.

---

## 🛠️ Technical Stack

- **Package & Dependency Manager**: `uv` (exclusive package manager)
- **Python Version**: `3.10+` (isolated in `.venv`)
- **CLI Framework**: `Python Fire`
- **Data Validation & Schemas**: `Pydantic` v2 (`src/models.py`)
- **Default LLM Model**: `Qwen/Qwen3-0.6B` (with robust extractive fallback)
- **Search Engine**: BM25 algorithm with Code-Aware tokenization (`camelCase` & `snake_case`) and file path boosting
- **Progress Tracking**: `tqdm`
- **Code Quality**: `flake8` (PEP 8) & `mypy` (strict type checking)

---

## 📁 Project Structure

```text
.
├── Makefile                      # Standard 42 Makefile targets
├── README.md                     # Comprehensive project documentation
├── pyproject.toml                # UV project & dependency configuration
├── uv.lock                       # Deterministic lockfile
├── .flake8                       # PEP8 linter configuration
├── datasets_public/              # Public datasets for evaluation & testing
│   └── public/
│       ├── AnsweredQuestions/    # Ground truth reference datasets (docs & code)
│       └── UnansweredQuestions/  # Question datasets for retrieval & answering
├── data/
│   ├── raw/                      # Raw corpus files (.py, .md)
│   ├── processed/                # Saved index (chunks.json & bm25_engine.pkl)
│   ├── datasets/                 # Local question datasets
│   └── output/
│       ├── search_results/       # Generated StudentSearchResults JSON
│       └── search_results_and_answer/ # Generated StudentSearchResultsAndAnswer JSON
└── src/                          # Executable python module (uv run python -m src)
    ├── __init__.py
    ├── __main__.py               # Python Fire CLI entry point
    ├── chunker.py                # Bimodal AST & Markdown Chunker
    ├── evaluator.py              # Batch runner & Recall@k Benchmark
    ├── generator.py              # Qwen LLM Answer Generator with Fallback
    ├── indexer.py                # Ingestion & Index persistence
    ├── models.py                 # Pydantic v2 schemas
    └── search_engine.py          # Code-aware BM25 Search Engine
```

---

## 🏗️ System Architecture

The pipeline consists of four modular components:

```
[ Raw Corpus (data/raw/) ]
           │
           ▼
  [ 1. CodeChunker ]  ────────►  [ 2. BM25SearchEngine ]
  (Python AST + Markdown)        (Code-aware Tokenization & Boosting)
                                           │
                                           ▼ Top-k Sources
                                 [ 3. AnswerGenerator ]
                                 (Qwen / Fallback LLM)
                                           │
                                           ▼
                                [ 4. Output Schemas ]
                                (Pydantic v2 JSON Models)
```

1. **Ingestion & Bimodal Chunking (`src/chunker.py`, `src/indexer.py`)**:
   - Parses Python files using Python's `ast` module into functional units (classes, functions, methods).
   - Parses Markdown files by hierarchical sections (`#`, `##`, `###`) and paragraphs.
   - Enforces a strict upper bound `max_chunk_size` (default: 2000 characters) and records exact character offsets `(first_character_index, last_character_index)`.

2. **Lexical Retrieval Engine (`src/search_engine.py`)**:
   - Implements BM25Okapi scoring.
   - Splits code identifiers into constituent tokens (`camelCase` -> `camel`, `case`; `snake_case` -> `snake`, `case`).
   - Applies lexical boosting when query tokens match file and module path names.

3. **Grounded Answer Generation (`src/generator.py`)**:
   - Prompts the `Qwen/Qwen3-0.6B` model with retrieved context snippets.
   - Constrains context length to avoid hallucinations and maintain answer relevance.
   - Features an automated extractive fallback mechanism to guarantee zero crashes during offline execution or resource constraints.

4. **Batch Execution & Evaluation (`src/evaluator.py`)**:
   - Executes dataset-wide batch search and answer operations with `tqdm` progress tracking.
   - Calculates Recall@k based on character span overlap against ground truth sources.

---

## ✂️ Chunking Strategies & Trade-offs

The chunking layer implements two distinct, tailored strategies:

- **Python AST Strategy**: Retains semantic integrity of Python blocks (functions, classes, docstrings). If a function exceeds `max_chunk_size`, line-level sliding windows are applied with an overlap of 200 characters.
- **Markdown / Text Strategy**: Preserves logical header structures and thematic paragraphs.

### Chunk Size Impact Analysis:
- **Too Small (< 200 chars)**: Snips context into fragments, stripping out definitions, import contexts, and variable usages, resulting in lower retrieval precision and fragmented generation.
- **Too Large (> 2000 chars)**: Dilutes keyword density for BM25 retrieval, risks truncation by LLM context windows, and violates evaluation constraints (`max_context_length = 2000`).

---

## 🔍 Retrieval & Ranking Details

- **BM25Okapi Algorithm**: Balances Term Frequency (TF) saturation with Inverse Document Frequency (IDF) document length normalization.
- **Code-Aware Tokenizer**:
  - `LLMEngine` $\rightarrow$ `['llm', 'engine', 'llmengine']`
  - `async_get_model_config` $\rightarrow$ `['async', 'get', 'model', 'config']`
- **File Path Boosting**: Adds a multiplier bonus when query tokens overlap with module or file path names (e.g., `openai_compatible_server.md` or `llm_engine.py`).

---

## 🚀 Installation & Virtual Environment

The project runs exclusively inside the `.venv` virtual environment managed by `uv`.

```bash
# Sync dependencies and development tools
make install

# Activate virtual environment manually (optional)
source .venv/bin/activate
```

---

## 💻 CLI Commands Usage

All commands are executed via `uv run python -m src <command>`:

### 1. Ingestion & Indexing
Ingests the raw corpus under `data/raw/` and builds the index in `data/processed/`.

```bash
uv run python -m src index --max_chunk_size 2000
```

### 2. Single Question Search
Retrieves the top-k source attributions for a query.

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

### 3. Batch Search on Public Dataset
Executes batch search over a dataset from `datasets_public` and outputs `StudentSearchResults` JSON.

```bash
# Docs Dataset
uv run python -m src search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# Code Dataset
uv run python -m src search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Single Question Answering
Retrieves top-k context and generates a grounded response.

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

### 5. Batch Answer Generation
Generates grounded answers for all questions in a search results file, producing `StudentSearchResultsAndAnswer` JSON.

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 6. Recall@k Evaluation
Evaluates search recall against reference ground-truth datasets.

```bash
# Evaluate Docs Retrieval
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json

# Evaluate Code Retrieval
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json

# Or evaluate both Docs & Code simultaneously with Makefile shortcut:
make eval
```

---

## 🛡️ Reliability & Edge Cases (Crash-Proofing)

The system gracefully handles all edge cases without raising unhandled Python Tracebacks:
- **Empty Queries**: `search ""` and `answer ""`
- **Nonsensical / Gibberish Input**: `search "asdfghjkl"`
- **Zero / Negative k values**: `search "query" --k 0` and `answer "query" --k -5`
- **Missing or Invalid Paths**: `search_dataset --dataset_path /nonexistent.json`

---

## 🧪 Quality & Strict Linting

```bash
# Run standard linting (flake8 + mypy)
make lint

# Run strict type checking (mypy --strict)
make lint-strict

# Clean caches and build artifacts
make clean
```
