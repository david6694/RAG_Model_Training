.PHONY: install run debug clean lint lint-strict eval

install:
	uv sync --extra dev

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache *.egg-info build dist .venv

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src

eval:
	@echo "=== 1/2 : SEARCH & EVALUATION DOCS ==="
	@uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
	@uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
	@echo "=== 2/2 : SEARCH & EVALUATION CODE ==="
	@uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
	@uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
