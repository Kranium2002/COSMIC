.PHONY: install format lint test bench build-ext clean-ext

install:
	poetry install

format:
	poetry run ruff format .

lint:
	poetry run ruff check .

test:
	poetry run pytest

bench:
	poetry run python benchmark_llm_finetune.py

build-ext:
	poetry run python -c "from cosmic.extension import get_extension; get_extension(verbose=True)"

clean-ext:
	rm -rf cosmic/extension/_build
