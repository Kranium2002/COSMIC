# Contributing

Thanks for contributing to `cosmic`.

## Development setup

```bash
poetry install
```

## Build the C++ extension

```bash
poetry run python -c "from cosmic.extension import get_extension; get_extension(verbose=True)"
```

## Lint and format

```bash
poetry run ruff format .
poetry run ruff check .
```

## Tests

```bash
poetry run pytest
```

## Benchmark

```bash
poetry run python benchmark_llm_finetune.py
```

## Pull request checklist
- Keep CPU-only constraints intact (no CUDA files or CUDA build logic)
- Add or update tests for functional changes
- Keep structured logging deterministic
- Update documentation if behavior changes
