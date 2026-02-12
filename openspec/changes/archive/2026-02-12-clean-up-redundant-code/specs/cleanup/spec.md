# Cleanup Spec

This change only performs cleanup operations without modifying any functional specifications.

## Requirements

1. Delete `src/legacy/` directory containing 15 deprecated Python scripts
2. Move `src/afml_polars_pipeline.py` to `examples/afml_polars_pipeline.py`
3. Update `pyproject.toml` scripts configuration

## Verification

After cleanup, verify:
- `src/legacy/` directory no longer exists
- `examples/afml_polars_pipeline.py` exists
- `uv run python examples/afml_polars_pipeline.py --help` works
- `pyproject.toml` references new location
