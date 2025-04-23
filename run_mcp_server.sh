#!/bin/sh
cd "$(dirname "$0")"
export PATH="/Users/kenanagyel/.pyenv/versions/3.12.8/bin:$PATH"
/Users/kenanagyel/.pyenv/versions/3.12.8/bin/poetry run python -m src.core.main --server --port 9000
