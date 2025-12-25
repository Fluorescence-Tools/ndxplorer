#!/usr/bin/env bash
set -euxo pipefail

# Install ndxplorer using pyproject.toml (entry points defined there)
"$PYTHON" -m pip install . --no-deps -vv --prefix="$PREFIX"