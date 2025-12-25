@echo off
setlocal enabledelayedexpansion

REM Install ndxplorer using pyproject.toml (entry points defined there)
"%PYTHON%" -m pip install . --no-deps -vv --prefix="%PREFIX%"

endlocal