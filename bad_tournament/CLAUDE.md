# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Run all tests: `python3 main.py --test`
- Run example tournament: `python3 main.py --run-example`
- Run with config file: `python3 main.py --config FILE.json`
- Save example config: `python3 main.py --save-example FILE.json`
- Export schedule to CSV: `python3 main.py --export-csv FILE.csv`
- Show pool structure options: `python3 main.py --suggest-pools N`
- Format code: `black /path/to/files/*.py`

## Code Style Guidelines
- Use Black formatter (v24.4.2) for consistent code style
- Use 4-space indentation (enforced by Black)
- Use snake_case for variables and functions
- Use CamelCase for classes
- Docstrings with triple double quotes
- Type hints with Python typing module
- Use dataclasses for data models
- Validate input in __post_init__ methods
- Raise ValueError with descriptive messages
- Comprehensive unit tests with unittest framework
- Use logging for informational output
- Always check for None with `is` operator, not `==`