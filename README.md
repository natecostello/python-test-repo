# Python Project Template

This is a lightweight, VS Code-ready Python template using pyenv + venv.

## Setup

### Recommended (simplest):
```bash
make init
```

This will:
- Check that the Python version in `.python-version` is installed via pyenv
- Create a `.venv` folder
- Install dependencies from `requirements.txt`

### Alternative (manual setup):
If you'd rather run the setup script directly:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

## Usage
- Code in `src/`
- Tests in `tests/`

## Activating the Virtual Env
```bash
source .venv/bin/activate
```

## Tools
- `black` for formatting
- `flake8` for linting
- `pytest` for testing

## Git Pre-commit Hooks
This project supports [pre-commit](https://pre-commit.com) to enforce formatting and linting at commit time.

### Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

After this, `black` and `flake8` will automatically run on staged files before each commit.
