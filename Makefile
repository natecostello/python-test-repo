init:
	pyenv local $(shell cat .python-version)
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && black src tests && flake8
