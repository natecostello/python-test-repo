#!/bin/bash
set -e

REQUIRED_VERSION=$(cat .python-version)

# Check if the required Python version is installed
if ! pyenv versions --bare | grep -qx "$REQUIRED_VERSION"; then
  echo "❌ Python $REQUIRED_VERSION is not installed via pyenv."
  echo "👉 Run: pyenv install $REQUIRED_VERSION"
  exit 1
fi

# Proceed with environment setup
pyenv local "$REQUIRED_VERSION"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt