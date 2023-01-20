include env.sh
export

default-retrain:
	python src/training.py

pre-commit:
	pre-commit run --all-files

venv:
	python -m venv .venv && \
	source .venv/Scripts/activate && \
	python -m pip install --upgrade pip setuptools && \
	pip install -r conf/requirements-train.txt
