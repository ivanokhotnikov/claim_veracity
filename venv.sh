#!/bin/sh
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools
pip install -r conf/requirements-train.txt
