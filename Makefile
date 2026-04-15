SHELL := /bin/bash
PYTHON_VERSION := 3.11

PYTHON_INTERPRETER = python3

install:

	# Conda creerà l'ambiente con la versione specificata
	conda env create -f environment.yml python=$(PYTHON_VERSION)


clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete