.PHONY: run test lint docker-build docker-run clean help

## Run the Streamlit app locally
run:
	streamlit run app.py

## Run the agent directly (CLI demo)
demo:
	python agent.py

## Run all tests with coverage report
test:
	pytest tests/ -v --tb=short --cov=agent --cov-report=term-missing

## Lint the codebase (requires ruff: pip install ruff)
lint:
	ruff check agent.py app.py

## Build Docker image
docker-build:
	docker build -t contentkosh-agent:latest .

## Run Docker container (pass OPENAI_API_KEY from env)
docker-run:
	docker run -p 8501:8501 -e OPENAI_API_KEY=$(OPENAI_API_KEY) contentkosh-agent:latest

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

## Show available commands
help:
	@grep -E '^##' Makefile | sed 's/## //'
