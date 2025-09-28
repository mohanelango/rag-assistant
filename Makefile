# Makefile for RAG Assistant
# Uses Python module execution with default configs but allows overrides

SETTINGS ?= configs/settings.yaml
SOURCES  ?= configs/sources.yaml
VECTORSTORE_DIR ?=
Q ?=

.PHONY: ingest ask serve test clean

## Build & persist vector store (Chroma by default)
ingest:
	@if [ -n "$(VECTORSTORE_DIR)" ]; then \
		echo ">>> Ingesting documents into custom vectorstore: $(VECTORSTORE_DIR)"; \
		python -m src.rag.ingest --settings $(SETTINGS) --sources $(SOURCES) --vectorstore $(VECTORSTORE_DIR); \
	else \
		echo ">>> Ingesting documents (using $(SETTINGS) & $(SOURCES))..."; \
		python -m src.rag.ingest --settings $(SETTINGS) --sources $(SOURCES); \
	fi

## Ask a question via CLI
ask:
ifndef Q
	$(error Q (question) is required: make ask Q="What is RAG?")
endif
	@echo ">>> Asking: $(Q)"
	python -m src.rag.cli --question "$(Q)" --settings $(SETTINGS)

## Run FastAPI server
serve:
	@echo ">>> Starting FastAPI server..."
	uvicorn src.rag.api:app --reload --port 8000

## Run tests with optional temp vectorstore
test:
	@if [ -n "$(VECTORSTORE_DIR)" ]; then \
		echo ">>> Running tests with custom vectorstore: $(VECTORSTORE_DIR)"; \
		VECTORSTORE_DIR=$(VECTORSTORE_DIR) pytest -v; \
	else \
		echo ">>> Running tests with default settings"; \
		pytest -v; \
	fi

## Remove vectorstore data (reset index)
clean:
	@echo ">>> Cleaning vectorstore data..."
	@if [ -n "$(VECTORSTORE_DIR)" ]; then \
		rm -rf "$(VECTORSTORE_DIR)"; \
	else \
		rm -rf vectorstore/chroma_index vectorstore/faiss_index; \
	fi
