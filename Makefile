# Makefile for RAG Assistant
# Uses Python module execution with default configs but allows overrides

SETTINGS ?= configs/settings.yaml
SOURCES  ?= configs/sources.yaml
Q ?=

.PHONY: ingest ask serve test clean

## Build & persist vector store (Chroma by default)
ingest:
	@echo ">>> Ingesting documents (using $(SETTINGS) & $(SOURCES))..."
	python -m src.rag.ingest --settings $(SETTINGS) --sources $(SOURCES)

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

## Remove vectorstore data (reset index)
clean:
	@echo ">>> Cleaning vectorstore data..."
	rm -rf vectorstore/chroma_index vectorstore/faiss_index
