# Contributing to RAG Assistant

Thank you for your interest in contributing to **RAG Assistant**!   
This project is meant to showcase a production-ready Retrieval-Augmented Generation (RAG) pipeline and welcomes improvements, bug fixes, and documentation updates.

---

## How to Get Started

**Read the [README.md](README.md)**  
   The README contains detailed instructions on:
   - Project setup (venv, dependencies, `.env` configuration)
   - Running ingestion, CLI queries, and API server
   - Cleaning and rebuilding the vector store
   - Running tests

**Fork and Clone**
```bash
git clone https://github.com/mohanelango/rag-assistant.git
cd rag-assistant
```

**Create a branch**
```bash
git checkout -b feature/my-feature
```
**Run Tests Before Committing**
```bash
make test VECTORSTORE_DIR=./vectorstore/test_index
```
### Guidelines

Follow PEP8 code style. Run black src tests before committing.

Keep commits descriptive (e.g., feat: add FAISS support, fix: close Chroma client).

Add/update tests for any new feature or bug fix.

Update documentation if you change configs or add new functionality.

### Submitting Your Work
Push your branch and open a Pull Request.

Include:

A short description of your change

Screenshots or logs (if relevant)

Confirmation that all tests pass locally

---