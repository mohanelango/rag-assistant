# Sample Queries for RAG Assistant

Here are a few example queries to quickly test the system:

---

### General AI / RAG Queries
```bash
python -m src.rag.cli --question "What is Retrieval-Augmented Generation and why is it useful?"
```
### Domain-Specific Example
```bash
python -m src.rag.cli --question "Summarize the Ready Tensor publication on technical excellence"
```
### FAISS vs Chroma Benchmark
```bash
python -m src.rag.cli --question "Compare FAISS and Chroma as vector stores"
```
### Wikipedia Example
```bash
python -m src.rag.cli --question "Who invented FAISS?"
```