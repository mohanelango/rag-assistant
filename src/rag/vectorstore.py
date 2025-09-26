from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """
    Build a HuggingFace embedding model.

    Args:
        model_name (str): Name of sentence-transformers model.

    Returns:
        HuggingFaceEmbeddings: Embedding function.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_or_load_vectorstore(
        vs_type: str,
        persist_dir: str,
        chroma_dir: str,
        embeddings: HuggingFaceEmbeddings,
        chunks: list[Document] | None = None,
):
    """
    Create or load a vector store (Chroma or FAISS).

    Args:
        vs_type (str): "chroma" or "faiss".
        persist_dir (str): FAISS storage path.
        chroma_dir (str): Chroma storage directory.
        embeddings: Embedding function.
        chunks (list[Document] | None): Optional documents for initial population.

    Returns:
        VectorStore: A vector store instance ready for use.
    """

    if vs_type == "faiss":
        Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)
        if chunks:
            return FAISS.from_documents(chunks, embedding=embeddings)
        else:
            raise ValueError("FAISS requires chunks to build; use FAISS.load_local to load.")
    elif vs_type == "chroma":
        Path(chroma_dir).parent.mkdir(parents=True, exist_ok=True)
        if chunks:
            return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=chroma_dir)
        else:
            return Chroma(embedding_function=embeddings, persist_directory=chroma_dir)
    else:
        raise ValueError("vectorstore.type must be 'faiss' or 'chroma'")
