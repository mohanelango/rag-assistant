from typing import Dict, Any, List
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from .utils import get_env

SYSTEM_PROMPT = """You are a precise assistant. Use ONLY the provided context to answer.
If the answer is incomplete, say what is known and what is missing.
Cite sources from metadata (URL or Wikipedia title).

Question: {question}
Context:
{context}

Answer:"""


def format_docs(docs: List[Document]) -> str:
    """
    Concatenate retrieved docs with source markers.

    Args:
        docs (List[Document]): Retrieved documents.

    Returns:
        str: Combined context string.
    """
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_llm(settings: Dict[str, Any]):
    """
    Initialize a language model client.

    Supports OpenAI, Ollama, HuggingFace.

    Args:
        settings (dict): Model config from settings.yaml.

    Returns:
        Chat model instance.
    """
    provider = settings["model"]["provider"]
    if provider == "openai":
        return ChatOpenAI(
            model=settings["model"]["openai"]["model"],
            temperature=settings["model"]["openai"]["temperature"],
            api_key=get_env("OPENAI_API_KEY"),
        )
    elif provider == "ollama":
        return ChatOllama(
            base_url=settings["model"]["ollama"]["base_url"],
            model=settings["model"]["ollama"]["model"],
            temperature=settings["model"]["ollama"]["temperature"],
        )
    elif provider == "huggingface":
        return HuggingFaceEndpoint(
            repo_id=settings["model"]["huggingface"]["repo_id"],
            temperature=settings["model"]["huggingface"]["temperature"],
            huggingfacehub_api_token=get_env("HUGGINGFACEHUB_API_TOKEN"),
        )
    else:
        raise ValueError("Unsupported model provider")


def load_vectorstore(settings: Dict[str, Any], embeddings: HuggingFaceEmbeddings):
    """
    Load a persisted vector store.

    Args:
        settings (dict): Config with vectorstore paths.
        embeddings: Embedding function.

    Returns:
        VectorStore: Chroma or FAISS instance.
    """
    vs_type = settings["vectorstore"]["type"]
    if vs_type == "faiss":
        folder = settings["vectorstore"]["persist_dir"]
        if not Path(folder).exists():
            raise FileNotFoundError(f"FAISS folder not found: {folder}. Run ingestion first.")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    else:
        chroma_dir = settings["vectorstore"]["chroma_dir"]
        return Chroma(embedding_function=embeddings, persist_directory=chroma_dir)


def build_rag_chain(settings: Dict[str, Any], embeddings: HuggingFaceEmbeddings):
    """
    Build a LangChain RAG pipeline.

    Args:
        settings (dict): Configuration dict.
        embeddings: Embedding function.

    Returns:
        Tuple(chain, retriever): Runnable chain and retriever object.
    """
    vs = load_vectorstore(settings, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": settings["retrieval"]["k"]})
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    llm = build_llm(settings)

    def extract_question(input_dict):
        return input_dict["question"]

    chain = (
            {"context": extract_question | retriever | (lambda docs: format_docs(docs)),
             "question": RunnablePassthrough()}
            | prompt
            | llm
    )
    return chain, retriever
