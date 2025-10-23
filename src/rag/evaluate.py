import json
import argparse
from .utils import load_yaml
from .chain import build_rag_chain
from .evaluation import precision_at_k, recall_at_k, mean_reciprocal_rank
from langchain_huggingface import HuggingFaceEmbeddings


def main(settings_path: str, eval_file: str):
    settings = load_yaml(settings_path)
    embeddings = HuggingFaceEmbeddings(model_name=settings["embeddings"]["model_name"])
    chain, retriever = build_rag_chain(settings, embeddings)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    precisions, recalls = [], []
    all_ranks = []
    for item in eval_data:
        question = item["question"]
        relevant = item["relevant_docs"]
        results = retriever.vectorstore.similarity_search(question, k=5)

        precisions.append(precision_at_k(results, relevant))
        recalls.append(recall_at_k(results, relevant))
        all_ranks.append([d.metadata.get("source") for d in results])

    mrr = mean_reciprocal_rank(all_ranks, [r for d in eval_data for r in d["relevant_docs"]])

    print(f"Precision@5: {sum(precisions) / len(precisions):.2f}")
    print(f"Recall@5: {sum(recalls) / len(recalls):.2f}")
    print(f"MRR: {mrr:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--settings", required=True, help="Path to settings.yaml")
    parser.add_argument("--evalset", required=True, help="Path to evaluation JSON file")
    args = parser.parse_args()
    main(args.settings, args.evalset)
