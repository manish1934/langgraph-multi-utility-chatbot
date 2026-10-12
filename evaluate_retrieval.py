"""
Retrieval evaluation for the RAG chatbot.

Usage:
    python evaluate_retrieval.py 1810_04805v2.pdf test_set.csv

Ye script LLM (Groq) use nahi karti, sirf retriever check karti hai.
Isliye rate limit ka koi issue nahi aata.
"""
import sys

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "1810_04805v2.pdf"
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "test_set.csv"

CHUNK_SIZES = [500, 800, 1000, 1500]
K_VALUES = [1, 3, 5, 8]

# Same embedding model as the chatbot
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = PyPDFLoader(PDF_PATH).load()
test_df = pd.read_csv(CSV_PATH)
max_k = max(K_VALUES)

print(f"PDF pages: {len(docs)} | Questions: {len(test_df)}")

rows = []
failures = {}  # (chunk_size, k) -> list of failed questions

for chunk_size in CHUNK_SIZES:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)

    # Retrieve once with max_k, then check top-k prefixes
    retrieved_pages = []
    for q in test_df["question"]:
        results = db.similarity_search(q, k=max_k)
        retrieved_pages.append([d.metadata["page"] + 1 for d in results])  # 0-based -> 1-based

    for k in K_VALUES:
        hits = 0
        missed = []
        for (_, row), pages in zip(test_df.iterrows(), retrieved_pages):
            if int(row["expected_page"]) in pages[:k]:
                hits += 1
            else:
                missed.append(row["question"])
        hit_rate = hits / len(test_df)
        rows.append({"chunk_size": chunk_size, "k": k, "hit_rate": round(hit_rate, 2)})
        failures[(chunk_size, k)] = missed

    print(f"chunk_size={chunk_size} done ({len(chunks)} chunks)")

results = pd.DataFrame(rows)
table = results.pivot(index="chunk_size", columns="k", values="hit_rate")
print("\nHit Rate@k (rows = chunk size, columns = k):\n")
print(table)

table.to_csv("eval_results.csv")

best = results.sort_values(["hit_rate", "k", "chunk_size"], ascending=[False, True, True]).iloc[0]
print(f"\nBest (highest hit rate, then smallest k): chunk_size={int(best.chunk_size)}, k={int(best.k)}, hit_rate={best.hit_rate}")

missed = failures[(int(best.chunk_size), int(best.k))]
if missed:
    print("\nQuestions that still failed in the best setting:")
    for q in missed:
        print(" -", q)