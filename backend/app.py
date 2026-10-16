
# =========================================================
# AuroraStay Hotels - Flask RAG Backend API
# =========================================================

import os
from pathlib import Path

from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag_utils import retrieve, rag


# =========================================================
# Configuration
# =========================================================

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_PATH = BASE_DIR / "faiss_index"

DEFAULT_K = 3


# =========================================================
# Initialize Flask application
# =========================================================

rag_api = Flask(__name__)


# =========================================================
# Load embedding model
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# =========================================================
# Load FAISS vector store
# =========================================================

if not VECTORSTORE_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {VECTORSTORE_PATH}"
    )

vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

print("AuroraStay FAISS index loaded successfully.")
print("Total vectors:", vectorstore.index.ntotal)


# =========================================================
# Helper functions
# =========================================================

def get_request_parameters():
    """Validate and extract query and retrieval count."""

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return None, None, "A JSON request body is required."

    query = data.get("query")

    if not isinstance(query, str) or not query.strip():
        return None, None, "A non-empty query is required."

    k = data.get("k", DEFAULT_K)

    if type(k) is not int or not 1 <= k <= 30:
        return None, None, "k must be an integer between 1 and 30."

    return query.strip(), k, None


def answer_to_text(answer):
    """Convert a LangChain AIMessage to plain text."""

    if hasattr(answer, "content"):
        return str(answer.content)

    return str(answer)


# =========================================================
# Home endpoint
# =========================================================

@rag_api.get("/")
def home():

    return jsonify({
        "application": "AuroraStay Hotels RAG API",
        "status": "running",
        "message": "Welcome to the AuroraStay RAG API"
    })


# =========================================================
# Endpoint 1: Retrieve relevant chunks
# =========================================================

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():

    query, k, error = get_request_parameters()

    if error:
        return jsonify({"error": error}), 400

    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# =========================================================
# Endpoint 2: Answer with relevant chunks
# =========================================================

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():

    query, k, error = get_request_parameters()

    if error:
        return jsonify({"error": error}), 400

    # Existing RAG function signature:
    # rag(query, vectorstore, k=15)

    generated_answer, chunks = rag(
        query=query,
        vectorstore=vectorstore,
        k=k
    )

    generated_answer = answer_to_text(generated_answer)

    return jsonify({
        "query": query,
        "k": k,
        "answer": generated_answer,
        "relevant_chunks": chunks
    })


# =========================================================
# Endpoint 3: Generate answer only
# =========================================================

@rag_api.post("/v1/answer")
def answer():

    query, k, error = get_request_parameters()

    if error:
        return jsonify({"error": error}), 400

    generated_answer, _ = rag(
        query=query,
        vectorstore=vectorstore,
        k=k
    )

    generated_answer = answer_to_text(generated_answer)

    return jsonify({
        "query": query,
        "answer": generated_answer
    })


# =========================================================
# Run Flask application
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    rag_api.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
