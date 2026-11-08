"""Retrieval for the rag dataset: a document corpus in ChromaDB, searched through a tool.

The corpus describes a fictional company, so a model can't answer from memory: it has to
search, and whether the search found the right documents can be checked against each
case's relevant_docs. Separating "didn't find it" from "found it and still answered wrong"
is the point of the dataset.

chromadb is an optional dependency (pip install -e ".[rag]"), imported only when an index
is built, so the other datasets and the API don't pay for it.
"""
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agenteval.tools import DEFAULT_TOOLS, Tool

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "rag_corpus.jsonl"

# "minilm" is ChromaDB's default model (all-MiniLM-L6-v2, downloaded on first use).
# "hashing" matches words, not meaning: it needs no download, so tests use it, and it is a
# lexical baseline to compare a semantic model against.
EMBEDDINGS = ("minilm", "hashing")
DEFAULT_EMBEDDING = "minilm"
DEFAULT_TOP_K = 4
MAX_TOP_K = 8

# Whole results are recorded, since scorers read the returned document ids and text from
# the trace. MAX_TOP_K of the corpus's longest documents must fit; a test checks they do.
MAX_RECORDED_SEARCH = 8000

class Document(BaseModel):
    id: str
    title: str
    date: str  # ISO date, so the agent can tell a superseded policy from the current one
    text: str

def load_corpus(path: Path = CORPUS_PATH) -> list[Document]:
    return [
        Document.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def corpus_version(documents: list[Document]) -> str:
    """Content hash of the corpus. The cases don't change when a document does, so this goes
    in the agent's config: editing the corpus then shows up as a change between runs."""
    content = json.dumps([document.model_dump() for document in documents], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:12]

HASHING_DIMENSIONS = 512

def hash_embed(text: str, dimensions: int = HASHING_DIMENSIONS) -> list[float]:
    """Bag of words hashed into a fixed-size unit vector. sha256 rather than hash(), whose
    value changes between Python processes, so results are the same on every run."""
    vector = [0.0] * dimensions
    for word in re.findall(r"\w+", text.casefold()):
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dimensions] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]

def _embedding_function(name: str) -> Any:
    from chromadb import EmbeddingFunction
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    if name == "minilm":
        return DefaultEmbeddingFunction()
    if name != "hashing":
        raise ValueError(f"unknown embedding {name!r}; use one of {EMBEDDINGS}")

    # Defined here so that importing this module doesn't import chromadb.
    class HashingEmbedding(EmbeddingFunction):
        def __init__(self) -> None:
            pass

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [hash_embed(text) for text in input]

        @staticmethod
        def name() -> str:
            return "hashing"

        def get_config(self) -> dict[str, Any]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> "HashingEmbedding":
            return HashingEmbedding()

    return HashingEmbedding()

class DocumentIndex:
    """One document per entry: the documents are short, so there is no chunking to tune."""

    def __init__(self, documents: list[Document], embedding: str = DEFAULT_EMBEDDING):
        import chromadb

        self.documents = {document.id: document for document in documents}
        # In-memory clients in one process share their collections, so each index gets its own.
        self.collection = chromadb.EphemeralClient().create_collection(
            f"corpus-{uuid.uuid4().hex}",
            embedding_function=_embedding_function(embedding),
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=list(self.documents),
            documents=[f"{document.title}\n{document.text}" for document in documents],
        )

    def search(self, query: str, k: int) -> list[dict[str, str]]:
        found = self.collection.query(query_texts=[query], n_results=min(k, len(self.documents)))
        return [
            {"doc_id": doc_id, "title": document.title, "date": document.date, "text": document.text}
            for doc_id in found["ids"][0]
            for document in [self.documents[doc_id]]
        ]

class SearchArguments(BaseModel):
    query: str

def retrieval_config(top_k: int, embedding: str, documents: list[Document] | None = None) -> dict[str, Any]:
    """The retrieval settings, for AgentConfig.params: changing any of them is a new version."""
    return {
        "embedding": embedding,
        "top_k": top_k,
        "corpus_version": corpus_version(documents if documents is not None else load_corpus()),
    }

def rag_tools(
    top_k: int = DEFAULT_TOP_K,
    embedding: str = DEFAULT_EMBEDDING,
    documents: list[Document] | None = None,
) -> list[Tool]:
    # Checked here rather than left to the model: the number of results is a setting of the
    # experiment, so two runs that differ only in top_k can be compared.
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
    index = DocumentIndex(documents if documents is not None else load_corpus(), embedding)

    return [
        Tool(
            name="search_docs",
            description=(
                "Search Tessaly Instruments' internal documents: policies, product specifications, "
                f"pricing and release notes. Returns the {top_k} best-matching documents, each with "
                "its id, title, date and text."
            ),
            arguments=SearchArguments,
            function=lambda query: index.search(query, top_k),
            max_recorded_chars=MAX_RECORDED_SEARCH,
        ),
        DEFAULT_TOOLS[0],  # calculator, for questions that combine numbers from several documents
    ]
