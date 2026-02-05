# Standard libraries
import json
import re
import os
import faiss
import numpy as np
import requests
import dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
FAISS_INDEX_PATH = "outputs/vector_index.faiss"
METADATA_PATH = "outputs/metadata.json"
KG_NODE_FILE = "outputs/kg_nodes.json"
KG_EMB_FILE = "outputs/kg_node_embeddings.npy"
NEO4J_URI = os.getenv("NEO4J_UR")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_URL = os.getenv("OLLAMA_UR")
OLLAMA_MODEL = "llama3.2"

# ---------------- LOAD MODELS ----------------
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- FAISS SEARCH ----------------
def vector_search(query, top_k=5, show=True):
    """
    Perform semantic search using FAISS.
    Converts query into embedding and retrieves top-k similar chunks.
    """
    # Encode query into vector
    q_emb = embedder.encode([query])

    # Search FAISS index
    distances, indices = index.search(np.array(q_emb), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        item = metadata[idx]
        item["_rank"] = rank + 1
        item["_distance"] = float(distances[0][rank])
        results.append(item)

    # Print results if required
    if show:
        print("\n SEMANTIC SEARCH RESULTS")
        for r in results:
            print(f"Rank {r['_rank']} | {r['source_name']} | {r['_distance']:.4f}")

    return results

# ---------------- NEO4J ----------------
# Create Neo4j driver
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def load_kg_nodes_from_db():
    """
    Load all entity names from Neo4j knowledge graph.
    """
    cypher = "MATCH (n:Entity) RETURN DISTINCT toLower(n.name) AS name"
    with driver.session(database="neo4j") as session:
        return [r["name"] for r in session.run(cypher)]

# Load cached KG nodes and embeddings if available
if os.path.exists(KG_NODE_FILE) and os.path.exists(KG_EMB_FILE):
    with open(KG_NODE_FILE, "r") as f:
        kg_node_names = json.load(f)
    kg_node_embeddings = np.load(KG_EMB_FILE)
else:
    # Fetch from Neo4j and generate embeddings
    kg_node_names = load_kg_nodes_from_db()
    kg_node_embeddings = embedder.encode(kg_node_names, batch_size=256)

    # Cache results
    json.dump(kg_node_names, open(KG_NODE_FILE, "w"))
    np.save(KG_EMB_FILE, kg_node_embeddings)

# ---------------- NLP ----------------
# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Stopwords for filtering irrelevant words
STOP_WORDS = set(stopwords.words("english"))

def extract_entities(text):
    """
    Extract potential entities from text using simple token filtering.
    """
    tokens = word_tokenize(text.lower())
    return list(set(
        t for t in tokens if t.isalnum() and t not in STOP_WORDS and len(t) > 2
    ))

def match_entities_to_kg_nodes(entities, threshold=0.65):
    """
    Match extracted entities to KG nodes using embedding similarity.
    """
    matched_nodes = set()

    for ent in entities:
        ent_emb = embedder.encode([ent])
        sims = cosine_similarity(ent_emb, kg_node_embeddings)[0]
        for name, score in zip(kg_node_names, sims):
            if score >= threshold:
                matched_nodes.add(name)
    return list(matched_nodes)

def kg_search(node_names):
    """
    Retrieve related facts from Neo4j for matched entities.
    """
    if not node_names:
        return []
    cypher = """
    MATCH (s:Entity)-[r]->(o)
    WHERE toLower(s.name) IN $names
    RETURN s.name AS source, type(r) AS relation, o.name AS target
    LIMIT 25
    """
    with driver.session(database="neo4j") as session:
        return [r.data() for r in session.run(cypher, names=node_names)]

# ---------------- LLM ----------------
def ollama_generate(prompt):
    """
    Send prompt to Ollama LLM and return generated response.
    """
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120
    )
    return response.json()["response"]

def hybrid_rag_answer(question):
    """
    Hybrid RAG pipeline:
    - Vector search (documents)
    - KG-based reasoning (Neo4j)
    - LLM generation
    """
    retrieved_chunks = vector_search(question)
    SEMANTIC_THRESHOLD = 0.8

    # Filter semantically relevant chunks
    semantic_relevant_chunks = [c for c in retrieved_chunks if c["_distance"] < SEMANTIC_THRESHOLD]
    semantic_relevant = len(semantic_relevant_chunks) > 0
    chunk_texts = [c["chunk_text"] for c in semantic_relevant_chunks]

    # Map source names to file paths
    source_map = {}
    for c in semantic_relevant_chunks:
        name = c.get("file_name", "Unknown")
        path = c.get("path", "")
        if name not in source_map:
            source_map[name] = path

    source_details = [{"name": n, "file": f} for n, f in source_map.items()]
    print(source_details)

    # KG entity extraction and search
    entities = extract_entities(question)
    matched_nodes = match_entities_to_kg_nodes(entities)
    kg_results = kg_search(matched_nodes)

    kg_relevant = len(kg_results) > 0
    has_sources = semantic_relevant or kg_relevant

    # Final prompt to LLM
    prompt = f"""
You are an enterprise AI assistant.

DOCUMENT CONTEXT:
{chunk_texts if semantic_relevant else "No relevant documents found."}

KNOWLEDGE GRAPH:
{kg_results if kg_relevant else "No relevant graph facts found."}

QUESTION:
{question}

Answer concisely.
"""
    answer = ollama_generate(prompt)

    return {
        "answer": answer,
        "source_details": source_details,
        "source_names": list(source_map.keys()) if has_sources else [],
        "has_sources": has_sources,
        "retrieved_chunks": retrieved_chunks,
        "kg_results": kg_results
    }

# ---------------- EVALUATION METRICS ----------------
def retrieval_relevance(question, retrieved_chunks):
    """
    Measures how relevant retrieved chunks are to the question.
    """
    q_emb = embedder.encode([question])
    scores = []

    for c in retrieved_chunks:
        c_emb = embedder.encode([c["chunk_text"]])
        sim = cosine_similarity(q_emb, c_emb)[0][0]
        scores.append(sim)

    return {
        "avg_similarity": float(np.mean(scores)),
        "min_similarity": float(np.min(scores)),
        "max_similarity": float(np.max(scores))
    }

def answer_grounding(answer, retrieved_chunks):
    """
    Measures how well the generated answer is grounded in retrieved documents.
    """
    a_emb = embedder.encode([answer])
    sims = []

    for c in retrieved_chunks:
        c_emb = embedder.encode([c["chunk_text"]])
        sims.append(cosine_similarity(a_emb, c_emb)[0][0])
    return {
        "avg_grounding": float(np.mean(sims)),
        "min_grounding": float(np.min(sims)),
        "max_grounding": float(np.max(sims))
    }

def evaluate_answer(question, answer, retrieved_chunks):
    """
    Combined evaluation of retrieval quality and answer grounding.
    """
    return {
        "retrieval_relevance": retrieval_relevance(question, retrieved_chunks),
        "answer_grounding": answer_grounding(answer, retrieved_chunks),
    }