import json
import pickle
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
KB_PATH = 'apps/helpdesk_bot/knowledge_base.json'
CACHE_PATH = 'apps/helpdesk_bot/embeddings_cache.pkl'


def load_knowledge_base():
    """Load all Q&A pairs from JSON file"""
    with open(KB_PATH, 'r') as f:
        data = json.load(f)

    questions = []
    answers = []

    for category in data['knowledge_base']:
        for entry in category['entries']:
            questions.append(entry['question'])
            answers.append(entry['answer'])

    return questions, answers


def build_or_load_cache():
    """
    Build FAISS index from knowledge base
    Cache it so we don't recompute on every request
    """
    questions, answers = load_knowledge_base()

    # Check if cache exists and is still valid
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        # If questions haven't changed, use cache
        if cache['questions'] == questions:
            return cache['index'], questions, answers

    # Build new embeddings
    embeddings = model.encode(questions)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save cache
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({
            'index': index,
            'questions': questions
        }, f)

    return index, questions, answers


def search_knowledge_base(query, top_k=5):
    """
    Find top 5 most relevant Q&A pairs for a user query
    """
    index, questions, answers = build_or_load_cache()

    # Convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Build results
    results = []
    for idx in indices[0]:
        if idx < len(questions):
            results.append({
                'question': questions[idx],
                'answer': answers[idx]
            })

    return results