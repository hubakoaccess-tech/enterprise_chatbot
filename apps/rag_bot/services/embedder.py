import faiss
import numpy as np
import os
from groq import Groq

client = Groq(api_key=os.getenv('GROQ_API_KEY'))


def get_embedding(text):
    """Get embedding using a simple TF-IDF like approach"""
    # Use sklearn for lightweight embeddings - no torch needed
    return None


def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def build_vector_store(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    chunks = split_into_chunks(text)
    if not chunks:
        return None, [], None

    # Use TF-IDF instead of sentence transformers — very lightweight
    vectorizer = TfidfVectorizer(max_features=384)
    embeddings = vectorizer.fit_transform(chunks).toarray()
    embeddings = normalize(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks, vectorizer


def search_similar_chunks(query, index, chunks, vectorizer, top_k=10):
    from sklearn.preprocessing import normalize

    query_embedding = vectorizer.transform([query]).toarray()
    query_embedding = normalize(query_embedding).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results