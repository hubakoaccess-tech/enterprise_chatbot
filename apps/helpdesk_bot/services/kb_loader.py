import json
import os
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, 'knowledge_base.json')

# Global vectorizer and index — built once at first request
_index = None
_questions = None
_answers = None
_vectorizer = None


def load_knowledge_base():
    with open(KB_PATH, 'r') as f:
        data = json.load(f)

    questions = []
    answers = []

    for category in data['knowledge_base']:
        for entry in category['entries']:
            questions.append(entry['question'])
            answers.append(entry['answer'])

    return questions, answers


def build_index():
    global _index, _questions, _answers, _vectorizer

    questions, answers = load_knowledge_base()

    vectorizer = TfidfVectorizer(max_features=384)
    embeddings = vectorizer.fit_transform(questions).toarray()
    embeddings = normalize(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    _index = index
    _questions = questions
    _answers = answers
    _vectorizer = vectorizer


def search_knowledge_base(query, top_k=5):
    global _index, _questions, _answers, _vectorizer

    # Build index on first request
    if _index is None:
        build_index()

    query_embedding = _vectorizer.transform([query]).toarray()
    query_embedding = normalize(query_embedding).astype('float32')

    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(_questions):
            results.append({
                'question': _questions[idx],
                'answer': _answers[idx]
            })

    return results