import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once when server starts (not on every request)
model = SentenceTransformer('all-MiniLM-L6-v2')


def split_into_chunks(text, chunk_size=500):
    """
    Split large text into smaller overlapping chunks
    Overlap helps so answers don't get cut off at chunk boundaries
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks


def build_vector_store(text):
    """
    Take plain text → return FAISS index + chunks
    This is called after document upload
    """
    # Step 1 - Split text into chunks
    chunks = split_into_chunks(text)
    
    if not chunks:
        return None, []

    # Step 2 - Convert chunks to embeddings
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    # Step 3 - Build FAISS index
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks


def search_similar_chunks(query, index, chunks, top_k=10):
    """
    Take user question → find most relevant chunks
    This is called on every chat message
    """
    # Convert question to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Search FAISS for top_k similar chunks
    distances, indices = index.search(query_embedding, top_k)

    # Return the actual text chunks
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results