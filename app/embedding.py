from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Initialize the model and FAISS index
model_transformer = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model_transformer.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)

def split_content_into_chunks(content, chunk_size=1000):
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

def embed_and_store_chunks_in_faiss(content):
    chunks = split_content_into_chunks(content)
    chunk_embeddings = model_transformer.encode(chunks, convert_to_tensor=False)
    faiss_index.add(np.array(chunk_embeddings))
    return chunks

def embed_question(question):
    return model_transformer.encode([question], convert_to_tensor=False)

def get_top_k_similar_chunks_faiss(question_embedding, top_k=3):
    distances, indices = faiss_index.search(np.array(question_embedding), top_k)
    return indices

def get_relevant_code_chunks(content, user_prompt):
    chunks = embed_and_store_chunks_in_faiss(content)
    question_embedding = embed_question(user_prompt)
    top_indices = get_top_k_similar_chunks_faiss(question_embedding)
    top_chunks = [chunks[i] for i in top_indices[0]]
    return top_chunks
