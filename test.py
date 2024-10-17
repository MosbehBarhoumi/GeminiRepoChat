import streamlit as st
import requests
import re
import base64
import os
import hashlib
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from dotenv import load_dotenv
import ast
from pythonchunkextractor import PythonChunkExtractor, split_python_content

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_sentence_transformer():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

@st.cache_resource
def initialize_gemini_model():
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel('gemini-pro')

model_transformer = load_sentence_transformer()
gemini_model = initialize_gemini_model()

# Repository content caching
class RepoCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_key(self, repo_url, include_extensions, exclude_extensions, content_filter):
        key = f"{repo_url}_{include_extensions}_{exclude_extensions}_{content_filter}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_cached_content(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read()
        return None

    def set_cached_content(self, cache_key, content):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_file, 'w') as f:
            f.write(content)



repo_cache = RepoCache()


# Fetch repository contents from GitHub API
def fetch_repo_contents(repo_url, token=None, include_extensions=None, exclude_extensions=None, content_filter=None):
    # Generate cache key based on input parameters
    cache_key = repo_cache.get_cache_key(repo_url, include_extensions, exclude_extensions, content_filter)

    # Check if content is cached
    cached_content = repo_cache.get_cached_content(cache_key)
    if cached_content:
        return cached_content

    # Extract owner and repo from the URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        raise ValueError("Invalid GitHub URL")

    owner, repo = match.groups()
    all_content = []

    # Adjust filters for extensions
    include_extensions = set(ext.strip().lower() for ext in include_extensions.split(',')) if include_extensions else None
    exclude_extensions = set(ext.strip().lower() for ext in exclude_extensions.split(',')) if exclude_extensions else set()

    # Recursive function to process repository contents
    def process_contents(path=''):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {'Authorization': f'token {token}'} if token else {}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch repository contents: {response.json().get('message', 'Unknown error')}")

        contents = response.json()

        # Process each file or directory
        if isinstance(contents, list):
            for item in contents:
                if item['type'] == 'dir':
                    process_contents(item['path'])
                elif item['type'] == 'file':
                    _, ext = os.path.splitext(item['name'])
                    ext = ext.lower()
                    if (include_extensions is None or ext in include_extensions) and ext not in exclude_extensions:
                        file_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{item['path']}"
                        file_response = requests.get(file_url, headers=headers)
                        if file_response.status_code == 200:
                            file_content = base64.b64decode(file_response.json()['content']).decode('utf-8', errors='ignore')
                            if content_filter is None or content_filter.lower() in file_content.lower():
                                all_content.append(f"File: {item['path']}\n\n{file_content}\n\n{'='*80}\n\n")
                        else:
                            print(f"Error fetching file {item['path']}: {file_response.status_code}")
        elif isinstance(contents, dict) and contents['type'] == 'file':
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if (include_extensions is None or ext in include_extensions) and ext not in exclude_extensions:
                file_content = base64.b64decode(contents['content']).decode('utf-8', errors='ignore')
                if content_filter is None or content_filter.lower() in file_content.lower():
                    all_content.append(f"File: {path}\n\n{file_content}\n\n{'='*80}\n\n")

    # Start processing from the root
    process_contents()
    content = ''.join(all_content)
    repo_cache.set_cached_content(cache_key, content)
    return content

# Split text files into chunks based on line length
def split_text_file(file_name, content):
    chunks = []
    lines = content.split('\n')
    chunk = []
    chunk_size = 0
    max_chunk_size = 1000  # Adjust as needed

    for line in lines:
        if chunk_size + len(line) > max_chunk_size and chunk:
            chunks.append(f"File: {file_name}\n\n" + '\n'.join(chunk))
            chunk = []
            chunk_size = 0
        chunk.append(line)
        chunk_size += len(line)

    if chunk:
        chunks.append(f"File: {file_name}\n\n" + '\n'.join(chunk))

    return chunks


# Split repository content into chunks based on file type
def split_content_into_smart_chunks(content):
    chunks = []
    files = content.split('=' * 80)

    for file in files:
        if not file.strip():
            continue

        file_name, file_content = file.split('\n\n', 1)
        file_name = file_name.replace('File: ', '').strip()
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() == '.py':
            chunks.extend(split_python_content(file_name, file_content))
        else:
            # Use existing logic for non-Python files
            chunks.extend(split_text_file(file_name, file_content))

    return chunks

def embed_chunks_and_question(chunks, question):
    if not chunks:
        return None, None
    chunk_embeddings = model_transformer.encode(chunks, convert_to_tensor=True, device=device)
    question_embedding = model_transformer.encode([question], convert_to_tensor=True, device=device)
    return chunk_embeddings, question_embedding

# Retrieve top-k similar chunks to the question based on cosine similarity
def get_top_k_similar_chunks(chunk_embeddings, question_embedding, chunks, top_k=3):
    if chunk_embeddings is None or question_embedding is None:
        return []

    # Ensure both tensors are on the same device
    chunk_embeddings = chunk_embeddings.to(device)
    question_embedding = question_embedding.to(device)

    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    top_chunks = [chunks[idx] for idx in top_k_indices.cpu().numpy()]
    return top_chunks


# Generate a refined version of the user's question using the Gemini AI model
def get_refined_prompt(model, initial_prompt):
    refinement_prompt = f"""Given the user's initial question: "{initial_prompt}"
    Please provide a refined version of this question that includes:
    1. A more detailed description of what to look for
    2. Potential keywords or function names that might be relevant
    3. A general pseudo-code structure of what the desired code might look like

    Format your response as follows:
    Refined Question: [Your refined question here]
    Keywords: [Comma-separated list of potential keywords]
    Pseudo-code:
    ```
    [Your pseudo-code here]
    ```
    """
    response = model.generate_content(refinement_prompt)
    return response.text

# Parse the refined prompt to extract the refined question, keywords, and pseudo-code
def parse_refined_prompt(refined_prompt):
    lines = refined_prompt.split('\n')
    refined_question = ''
    keywords = []
    pseudo_code = []
    current_section = ''

    for line in lines:
        if line.startswith('Refined Question:'):
            current_section = 'question'
            refined_question = line.replace('Refined Question:', '').strip()
        elif line.startswith('Keywords:'):
            current_section = 'keywords'
            keywords = [k.strip() for k in line.replace('Keywords:', '').split(',')]
        elif line.startswith('```'):
            if current_section != 'pseudo_code':
                current_section = 'pseudo_code'
            else:
                current_section = ''
        elif current_section == 'pseudo_code':
            pseudo_code.append(line)

    return refined_question, keywords, '\n'.join(pseudo_code)

def get_relevant_code_chunks(content, user_question, refined_question, keywords):
    chunks = split_content_into_smart_chunks(content)
    if not chunks:
        return []

    # Combine refined question and keywords for embedding
    search_query = f"{refined_question} {' '.join(keywords)}"
    chunk_embeddings, question_embedding = embed_chunks_and_question(chunks, search_query)
    top_chunks = get_top_k_similar_chunks(chunk_embeddings, question_embedding, chunks)
    return top_chunks

# Modified function
def get_chat_response(model, question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""
    You are an AI assistant that answers questions based ONLY on the provided content from a GitHub repository. 
    Your task is to answer the following question using ONLY the information in the given context. 
    If the answer cannot be found in the provided content, explicitly state that the information is not available in the given repository content.

    Context from the GitHub repository:
    {context}

    Question: {question}

    Please follow these rules:
    1. Only use information present in the provided context to answer the question.
    2. If the answer is not in the context, say "The information about [topic] is not available in the given repository content."
    3. Do not make up or infer information that is not explicitly stated in the context.
    4. If you use any information from the context, reference the file name it came from.

    Your response:
    """
    response = model.generate_content(prompt)
    return response.text

# New function
def verify_response_relevance(response, relevant_chunks):
    # Check if the response contains any significant content from the relevant chunks
    chunk_content = " ".join(relevant_chunks).lower()
    response_lower = response.lower()
    
    # List of phrases indicating the information is not in the repository
    not_found_phrases = [
        "is not available in the given repository content",
        "cannot be found in the provided content",
        "is not present in the repository",
        "no information about this in the repository"
    ]
    
    # Check if any of the not_found_phrases are in the response
    if any(phrase in response_lower for phrase in not_found_phrases):
        return True  # The response correctly indicates the information is not in the repo
    
    # Check if the response contains any significant content from the chunks
    significant_words = set(word.lower() for word in response.split() if len(word) > 4)
    matching_words = significant_words.intersection(set(chunk_content.split()))
    
    return len(matching_words) > 0  # Return True if there's significant overlap

def main():
    st.title("GitHub Repository Chat with Gemini AI")

    if "repo_content" not in st.session_state:
        st.session_state.repo_content = None

    st.sidebar.header("Repository Information")
    repo_url = st.sidebar.text_input("GitHub Repository URL")
    include_extensions = st.sidebar.text_input("Include Extensions", "py,md,txt")
    exclude_extensions = st.sidebar.text_input("Exclude Extensions", "gitignore")
    content_filter = st.sidebar.text_input("Content Filter (optional)")

    if st.sidebar.button("Fetch Repository"):
        if repo_url:
            with st.spinner("Fetching repository contents..."):
                try:
                    st.session_state.repo_content = fetch_repo_contents(repo_url, GITHUB_TOKEN, include_extensions, exclude_extensions, content_filter)
                    st.success("Repository contents fetched successfully!")
                except Exception as e:
                    st.error(f"Error fetching repository: {str(e)}")
        else:
            st.warning("Please enter a GitHub repository URL.")

    st.header("Chat with the Repository")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the repository"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.repo_content:
            with st.spinner("Analyzing repository and generating response..."):
                refined_prompt = get_refined_prompt(gemini_model, question)
                refined_question, keywords, pseudo_code = parse_refined_prompt(refined_prompt)

                st.subheader("Question Refinement")
                st.write(f"Refined Question: {refined_question}")
                st.write(f"Keywords: {', '.join(keywords)}")
                if pseudo_code:
                    st.code(pseudo_code, language="python")

                relevant_chunks = get_relevant_code_chunks(st.session_state.repo_content, question, refined_question, keywords)

                st.subheader("Relevant Code Chunks")
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.text_area(f"Chunk {i}", chunk, height=150)

                chat_response = get_chat_response(gemini_model, refined_question, relevant_chunks)
                
                st.subheader("Raw AI Response")
                st.text_area("Raw Response", chat_response, height=200)

                is_relevant = verify_response_relevance(chat_response, relevant_chunks)
                st.write(f"Response deemed relevant: {is_relevant}")

                if is_relevant:
                    st.session_state.messages.append({"role": "assistant", "content": chat_response})
                    st.chat_message("assistant").markdown(chat_response)
                else:
                    warning_message = "I apologize, but I couldn't find relevant information in the repository to answer your question. Could you please rephrase your question or ask about something specific to the code in this repository?"
                    st.session_state.messages.append({"role": "assistant", "content": warning_message})
                    st.chat_message("assistant").warning(warning_message)
        else:
            st.warning("Please fetch a repository first before asking questions.")

    st.sidebar.write("Chatbot powered by Gemini AI")

if __name__ == "__main__":
    main()