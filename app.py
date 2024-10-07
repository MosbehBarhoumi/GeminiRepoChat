from flask import Flask, render_template, request, send_file
import requests
from io import StringIO
import tempfile
import re
import base64
import time
from functools import wraps
import os
import json
import hashlib
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Rate limiting
RATE_LIMIT = 60  # requests per hour for unauthenticated users
RATE_LIMIT_PERIOD = 3600  # 1 hour in seconds

class RateLimiter:
    def __init__(self):
        self.requests = []

    def add_request(self):
        current_time = time.time()
        self.requests = [t for t in self.requests if current_time - t < RATE_LIMIT_PERIOD]
        self.requests.append(current_time)

    def can_make_request(self):
        return len(self.requests) < RATE_LIMIT

rate_limiter = RateLimiter()

def rate_limited(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not rate_limiter.can_make_request():
            raise Exception("Rate limit exceeded. Please try again later or use authentication.")
        rate_limiter.add_request()
        return f(*args, **kwargs)
    return decorated_function

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

# Load a pre-trained sentence transformer model
model_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_repo_contents(repo_url, token=None, include_extensions=None, exclude_extensions=None, content_filter=None):
    cache_key = repo_cache.get_cache_key(repo_url, include_extensions, exclude_extensions, content_filter)
    cached_content = repo_cache.get_cached_content(cache_key)
    if cached_content:
        return cached_content

    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        raise ValueError("Invalid GitHub URL")
    
    owner, repo = match.groups()
    all_content = StringIO()

    include_extensions = set(ext.strip().lower() for ext in include_extensions.split(',')) if include_extensions else None
    exclude_extensions = set(ext.strip().lower() for ext in exclude_extensions.split(',')) if exclude_extensions else set()

    def process_contents(path=''):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {'Authorization': f'token {token}'} if token else {}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repository contents: {response.json().get('message', 'Unknown error')}")

        contents = response.json()

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
                                all_content.write(f"File: {item['path']}\n\n")
                                all_content.write(file_content)
                                all_content.write("\n\n" + "="*80 + "\n\n")
        elif isinstance(contents, dict) and contents['type'] == 'file':
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if (include_extensions is None or ext in include_extensions) and ext not in exclude_extensions:
                file_content = base64.b64decode(contents['content']).decode('utf-8', errors='ignore')
                if content_filter is None or content_filter.lower() in file_content.lower():
                    all_content.write(f"File: {path}\n\n")
                    all_content.write(file_content)
                    all_content.write("\n\n" + "="*80 + "\n\n")

    process_contents()
    content = all_content.getvalue()
    repo_cache.set_cached_content(cache_key, content)
    return content

# Step 1: Split the content into smaller chunks
def split_content_into_chunks(content, chunk_size=1000):
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Step 2: Embed the chunks and the user's question
def embed_chunks_and_question(chunks, question):
    chunk_embeddings = model_transformer.encode(chunks, convert_to_tensor=True)
    question_embedding = model_transformer.encode(question, convert_to_tensor=True)
    return chunk_embeddings, question_embedding

# Step 3: Get the top-k most similar chunks
def get_top_k_similar_chunks(chunk_embeddings, question_embedding, chunks, top_k=3):
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_k_indices = similarities.topk(k=top_k).indices
    top_chunks = [chunks[idx] for idx in top_k_indices]
    return top_chunks

# Function to get relevant chunks based on user’s prompt
def get_relevant_code_chunks(content, user_question):
    chunks = split_content_into_chunks(content)
    chunk_embeddings, question_embedding = embed_chunks_and_question(chunks, user_question)
    top_chunks = get_top_k_similar_chunks(chunk_embeddings, question_embedding, chunks)
    return top_chunks

# Gemini API configuration
def initialize_model(api_key, model_name="gemini-pro"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def get_chat_response(model, prompt):
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'repo_url' in request.form:
            # GitHub repo fetching logic
            repo_url = request.form['repo_url']
            token = request.form.get('github_token')
            include_extensions = request.form.get('include_extensions')
            exclude_extensions = request.form.get('exclude_extensions')
            content_filter = request.form.get('content_filter')
            user_prompt = request.form.get('gemini_prompt')
            api_key = request.form.get('gemini_api_key')
            try:
                # Fetch repository contents
                if token:
                    content = fetch_repo_contents(repo_url, token, include_extensions, exclude_extensions, content_filter)
                else:
                    content = rate_limited(fetch_repo_contents)(repo_url, None, include_extensions, exclude_extensions, content_filter)

                # Get the relevant code chunks
                relevant_chunks = get_relevant_code_chunks(content, user_prompt)

                # Prepare the final prompt
                final_prompt = f"Here are the most relevant parts of the code:\n\n{''.join(relevant_chunks)}\n\nUser's question: {user_prompt}"

                # Send to Gemini API
                model = initialize_model(api_key)
                gemini_response = get_chat_response(model, final_prompt)

                # Pass the relevant chunks to the template
                return render_template('index.html', gemini_response=gemini_response, relevant_chunks=relevant_chunks)
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                return render_template('index.html', error=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
