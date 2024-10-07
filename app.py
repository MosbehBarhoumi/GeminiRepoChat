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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        repo_url = request.form['repo_url']
        token = request.form.get('github_token')
        include_extensions = request.form.get('include_extensions')
        exclude_extensions = request.form.get('exclude_extensions')
        content_filter = request.form.get('content_filter')
        try:
            if token:
                content = fetch_repo_contents(repo_url, token, include_extensions, exclude_extensions, content_filter)
            else:
                content = rate_limited(fetch_repo_contents)(repo_url, None, include_extensions, exclude_extensions, content_filter)
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
                temp_file.write(content)
            
            return send_file(temp_file.name, as_attachment=True, download_name='repo_contents.txt')
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', error=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)