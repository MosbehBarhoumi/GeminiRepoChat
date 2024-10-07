import os
import requests
import base64
from io import StringIO
from .repo_cache import repo_cache
import re


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
