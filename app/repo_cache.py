import os
import hashlib

class RepoCache:
    def __init__(self, cache_dir='./cache'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

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
