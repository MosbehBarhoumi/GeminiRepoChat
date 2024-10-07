import time
from functools import wraps

RATE_LIMIT = 60  # requests per hour
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
            raise Exception("Rate limit exceeded. Please try again later.")
        rate_limiter.add_request()
        return f(*args, **kwargs)
    return decorated_function
