import time
from functools import wraps

def timer(func):
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[{func.__qualname__}] Execution time: {execution_time:.4f} sec")
        return result
    return wrapper
