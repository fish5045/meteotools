from time import time


def timer(func):
    def func_wrapper(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f'{func.__name__} cost time: {(t1-t0):.5f} (s).')
        return result
    return func_wrapper
