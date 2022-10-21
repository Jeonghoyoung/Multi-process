import time


def clock(func):
    def clocked(*args):  # Define inner function clocked to accept any number of positional arguments.
        t0 = time.perf_counter()
        result = func(*args)  # This line only works because the closure for clocked encompasses the func free variable.
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s' % (elapsed, name))
        return result

    return clocked  # Return the inner function to replace the decorated function.