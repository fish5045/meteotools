
from inspect import getargspec


def decorate(fn):
    argspec = getargspec(fn)
    second_argname = argspec[0][1]

    def inner(*args, **kwargs):
        special_value = (kwargs[second_argname]
                         if second_argname in kwargs else args[1])
        if special_value == 2:
            print("foo")
        else:
            print("no foo for you")
        return fn(*args, **kwargs)
    return inner


@decorate
def foo(a, b, c=3):
    pass


foo(1, 2)
foo(1, b=2, c=4)
foo(1, 3, 5)
foo(1, b=6, c=5)
