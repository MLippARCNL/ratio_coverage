from typing import ParamSpec, TypeVar, Callable


def disablePrint():
    import os, sys
    old_out = sys.stdout
    old_err = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    return old_out, old_err


def enablePrint(old_out=None, old_err=None):
    import sys
    if old_out is None:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = old_out
    if old_err is None:
        sys.stderr = sys.__stderr__
    else:
        sys.stderr = old_err


class DummyFile(object):
    file = None
    data = ""

    def __init__(self, file, start):
        self.file = file
        self.start = start

    def write(self, x):
        from tqdm import tqdm

        x = x.split('\n')

        for i in range(len(x) - 1):
            tqdm.write(self.data + x[i], file=self.file)
            self.data = ""
        self.data += x[len(x) - 1]
        if x[len(x) - 1] == "\n":
            tqdm.write(self.data, file=self.file)
            self.data = ""


class redirect:
    old_out = None
    old_err = None

    def __init__(self):
        pass

    def __enter__(self):
        import sys
        self.old_out = sys.stdout
        self.old_err = sys.stderr
        sys.stdout = DummyFile(self.old_err, "?")
        sys.stderr = DummyFile(self.old_err, "!")

    def __exit__(self, type, value, traceback):
        import sys
        sys.stdout = self.old_out
        sys.stderr = self.old_err


class Timer:
    def __init__(self, name, quiet=False):
        self.time = None
        self.name = name
        self.quite = quiet
        self.old = None

    def __enter__(self):
        import time
        self.time = time.time()
        if self.quite:
            self.old = disablePrint()
        return self.time

    def __exit__(self, type, value, traceback):
        import time
        if self.quite and self.old is not None:
            enablePrint(*self.old)
        print("--- %s %s seconds --- " % (self.name, time.time() - self.time))


def pprint(*args, linestart="# ", **kwargs):
    my_msg = linestart if linestart is not None else ""
    for a in args:
        my_msg += f"{a:<30}"
    for k, v in kwargs.items():
        my_msg += f"{k:>30}={v:<30}"
    print(my_msg)


_P = ParamSpec("_P")
_T = TypeVar("_T")


def inherit_signature_from(_to: Callable[_P, _T]) -> Callable[[Callable[..., _T]], Callable[_P, _T]]:
    """Set the signature checked by pyright/vscode to the signature of another function."""
    return lambda x: x  # type: ignore
