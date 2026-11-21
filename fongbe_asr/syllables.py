from contextlib import contextmanager
from pathlib import Path
import runpy
import sys


@contextmanager
def syllabifier(official_path=None, synthetic=False):
    if synthetic:
        from .synthetic import synthetic_syllabify

        yield synthetic_syllabify
    elif official_path is not None:
        path = Path(official_path).resolve(strict=True)
        sys.path.insert(0, str(path.parent))
        try:
            fn = runpy.run_path(str(path)).get("syllabify")
            if not callable(fn):
                raise ValueError(f"{path} must define syllabify(text)")
            yield fn
        finally:
            sys.path.pop(0)
    else:
        yield lambda text: text
