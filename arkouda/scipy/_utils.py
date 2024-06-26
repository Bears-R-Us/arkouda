from numpy import ndarray

from arkouda import pdarray


def clean_doc_string(doc_string: str):
    return (
        doc_string.replace("from scipy.stats import", "from arkouda.scipy.stats import")
        .replace("import numpy as", "import arkouda.numpy as")
        .replace("np", "aknp")
    )


def not_implemented(*args, **kwargs):
    """
    This function is not yet implemented.
    """
    raise NotImplementedError


def arrays_not_implemented_decorator(func):
    def inner1(*args, **kwargs):
        for arg in args:
            print(arg)
            if isinstance(arg, (ndarray, pdarray)):
                raise NotImplementedError("This function is not yet implemented for arrays.")

        return func(*args, **kwargs)

    return inner1
