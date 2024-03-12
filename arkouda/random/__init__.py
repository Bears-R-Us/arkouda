from ._generator import Generator, default_rng  # noqa
from ._legacy import randint, standard_normal, uniform

__all__ = [
    'Generator',
    'randint',
    'standard_normal',
    'uniform',
]
