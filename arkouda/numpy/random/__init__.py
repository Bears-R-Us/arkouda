from .generator import Generator, default_rng  # noqa
from .legacy import rand, randint, standard_normal, uniform

__all__ = [
    "Generator",
    "rand",
    "randint",
    "standard_normal",
    "uniform",
]
