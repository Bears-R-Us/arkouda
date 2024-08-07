***********
Random in Arkouda
***********
Pseudo random number generation in Arkouda is modeled after numpy.
Just like in numpy the preferred way to access the random functionality in arkouda is via ``Generator`` objects.
If a ``Generator`` is initialized with a seed, the stream of random numbers it produces can be reproduced
by a new ``Generator`` with the same seed. This reproducibility is not guaranteed across releases.

.. autoclass:: arkouda.random.Generator

Creation
=========
To create a ``Generator`` use the ``default_rng`` constructor

.. autofunction:: arkouda.random.default_rng

Features
==========

choice
---------
.. autofunction:: arkouda.random.Generator.choice

exponential
---------
.. autofunction:: arkouda.random.Generator.exponential

integers
---------
.. autofunction:: arkouda.random.Generator.integers

logistic
---------
.. autofunction:: arkouda.random.Generator.logistic

lognormal
---------
.. autofunction:: arkouda.random.Generator.lognormal

normal
---------
.. autofunction:: arkouda.random.Generator.normal

permutation
---------
.. autofunction:: arkouda.random.Generator.permutation

poisson
---------
.. autofunction:: arkouda.random.Generator.poisson

shuffle
---------
.. autofunction:: arkouda.random.Generator.shuffle

random
---------
.. autofunction:: arkouda.random.Generator.random

standard_exponential
---------
.. autofunction:: arkouda.random.Generator.standard_exponential

standard_normal
---------
.. autofunction:: arkouda.random.Generator.standard_normal

uniform
---------
.. autofunction:: arkouda.random.Generator.uniform
