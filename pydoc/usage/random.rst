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

integers
---------
.. autofunction:: arkouda.random.Generator.integers

random
---------
.. autofunction:: arkouda.random.Generator.random

standard_normal
---------
.. autofunction:: arkouda.random.Generator.standard_normal

uniform
---------
.. autofunction:: arkouda.random.Generator.uniform
