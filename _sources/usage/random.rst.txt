****************
Random in Arkouda
****************
Pseudo random number generation in Arkouda is modeled after numpy.
Just like in numpy the preferred way to access the random functionality in arkouda is via ``Generator`` objects.
If a ``Generator`` is initialized with a seed, the stream of random numbers it produces can be reproduced
by a new ``Generator`` with the same seed. This reproducibility is not guaranteed across releases.

.. autoclass:: arkouda.random.Generator
   :no-index:

Creation
=========
To create a ``Generator`` use the ``default_rng`` constructor

.. autofunction:: arkouda.random.default_rng
   :no-index:

Features
==========

choice
---------
.. autofunction:: arkouda.random.Generator.choice
   :no-index:

exponential
-----------
.. autofunction:: arkouda.random.Generator.exponential
   :no-index:

integers
--------
.. autofunction:: arkouda.random.Generator.integers
   :no-index:

logistic
--------
.. autofunction:: arkouda.random.Generator.logistic
   :no-index:

lognormal
---------
.. autofunction:: arkouda.random.Generator.lognormal
   :no-index:

normal
------
.. autofunction:: arkouda.random.Generator.normal
   :no-index:

permutation
-----------
.. autofunction:: arkouda.random.Generator.permutation
   :no-index:

poisson
-------
.. autofunction:: arkouda.random.Generator.poisson
   :no-index:

shuffle
-------
.. autofunction:: arkouda.random.Generator.shuffle
   :no-index:

random
------
.. autofunction:: arkouda.random.Generator.random
   :no-index:

standard_exponential
--------------------
.. autofunction:: arkouda.random.Generator.standard_exponential
   :no-index:

standard_normal
---------------
.. autofunction:: arkouda.random.Generator.standard_normal
   :no-index:

uniform
-------
.. autofunction:: arkouda.random.Generator.uniform
   :no-index:
