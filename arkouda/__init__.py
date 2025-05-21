# flake8: noqa
# isort: skip_file
# do not run isort, imports are order dependent
"""
Arkouda: Exploratory data science at scale.

Arkouda is a Python API for exploratory data analysis on massive datasets. It
leverages a Chapel-based backend to enable high-performance computing on
distributed systems, while exposing a familiar NumPy- and Pandas-like interface
to Python users.

Key Features
------------
- `pdarray` and `Strings` types for working with large numeric and string arrays.
- `Categorical`, `Series`, `DataFrame`, and `Index` for labeled data analysis.
- High-performance `GroupBy`, reductions, and broadcasting operations.
- Interoperability with NumPy and Pandas for ease of use.
- A scalable architecture suitable for HPC and cloud environments.

Example:
-------
>>> import arkouda as ak
>>> ak.connect()
>>> a = ak.array([1, 2, 3])
>>> b = a + 5
>>> print(b)
array([6 7 8])

For full documentation, visit: https://bears-r-us.github.io/arkouda/

"""

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from arkouda.numpy import *
from arkouda.pandas import *
from arkouda.client import *
from arkouda.client_dtypes import *
from arkouda.groupbyclass import *
from arkouda.categorical import *
from arkouda.logger import *
from arkouda.infoclass import *
from arkouda.dataframe import *
from arkouda.index import *
from arkouda.alignment import *
from arkouda.plotting import *
from arkouda.accessor import *
from arkouda.io import *
from arkouda.scipy.special import *
from arkouda.scipy import *
from arkouda.testing import *
from arkouda.apply import *
