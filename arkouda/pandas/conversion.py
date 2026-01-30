from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from typeguard import typechecked

from arkouda.numpy.dtypes import SeriesDTypes
from arkouda.numpy.pdarrayclass import pdarray


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = ["from_series"]


@typechecked
def from_series(
    series: pd.Series, dtype: Optional[Union[type, str]] = None
) -> Union[pdarray, "Strings"]:
    """
    Convert a pandas ``Series`` to an Arkouda ``pdarray`` or ``Strings``.

    If ``dtype`` is not provided, the dtype is inferred from the pandas
    ``Series`` (using pandas' dtype metadata). If ``dtype`` is provided, it
    is used as an override and normalized via Arkouda's dtype resolution rules.

    In addition to the core numeric/bool types, this function supports
    datetime and timedelta Series of **any** resolution (``ns``, ``us``, ``ms``,
    etc.) by converting them to an ``int64`` pdarray of nanoseconds.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to convert.
    dtype : Optional[Union[type, str]], optional
        Optional dtype override. This may be a Python type (e.g. ``bool``),
        a NumPy scalar type (e.g. ``np.int64``), or a dtype string.

        String-like spellings are normalized to Arkouda string dtype, including:
        ``"object"``, ``"str"``, ``"string"``, ``"string[python]"``,
        and ``"string[pyarrow]"``.

    Returns
    -------
    Union[pdarray, Strings]
        An Arkouda ``pdarray`` for numeric/bool/datetime/timedelta inputs, or an
        Arkouda ``Strings`` for string inputs.

    Raises
    ------
    ValueError
        Raised if the dtype cannot be interpreted or is unsupported for conversion.

    Examples
    --------
    >>> import arkouda as ak
    >>> import numpy as np
    >>> import pandas as pd

    # ints
    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.randint(0, 10, 5)))
    array([4 3 3 5 0])

    >>> ak.from_series(pd.Series(['1', '2', '3', '4', '5']), dtype=np.int64)
    array([1 2 3 4 5])

    # floats
    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.uniform(low=0.0, high=1.0, size=3)))
    array([0.089433234324597599 0.1153776854774361 0.51874393620990389])

    # bools
    >>> np.random.seed(1864)
    >>> ak.from_series(pd.Series(np.random.choice([True, False], size=5)))
    array([True True True False False])

    # strings: pandas dtype spellings normalized to Arkouda Strings
    >>> ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
    array(['a', 'b', 'c', 'd', 'e'])

    >>> ak.from_series(pd.Series(['a', 'b', 'c'], dtype="string[pyarrow]"))
    array(['a', 'b', 'c'])

    # datetime: any resolution is accepted, returned as int64 nanoseconds
    >>> ak.from_series(pd.Series(pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01')])))
    array([1514764800000000000 1514764800000000000])

    Notes
    -----
    - Datetime and timedelta Series are converted to ``int64`` nanoseconds.
    - String-like pandas dtypes (including ``object``) are treated as string and
      converted to Arkouda ``Strings``.
    """
    from arkouda.numpy.pdarraycreation import array

    if not dtype:
        dt = series.dtype.name
    else:
        dt = str(dtype)

    # normalize pandas string dtype spellings
    if dt in {"object", "str", "string[python]", "string[pyarrow]"}:
        dt = "string"

    try:
        """
        If the Series has a object dtype, set dtype to string to comply with method
        signature that does not require a dtype; this is required because Pandas can infer
        non-str dtypes from the input np or Python array.
        """
        # handle datetime/timedelta with any resolution (us, ms, ns, ...)
        if series.dtype.kind in ("M", "m"):
            # normalize to ns then convert to int64 nanoseconds
            if series.dtype.kind == "M":
                n_array = series.to_numpy(dtype="datetime64[ns]").astype("int64")
            else:
                n_array = series.to_numpy(dtype="timedelta64[ns]").astype("int64")
            return array(n_array, dtype=np.int64)

        n_array = series.to_numpy(dtype=SeriesDTypes[dt])
    except KeyError:
        raise ValueError(
            f"dtype {dt} is unsupported. Supported dtypes are bool, float64, int64, string, "
            f"datetime64[ns], and timedelta64[ns]"
        )
    return array(n_array)
