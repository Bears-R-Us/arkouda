from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar, Union

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
    Converts a Pandas Series to an Arkouda pdarray or Strings object. If
    dtype is None, the dtype is inferred from the Pandas Series. Otherwise,
    the dtype parameter is set if the dtype of the Pandas Series is to be
    overridden or is  unknown (for example, in situations where the Series
    dtype is object).

    Parameters
    ----------
    series : pd.Series
        The Pandas Series with a dtype of bool, float64, int64, or string
    dtype : Optional[Union[type, str]]
        The valid dtype types are np.bool, np.float64, np.int64, and np.str

    Returns
    -------
    Union[pdarray, Strings]

    Raises
    ------
    ValueError
        Raised if the Series dtype is not bool, float64, int64, string, datetime, or timedelta

    Examples
    --------
    >>> import arkouda as ak
    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.randint(0,10,5)))
    array([4 3 3 5 0])

    >>> ak.from_series(pd.Series(['1', '2', '3', '4', '5']),dtype=np.int64)
    array([1 2 3 4 5])

    >>> np.random.seed(1701)
    >>> ak.from_series(pd.Series(np.random.uniform(low=0.0,high=1.0,size=3)))
    array([0.089433234324597599 0.1153776854774361 0.51874393620990389])

    >>> ak.from_series(
    ...     pd.Series([
    ...         '0.57600036956445599',
    ...         '0.41619265571741659',
    ...         '0.6615356693784662',
    ...     ]),
    ...     dtype=np.float64,
    ... )
    array([0.57600036956445599 0.41619265571741659 0.6615356693784662])

    >>> np.random.seed(1864)
    >>> ak.from_series(pd.Series(np.random.choice([True, False],size=5)))
    array([True True True False False])

    >>> ak.from_series(pd.Series(['True', 'False', 'False', 'True', 'True']), dtype=bool)
    array([True True True True True])

    >>> ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
    array(['a', 'b', 'c', 'd', 'e'])

    >>> ak.from_series(pd.Series(pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01')])))
    array([1514764800000000000 1514764800000000000])

    Notes
    -----
    The supported datatypes are bool, float64, int64, string, and datetime64[ns]. The
    data type is either inferred from the the Series or is set via the dtype parameter.

    Series of datetime or timedelta are converted to Arkouda arrays of dtype int64 (nanoseconds)

    A Pandas Series containing strings has a dtype of object. Arkouda assumes the Series
    contains strings and sets the dtype to str
    """
    from arkouda.numpy.pdarraycreation import array

    if not dtype:
        dt = series.dtype.name
    else:
        dt = str(dtype)
    try:
        """
        If the Series has a object dtype, set dtype to string to comply with method
        signature that does not require a dtype; this is required because Pandas can infer
        non-str dtypes from the input np or Python array.
        """
        if dt == "object":
            dt = "string"

        n_array = series.to_numpy(dtype=SeriesDTypes[dt])  # type: ignore
    except KeyError:
        raise ValueError(
            f"dtype {dt} is unsupported. Supported dtypes are bool, float64, int64, string, "
            f"datetime64[ns], and timedelta64[ns]"
        )
    return array(n_array)
