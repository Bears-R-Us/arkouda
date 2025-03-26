from collections import UserDict

from tabulate import tabulate


__all__ = [
    "Row",
]

"""
Row structure based on UserDict.
"""


class Row(UserDict):
    """
    A dictionary‐like representation of a single row in an Arkouda DataFrame.

    Wraps the column→value mapping for one row and provides convenient ASCII
    and HTML formatting for display.

    Parameters
    ----------
    data : dict
        Mapping of column names to their corresponding values for this row.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.row import Row
    >>> df = ak.DataFrame({'x': ak.array([10, 20]), 'y': ak.array(['a', 'b'])})

    Suppose df[0] returns {'x': 10, 'y': 'a'}
    >>> row = Row({'x': 10, 'y': 'a'})
    >>> print(row)
    keys    values
    ------  --------
    x       10
    y       a

    """

    def __str__(self) -> str:
        """
        Return an ASCII‐formatted table representation of the row.

        Returns
        -------
        str
            An ASCII table with two columns: 'keys' and 'values'.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.row import Row
        >>> row = Row({'a': 1, 'b': 'foo'})
        >>> print(str(row))
        keys    values
        ------  --------
        a       1
        b       foo

        """
        return tabulate(self.items(), headers=["keys", "values"], showindex=False)

    def __repr__(self) -> str:
        """
        Return the standard dictionary representation of the row.

        Returns
        -------
        str
            The string returned by `dict(self).__repr__()`.

        """
        return dict(self).__repr__()

    def _repr_html_(self) -> str:
        """
        Return an HTML‐formatted table representation of the row.

        Returns
        -------
        str
            An HTML table with two columns: 'keys' and 'values'.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.row import Row
        >>> row = Row({'a': 1, 'b': 'foo'})
        >>> html = row._repr_html_()
        >>> print(html.startswith('<table'))
        True

        """
        headers = ["keys", "values"]
        return tabulate(self.items(), headers=headers, tablefmt="html", showindex=False)
