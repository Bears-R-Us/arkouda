from collections import UserDict
from tabulate import tabulate
import akutil as aku

__all__ = [ "Row", ]

"""
Row structure based on UserDict.
"""

class Row(UserDict):
    """
    This class is useful for printing and working with individual rows of a
    of an aku.DataFrame.
    """

    def __str__(self):
        """
        Return ascii-formatted version of the dataframe.
        """

        return tabulate(self.items(), headers=['keys', 'values'], showindex=False)

    def __repr__(self):
        return dict(self).__repr__()

    def _repr_html_(self):
        """
        Return html-formatted version of the dataframe.
        """

        headers = ['keys', 'values']
        return tabulate(self.items(), headers = headers, tablefmt='html', showindex=False)
