from arkouda.client import generic_msg, verbose, maxTransferBytes, pdarrayIterThresh
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.pdarrayclass import *
from arkouda.pdarraycreation import *
from arkouda.pdarraysetops import *
from arkouda.numeric import value_counts
from arkouda.groupbyclass import *

import pandas as pd

__all__ = ["DFrame"]

class DFrame:
    # Arkouda implementation of a DataFrame. Data exists on the arkouda server. Current 
    # implementation serves as an organizational center for pointers to arrays on 
    # server. DFrame itself does not exist on the server.

    # To implement-
    # Attributes - name, columns, arrays, dtypes, shape, itemsize, ndim, index
    # Selection - filter
    # Getter - iloc, loc
    # Setter - __setitem__
    # Visual representation - __str__, __repr__
    # SortBy
    # PDarray - dealing with null, dropna
    # Integration with pandas - to_pddf
    # Aggregation - agg
    # Bool - any, all
    # Operations - abs?, add?
    # Concatenation - assign, append, drop, insertion
    # Stats - corr, cov

    objtype = 'df'
    BinOps = frozenset(["+", "-", "*", "/", "//", "%", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>","**"])
    OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=","**="])

    def __init__(self, data):

        # Using initial dictionary, create a dictionary of pdarrays. Each key
        # is a pointer to the array on the server that the column represents.

        self.data = {col: array(val) for col, val in data.items()}
        self.columns = list(self.data.keys())
        self.arrays = list(self.data.values())
        self.dtypes = [array.dtype for array in self.arrays]
        self.shape = [len(self.columns), len(self.arrays[0])]
        self.index = range(0, self.shape[1])
        self.size = self.shape[0] * self.shape[1]

    def __repr__(self):

        return repr(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}))

    def __str__(self):
        return str(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}))

    def __getitem__(self, key):
        if isinstance(key, int):
            newData = {col: [array[key]] for col, array in self.data.items()}
            return DFrame(newData)
        elif isinstance(key, str) and key in self.columns:
            index = self.columns.index(key)
            return self.arrays[index]
        elif isinstance(key, slice):
            newData = {col: array[key] for col, array in self.data.items()}
            return DFrame(newData)
    
    def __setitem__(self, key, val):
        return

    def filter(self, condition):

        #Given a condition, return all rows that satisfy

        return DFrame({col: val[condition] for col, val in self.data.items()})
    
    def head(self, amount=5):

        # Return the first "amount" entries

        return self[:amount]

    def tail(self, amount=5):

        # Return the last "amount" entries

        return self[-amount:]

    def value_counts(self, column):

        # Return the number of entries for each unique type

        return value_counts(self[column])

    def groupby(self, columns):

        # Create GroupBy object straight from DFrame

        if isinstance(columns, list):
            grouped = []
            for col in columns:
                grouped.append(self[col])
            return GroupBy(grouped)
        return GroupBy(self[columns])

    def sortby(self, columns):

        x = [[num] + list(self.data.keys()) for num in range(0, len(list(self.data.values())[0]))]
        newdict = {}
        for com in x:
            index = com[0]
            newdict[index] = {}
            for i in range(1, len(com)):
                column = com[i]
                newdict[index][column] = self.data[column][index]
        sortDict = sorted(newdict, key = lambda x: (newdict[x][col] for col in columns))
        return sortDict




    



        
    
