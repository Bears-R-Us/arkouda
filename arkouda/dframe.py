from arkouda.client import generic_msg, verbose, maxTransferBytes, pdarrayIterThresh
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.pdarrayclass import *
from arkouda.pdarraycreation import *
from arkouda.pdarraysetops import *
from arkouda.numeric import *
from arkouda.groupbyclass import *
from arkouda.sorting import * 

import pandas as pd

__all__ = ["DFrame"]

class DFrame:

    """
    Arkouda implementation of a DataFrame. Data exists on the arkouda server. Current 
    implementation serves as an organizational center for pointers to arrays on 
    server and can only handle int/float pdarrays. Data is kept as a dictionary of pdarrays. 
    DFrame itself does not exist on the server. Operations on the DFrame serve as calls to
    pdarray methods.

    

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
    #

    """

    objtype = 'DFrame'

    def __init__(self, data, index = None):
        """
        Parameters
        ----------
        data :  array_like
            dictionary or pandas DataFrame

        Returns
        -------
        ak.DFrame
            Instance of arkouda DFrame that serves as a dictionary of pdarrays

        """
        self.Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])
    
        if isinstance(data, dict):
            self.data = {col: array(val) for col, val in data.items()}
            
        elif isinstance(data, pd.DataFrame):
            self.data = {col: data[col] for col in data.columns}

        else:
            raise TypeError("data must be dictionary or pandas DataFrame")
        self.columns = list(self.data.keys())
        self.arrays = list(self.data.values())
        self.dtypes = [array.dtype for array in self.arrays]
        self.shape = [len(self.arrays[0]), len(self.columns)]

        if index == None:
            self.index = range(0, self.shape[0])
        else:
            self.index = index
        
        self.size = self.shape[0] * self.shape[1]
        self.axes = (index, self.columns)
        self.ndim = self.shape[0]


    def __repr__(self):
        """
        Currently using pandas representation, requiring transfer of data to pandas. Using until
        a more efficient, yet still visually good representation is found.

        """

        return repr(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}, index = self.index))

    def __str__(self):

        """
        Currently using pandas string representation, requiring transfer of data to pandas. Using until
        a more efficient, yet still visually good representation is found.

        """

        return str(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}))

    def __getitem__(self, key):

        if isinstance(key, int) or isinstance(key, slice) or isinstance(pdarray):
            newData = {col: [array[key]] for col, array in self.data.items()}
            return DFrame(newData)
        elif isinstance(key, str) and key in self.columns:
            index = self.columns.index(key)
            return self.arrays[index]
        # elif isinstance(key, slice):
        #     newData = {col: array[key] for col, array in self.data.items()}
        #     return DFrame(newData)
        # elif isinstance(key, pdarray):
        #     newData = {col: array[key] for col, array in self.data.items()}
        #     return DFrame(newData)
    
    def __setitem__(self, key, val):

        return

    def filter(self, condition):

        # Given a condition, return all rows that satisfy

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

    def sortby(self, column):
        
        sorting = argsort(self.data[column])
        sortDict = {col: val[sorting] for col, val in self.data.items()}
        return sortDict

    def to_pddf(self):

        return pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}, index = self.index)

    def abs(self):

        return DFrame({col: abs(self.data[col]) for col in self.columns})

    def agg(self, func, axis = 0):
        if func not in self.Reductions:
            raise ValueError("Unsupported reduction: {}\nMust be one of {}".format(func, self.Reductions))

        if func == "min":
            apply = min
        elif func == "max":
            apply = max
        elif func == "any":
            apply = any
        elif func == 'all':
            apply = all
        elif func == "mean":
            apply = mean
        elif func == "prod":
            apply = prod
        elif func == "sum":
            apply = sum
        elif func == "argmin":
            apply = argmin
        elif func == "argmax":
            apply == argmax      

        if axis == 0:
            aggData = {col: apply(self.data[col]) for col in self.columns}
            return DFrame(aggData, index = self.index)
        elif axis == 1:
            aggData = {i: apply(self.data[i]) for i in self.index}
            return DFrame(aggData, index = self.columns)
        else:
            raise ValueError("axis must be 0 or 1") 

    def all(self, axis = 0):
        
        return agg(all, axis)

    def any(self, axis = 0):
        
        return agg(any, axis)

    def max(self, axis = 0):

        return agg(max, axis)

    def min(self, axis = 0):
        
        return agg(min, axis)

    def argmin(self, axis = 0):
        
        return agg(argmin, axis)

    def argmax(self, axis = 0):

        return agg(argmax, axis)

    def prod(self, axis = 0):
        
        return agg(prod, axis)

    def sum(self, axis = 0):
        
        return agg(sum, axis)

    def mean(self, axis = 0):

        return agg(mean, axis)

    def transpose(self):

        transposed = zip(*self.arrays)
        transposedList = list(transposed)
        return DFrame({i: transposedList[i] for i in index}, index = self.columns)
    
    def assign(self, dictionary):

        for key, val in dictionary.items():
            self.data[key] = val
            self.columns.append(key)
            self.arrays.append(val)
            self.dtypes.append(val.dtype)
            self.shape[1] += 1
            self.size = self.shape[0] * self.shape[1]
            self.axes = (index, self.columns)
            self.ndim = self.shape[0]
    
    def drop_columns(self, cols):

        for col in cols:
            self.columns.remove(col)
            self.arrays.remove(self.data[col])
            self.dtypes.remove(self.data[col].dtype)
            self.data.pop(col, None)
            self.shape[1] -= 1
            self.size = self.shape[0] * self.shape[1]
            self.axes = (index, self.columns)
            self.ndim = self.shape[0]
    
    BinOps = frozenset(["+", "-", "*", "/", "//", "%", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>","**"])
    OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=","**="])

    def __add__(self, other):
        
        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] + other[col]

        return DFrame(newDict, index = self.index)

    def __sub__(self, other):

        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] - other[col]

        return DFrame(newDict, index = self.index)
    
    def __rsub__(self, other):

        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] - other[col]

        return DFrame(newDict, index = self.index)
    
    def __mul__(self, other):

        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] * other[col]

        return DFrame(newDict, index = self.index)

    def __truediv__(self, other):

        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] / other[col]

        return DFrame(newDict, index = self.index)

    def __floordiv__(self, other):

        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] // other[col]

        return DFrame(newDict, index = self.index)

    def __mod__(self, other):
        
        newDict = {}
        for col in self.columns:
            newDict[col] = self.data[col] % other[col]

        return DFrame(newDict, index = self.index)

    def appendRow(self, pda, indice):
        pdaLength = range(0, len(pda))

        for i in pdaLength:
            ak.concatenate(self.data[self.columns[i]], pda[i])
        
        self.arrays = list(self.data.values())
        self.dtypes = [array.dtype for array in self.arrays]
        self.shape[0] += 1
        self.index.append(indice)
        self.size = self.shape[0] * self.shape[1]
        self.axes = (index, self.columns)
        self.ndim = self.shape[0]
        
    
            

    



    



        
    
