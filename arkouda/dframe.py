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

__all__ = ["DFrame", "to_akframe", "convert_columns"]

class DFrame:

    """
    Arkouda implementation of a DataFrame. Data exists on the arkouda server. Current 
    implementation serves as an organizational center for pointers to arrays on 
    server and can only handle int/float pdarrays. Data is kept as a dictionary of pdarrays. 
    DFrame itself does not exist on the server. Operations on the DFrame serve as calls to
    pdarray methods.

    Attributes
    ----------
    data : dictionary of pdarrays
        Collection of pointers to pdarrays, where each key is the name of the column
        corresponding to the pdarray value
    columns : list of strings
        An ordered list of column names
    arrays : list of pdarrays
        An ordered list of pdarrays, each corresponding to the same index of columns
    dtypes : list of dtypes
        An ordered list of data types corresponding to arrays
    shape : tuple
        The sizes of each dimension of the DFrame
    index : list
        An ordered list of row names
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    size : int
        The number of elements in the DFrame
    axes : tuple
        The names of each axis of the DFrame
    ndim : int
        The length of one array
    itemsize : int
        The size in bytes of each element

    TODO
    -------
    Dealing with NaN values
    Removal of rows
    Fix index with methods
    """

    objtype = 'DFrame'

    def __init__(self, data, index = False):
        """
        Parameters
        ----------
        data :  array_like
            dictionary or pandas DataFrame

        Returns
        -------
        ak.DFrame
            Instance of arkouda DFrame that serves as a dictionary of pdarrays

        TODO
        -------
        More or less attributes

        """
    
        if isinstance(data, dict):
            self.data = {col: array(val) for col, val in data.items()}
        elif isinstance(data, pd.DataFrame):
            self.data = {col: array(data[col]) for col in data.columns}
        else:
            raise TypeError("data must be dictionary or pandas DataFrame")

        self.columns = list(self.data.keys())
        self.arrays = list(self.data.values())
        self.dtypes = [array.dtype for array in self.arrays]
        self.shape = [len(self.arrays[0]), len(self.columns)]

        if index != False:
            self.index = index
        else:
            self.index = range(0, self.shape[0])
        
        self.size = self.shape[0] * self.shape[1]
        self.axes = (index, self.columns)
        self.ndim = self.shape[0]
        self.Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])


    def __repr__(self):
        """
        TODO
        -------
        Come up with better visualization that does not require transferring to pandas
        """

        return repr(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}, index = self.index))

    def __str__(self):

        """
        TODO
        -------
        Come up with better visualization that does not require transferring to pandas
        """

        return str(pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}))

    def __getitem__(self, key):

        """
        TODO
        -------
        [index, col]
        [col:col]
        [index:index, col]
        [index, col:col]
        isinstance can't just check int or slice
        Raise error for bad key
        """
        

        if isinstance(key, int) or isinstance(key, str):
            if key not in self.columns:
                raise TypeError("key must exist in either the index or the columns")
            elif key in self.columns:
                index = self.columns.index(key)
                return self.arrays[index]

        elif isinstance(key, slice) or isinstance(key, pdarray):
            newData = {col: array[key] for col, array in self.data.items()}
            return DFrame(newData, index = self.index[key])

    def __setitem__(self, key, val):

        if key in self.columns:
            self.data[key] = val
        elif key in self.index:
            for col in self.columns:
                self.data[col][key] = val

        """
        TODO
        -------
        Consider where index and columns have same values
        """

        return

    def filter(self, condition):

        """
        Filters out all unwanted rows of the DFrame. Created before __getitem__, 
        but serves the same purpose as __getitem__. 

        Parameters
        ----------
        condition :  boolean array
            filter condition

        Returns
        -------
        ak.DFrame
            Subsection of DFrame that satisfies condition.

        TODO
        -------
        Raise error for bad condition
        """
        if not isinstance(condition, pdarray) :
            raise TypeError("Condition must be boolean list")
        filterIndex = array(self.index)[condition].to_ndarray().tolist()
        return DFrame({col: val[condition] for col, val in self.data.items()}, index = filterIndex)
    
    def head(self, limit = 5):

        """
        Exact same as pandas DataFrame head
        """
        if not isinstance(limit, int):
            raise TypeError("Invalid limit")
        return self[:limit]

    def tail(self, limit = 5):

        """
        Exact same as pandas DataFrame tail
        """
        if not isinstance(limit, int):
            raise TypeError("Invalid limit")
        return self[-limit:]

    def count(self):

        """
        Apply value_counts to every column of the DFrame.

        TODO
        -------
        Make sure it works
        """

        countData = {}
        for col in self.columns:
            countData[col] = value_counts(self.data[col])
        return DFrame(countData, index = "count")     

    def groupby(self, cols):

        """
        Directly creates GroupBy object from the columns of the DFrame. More similar
        to pandas GroupBy creation as it does not need the DFrame name for every 
        column.

        TODO
        -------
        Raise error for columns not in self.columns
        """

        check = array([col in self.columns for col in cols])

        if not all(check):
            raise TypeError("Nonexistant column(s) in parameter cols")

        if isinstance(cols, list):
            grouped = []
            for col in cols:
                grouped.append(self[col])
            return GroupBy(grouped)
        return GroupBy(self[cols])

    def sortby(self, cols):

        """
        Currently takes one column and sorts rows by one column. Ideally, it could
        sort with multiple columns, requiring tiebreaking.

        TODO
        -------
        Raise error for columns not in self.columns
        Implement more advanced implementation of sort (probably requires Chapel code)
        """
        check = array([col in self.columns for col in cols])

        if not all(check):
            raise TypeError("Nonexistant column(s) in parameter cols")

        sortOrder = [self.data[col] for col in cols]
        sorting = coargsort(sortOrder)
        sortDict = {col: val[sorting] for col, val in self.data.items()}
        return DFrame(sortDict, index = self.index)

    def to_pddf(self):

        """ 
        Integration with pandas

        Returns
        -------
        pd.DataFrame
            pandas DataFrame with all of the arkouda DFrame data transferred to it
        """

        return pd.DataFrame({col: array.to_ndarray() for col, array in self.data.items()}, index = self.index)

    def abs(self):

        """
        Apply abs function to every column to get DFrame with only nonnegative
        values

        TODO
        -------
        Fix index
        """

        return DFrame({col: abs(self.data[col]) for col in self.columns}, index = self.index)

    def agg(self, func, axis = 0):
        
        """
        Aggregation

        TODO
        -------
        Should add a columns parameter
        Condense code
        """
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

    # Aggregate columns with all function
    def all(self, axis = 0):
        
        return agg(all, axis)

    # Aggregate columns with "any" function
    def any(self, axis = 0):
        
        return agg(any, axis)

    # Aggregate columns with max function
    def max(self, axis = 0):

        return agg(max, axis)

    # Aggregate columns with min function
    def min(self, axis = 0):
        
        return agg(min, axis)
    
    # Aggregate columns with argmin function
    def argmin(self, axis = 0):
        
        return agg(argmin, axis)

    # Aggregate columns with argmax function
    def argmax(self, axis = 0):

        return agg(argmax, axis)

    # Aggregate columns with prod function
    def prod(self, axis = 0):
        
        return agg(prod, axis)

    # Aggregate columns with sum function
    def sum(self, axis = 0):
        
        return agg(sum, axis)

    # Aggregate columns with mean function
    def mean(self, axis = 0):

        return agg(mean, axis)

    def transpose(self):

        """
        Take transpose of DFrame

        Returns
        -------
        New instance of DFrame with rows and columns flipped

        TODO
        -------
        Make sure it works
        """

        transposed = zip(*self.arrays)
        transposedList = list(transposed)
        return DFrame({i: transposedList[i] for i in self.index}, index = self.columns)
    
    def assign(self, dictionary):

        """
        Assign all "columns" in dictionary as columns in DFrame. Alter attributes
        accordingly.

        TODO
        -------
        Ensure that dictionary is dictionary of pdarrays and all arrays are the 
        same length as arrays in DFrame
        """
        check = array([col in self.columns for col in dictionary.keys])
        check2 = array([array.size == shape[0] for array in self.arrays])

        if not all(check) and not all(check2):
            raise TypeError("Nonexistant column(s) in input dictionary and array(s) of improper length exist")
        elif not all(check2):
            raise TypeError("Array(s) of improper length exist")
        elif not all(check):
            raise TypeError("Nonexistant column(s) in input dictionary")
    
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

        """
        Remove unwanted columns and change attributes accordingly

        TODO
        -------
        Make sure that cols in self.columns

        """
        check = array([col in self.columns for col in dictionary.keys])
        if not all(check):
            raise TypeError("Nonexistant column(s) in argument cols")


        for col in cols:
            self.columns.remove(col)
            self.arrays.remove(self.data[col])
            self.dtypes.remove(self.data[col].dtype)
            self.data.pop(col, None)
            self.shape[1] -= 1
            self.size = self.shape[0] * self.shape[1]
            self.axes = (index, self.columns)
            self.ndim = self.shape[0]
    
    """
    TODO
    -------
    Check that all operations work
    Implement r versions
    """
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

    # def appendRow(self, pda, indice = self.shape[0] + 1):

    #     """
    #     Add row to end of DFrame with indicated indice.

    #     TODO
    #     -------
    #     Make sure input is a pda
    #     """
    #     if pda.size != self.shape[1]:
    #         raise ValueError("pda is not the right size")
       
    #     pdaLength = range(0, len(pda))

    #     for i in pdaLength:
    #         ak.concatenate(self.data[self.columns[i]], pda[i])
        
    #     self.arrays = list(self.data.values())
    #     self.dtypes = [array.dtype for array in self.arrays]
    #     self.shape[0] += 1
    #     self.index.append(indice)
    #     self.size = self.shape[0] * self.shape[1]
    #     self.axes = (index, self.columns)
    #     self.ndim = self.shape[0]

def to_akframe(df): 
    names = []
    dictionaries = []

    for col in df.columns:
        if df[col].dtype == pd.Timestamp:
            altered = df[col] - df[col].min()
            df[col] = altered.apply(lambda x: x.total_seconds())

        elif df[col].dtype != int:
            dictionaries.append(dict(zip(df[col].unique(), range(0, df[col].unique().size))))
            names.append(col)
    
    if len(names) >= 1: 
        df = convert_columns(df, names, dictionaries)
    
    return DFrame(df)

def convert_columns(data, columns, dictionaries):
    data_copy = data.copy()
    i = 0
    for col in columns:
        data_copy[col] = data_copy[col].map(dictionaries[i])
        i += 1
    return data_copy



    

        
    
            

    



    



        
    
