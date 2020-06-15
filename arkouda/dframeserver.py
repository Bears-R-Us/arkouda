from arkouda.client import generic_msg, verbose, maxTransferBytes, pdarrayIterThresh
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS
from arkouda.pdarrayclass import *
from arkouda.pdarraycreation import *
from arkouda.pdarraysetops import *

import pandas as pd

__all__ = ["DFrameServer", "dFrame"]

class DFrameServer:
    # Arkouda implementation of a DataFrame. Data exists on the arkouda server.
    # To implement-
    # Attributes - name, columns, arrays, dtypes, shape, itemsize, ndim, index
    # Selection - filter
    # Getter - iloc, loc
    # Setter - __setitem__
    # Visual representation - __str__, __repr__, head, tail
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

    def __init__(self, columns, arrays):
        
        self.columns = columns
        self.arrays = arrays
        self.name = name
        # self.dtypes = dtypes
        # self.shape = shape
        # self.ndim = ndim
        # self.index = index
        # self.axes = (index, columns)
        # self.size = shape[0] * shape[1]

    def __del__(self):
        try:
            generic_msg("delete {}".format(self.name))
        except:
            pass

    


def dFrame(dictionary):
    columns = list(dictionary.keys())
    arrays = [array(d) for d in dictionary.values()]
    arrayNames = str([array.name for array in arrays]).replace(" ", "")
    reqMsg = "dframe {} {}".format(len(columns), arrayNames)
    repMsg = generic_msg(reqMsg)
    return create_dframe(repMsg)

def create_dframe(repMsg):
    fields = repMsg.split()
    columns = fields[1]
    arrays = fields[2]
    # arrays = int(fields[3])
    # dtypes = arrays
    # shape = [int(el) for el in fields[4][1:-1].split(',')]
    # index = fields[5]
    # ndim =  fields[6]
    return DFrameServer(columns,arrays)