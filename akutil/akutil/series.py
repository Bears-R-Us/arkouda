from pandas._config import get_option
import pandas as pd
import numpy as np
import akutil as aku
import arkouda as ak

__all__ = [
    "Series",
    ]

import operator

def natural_binary_operators(cls):
    for name, op in {
        '__add__': operator.add,
        '__sub__': operator.sub,
        '__mul__': operator.mul,
        '__truediv__': operator.truediv,
        '__floordiv__': operator.floordiv,
        '__and__': operator.and_,
        '__or__': operator.or_,
        '__xor__': operator.xor,
        '__eq__' : operator.eq,
        '__ge__' : operator.ge,
        '__gt__' : operator.gt,
        '__le__' : operator.le,
        '__lshift__' : operator.lshift,
        '__lt__' : operator.lt,
        '__mod__' : operator.mod,
        '__ne__' : operator.ne,
        '__or__' : operator.or_,
        '__rshift__' : operator.rshift,
    }.items():
        setattr(cls, name, cls._make_binop(op))

    return cls

def unary_operators(cls):
    for name, op in {
        '__invert__': operator.invert,
        '__neg__': operator.neg,
    }.items():
        setattr(cls, name, cls._make_unaryop(op))

    return cls

def aggregation_operators(cls):
        for name in ['max', 'min', 'mean', 'sum', 'std', 'argmax', 'argmin', 'prod'] :
            setattr(cls,name, cls._make_aggop(name))
        return cls


@unary_operators
@aggregation_operators
@natural_binary_operators
class Series:
    """
    One-dimensional arkouda array with axis labels. 
    
    Parameters
    ----------
    ar_tuple : 2-tuple of arkouda arrays with the first being the grouping key(s) and the 
             second being the value. The grouping key(s) will be the index of the series.
    
    """
    def __init__(self, ar_tuple=None,data=None, index=None):
        if ar_tuple is not None:
            self.index = aku.Index.factory(ar_tuple[0])
            self.values = ar_tuple[1]
        elif data is None:
            raise TypeError("ar_tuple and data cannot both be null")
        
        else:
            if not isinstance(data,ak.pdarrayclass.pdarray):
                data = ak.array(data)
            self.values= data
            
            if index is None:
                index = ak.arange(data.size)
            self.index = aku.Index.factory(index)
        if self.index.size != self.values.size:
            raise ValueError("Index and data must have same length")
        self.size = self.index.size
        
    def __len__(self):
        return self.values.size
    
    def __repr__(self):
        """
        Return ascii-formatted version of the series.
        """

        if len(self) == 0:
            return 'Series([ -- ][ 0 values : 0 B])'

        maxrows = pd.get_option('display.max_rows')
        if len(self) <= maxrows:
            prt = self.to_pandas()
            length_str = ""
        else:
            prt = pd.concat([self.head(maxrows//2+2).to_pandas(), 
                             self.tail(maxrows//2).to_pandas()])
            length_str = "\nLength {}".format(len(self))
        return prt.to_string(
            dtype=prt.dtype,
            min_rows=get_option("display.min_rows"),
            max_rows=maxrows,
            length=False,
        ) + length_str
    
    def __getitem__(self, key):
        if type(key) == Series:
            # @TODO align the series indexes
            key = key.values
        return Series ( ( self.index[key], self.values[key] ))
    
    def locate(self,key):
        """Lookup values by index label
        
        The input can be a scalar, a list of scalers, or a list of lists (if the series has a MultiIndex).
        As a special case, if a Series is used as the key, the series labels are preserved with its values
        use as the key.
        
        Keys will be turned into arkouda arrays as needed.
        
        Returns
        -------
        
        A Series containing the values corresponding to the key.
        """
        t =type(key)
        if isinstance(key,Series):
            # special case, keep the index values of the Series, and lookup the values
            labels = key.index
            key = key.values
            v = aku.lookup(self.index.index,self.values,key)
            return Series( (labels, v))
        elif isinstance(key,ak.pdarrayclass.pdarray):
            idx = self.index.lookup(key)
        elif t == list or t == tuple:
            key0 = key[0]
            if isinstance(key0,list) or isinstance(key0,tuple):
                # nested list. check if already arkouda arrays
                if not isinstance(key0[0], ak.pdarrayclass.pdarray):
                    # convert list of lists to list of pdarrays
                    key = [ ak.array(a) for a in np.array(key).T.copy() ]
                
            elif not isinstance(key0,ak.pdarrayclass.pdarray):
                # a list of scalers, convert into arkouda array
                key = ak.array(key)
            # else already list if arkouda array, use as is
            idx = self.index.lookup(key)
        else:
            # scalar value
            idx = self.index == key 
        return Series( (self.index[idx], self.values[idx]) )
            
            
    
    @classmethod
    def _make_binop(cls, operator):
        def binop(self, other):
                if type(other) == Series:
                    if self.index._check_aligned(other.index) :
                        return cls(  (self.index,operator(self.values, other.values)) ) 
                    else:
                        idx = self.index._merge(other.index).index
                        a = aku.lookup( self.index.index, self.values, idx, fillvalue=0)
                        b = aku.lookup( other.index.index, other.values, idx, fillvalue=0)
                        return cls( (idx,operator(a,b)))
                else:
                    return cls(  (self.index,operator(self.values, other)) ) 
        return binop

    @classmethod
    def _make_unaryop(cls, operator):
        def unaryop(self):
                    return cls(  (self.index,operator(self.values)) )
        return unaryop

    @classmethod
    def _make_aggop(cls, name):
        def aggop(self):
                return getattr(self.values,name)()
        return aggop

    def add(self,b):

        index = self.index.concat(b.index).index
            
        values = ak.concatenate( [self.values, b.values],ordered=False)
        return Series(ak.GroupBy( index).sum(values))

        
    def topn(self,n=10):
        """ Return the top values of the series
        
        Parameters
        ----------
        n: Number of values to return
        
        Returns
        -------
        A new Series with the top values
        """
        k = self.index
        v = self.values
        
        idx = ak.argmaxk(v,n)       
        idx = idx[-1:-n-1:-1]
        
        return Series( (k[idx], v[idx]))

    
    def sort_index(self,ascending=True):
        """ Sort the series by its index
             
        Returns
        -------
        A new Series sorted.
        """
        
        idx = self.index.argsort(ascending=ascending)
        return Series( (self.index[idx], self.values[idx]))
    
    def sort_values(self,ascending=True):
        """ Sort the series numerically
             
        Returns
        -------
        A new Series sorted smallest to largest
        """

        if not ascending:
            if isinstance(self.values, ak.pdarray) and self.values.dtype in (ak.int64, ak.float64):
                # For numeric values, negation reverses sort order
                idx = ak.argsort(-self.values)
            else:
                # For non-numeric values, need the descending arange because reverse slicing not supported
                idx = ak.argsort(self.values)[ak.arange(self.values.size-1, -1, -1)]
        else:
            idx = ak.argsort(self.values)
        return Series((self.index[idx], self.values[idx]))
        
    def tail(self,n=10):
        """Return the last n values of the series"""

        idx_series = (self.index[-n:])
        return Series( (idx_series, self.values[-n:]))
    
    def head(self,n=10):
        """Return the first n values of the series"""

        idx_series = (self.index[0:n])
        return Series( (idx_series, self.values[0:n]))

    
    def to_pandas (self):
        """Convert the series to a local PANDAS series"""

        idx = self.index.to_pandas()
        val = aku.util.convert_if_categorical(self.values)
        return pd.Series ( val.to_ndarray(), index=idx)
    
    def value_counts(self,sort=True):
        """Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        
        Parameters
        ----------
        
        sort : Boolean. Whether or not to sort the results.  Default is true.
        """
        
        s = Series(ak.value_counts(self.values))
        if sort:
            s = s.sort_values(ascending=False)
        return s
    
    def register(self,label):
        """Register the series with arkouda
        
        Parameters
        ----------
        label : Arkouda name used for the series
        
        Returns
        -------
        Numer of keys
        """
 
        retval = self.index.register(label)
        aku.register(self.values, "{}_value".format(label))
        return retval
    
    @staticmethod
    def attach(label,nkeys=1):
        """Retrieve a series registered with arkouda
        
        Parameters
        ----------
        label: name used to register the series
        nkeys: number of keys, if a multi-index was registerd
        """
        v = ak.attach_pdarray(label + "_value")
    
        if nkeys == 1:
            k = ak.attach_pdarray(label + "_key")
        else:
            k = [ ak.attach_pdarray("{}_key_{}".format(label,i)) for i in range(nkeys)]

        return Series ( (k,v) )
    
    @staticmethod
    def _all_aligned(array):
        """Is an array of Series indexed aligned?"""
        
        itor = iter(array)
        a1 = next(itor).index
        for a2 in itor:
            if a1._check_aligned(a2.index) == False:
                return False
        return True


    @staticmethod
    def concat( arrays, axis=0,index_labels=None, value_labels=None):
        """Concatenate in arkouda a list of arkouda Series or grouped arkouda arrays horizontally or vertically.
      
        If a list of grouped arkouda arrays is passed they are converted to a series. Each grouping is a 2-tuple 
        with the first item being the key(s) and the second being the value.

        If horizontal, each series or grouping must have the same length and the same index. The index of the series is 
        converted to a column in the dataframe.  If it is a multi-index,each level is converted to a column.

        Parameters
        ----------    
        arrays:  The list of series/groupings to concat.
        axis  :  Whether or not to do a verticle (axis=0) or horizontal (axis=1) concatenation
        index_labels:  column names(s) to label the index.
        value_labels:  column names to label values of each series.

        Returns
        -------
        axis=0: an arkouda series.
        axis=1: an arkouda dataframe.
        """

        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        if type(next(iter(arrays))) == tuple:
            arrays = [ Series(i) for i in arrays]

        if axis == 1:
            # Horizontal concat
            if value_labels == None:
                value_labels = [ "val_{}".format(i) for i in range(len(arrays))]
                
            if Series._all_aligned(arrays):

                data = next(iter(arrays)).index.to_dict(index_labels)

                for col,label in zip(arrays,value_labels):
                    data [ str(label) ] = col.values
                    
            else:
                aitor = iter(arrays)
                idx = next(aitor).index ;
                idx = idx._merge_all([ i.index for i in aitor])

                data = idx.to_dict(index_labels)
                
                for col,label in zip(arrays,value_labels):
                    data[str(label)] = aku.lookup( col.index.index, col.values, idx.index, fillvalue=0)

            retval =  aku.DataFrame( data) 
        else:
            # Verticle concat
            idx = arrays[0].index 
            v   = arrays[0].values
            for other in arrays[1:]:
                idx = idx.concat(other.index)
                v = ak.concatenate( [v,other.values], ordered=True )
            retval = aku.Series ( (idx,v))
            
        return retval

    @staticmethod
    def pdconcat( arrays, axis=0,labels=None):
        """Concatenate a list of arkouda Series or grouped arkouda arrays, returning a PANDAS object.

        If a list of grouped arkouda arrays is passed they are converted to a series. Each grouping is a 2-tuple 
        with the first item being the key(s) and the second being the value.

        If horizontal, each series or grouping must have the same length and the same index. The index of the series is 
        converted to a column in the dataframe.  If it is a multi-index,each level is converted to a column.

        Parameters
        ----------    
        arrays:  The list of series/groupings to concat.
        axis  :  Whether or not to do a verticle (axis=0) or horizontal (axis=1) concatenation
        labels:  names to give the columns of the data frame.

        Returns
        -------
        axis=0: a local PANDAS series
        axis=1: a local PANDAS dataframe
        """
        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        if type(arrays[0]) == tuple:
            arrays = [ Series(i) for i in arrays]

        if axis == 1:
            idx = arrays[0].index.to_pandas()

            cols = []
            for col in arrays:
                cols.append( pd.Series ( col.values.to_ndarray(), index=idx) )
            retval =  pd.concat ( cols,axis =1 )
            if labels != None:
                retval.columns=labels
        else:
            retval = pd.concat( [ s.to_pandas() for s in arrays ] )
             
        return retval

