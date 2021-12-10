import pandas as pd
import akutil as aku
import arkouda as ak

class Index:
    def __init__(self,index):
        self.index = index
        self.size = index.size
    @staticmethod
    def factory(index):
        t = type(index)
        if isinstance(index,Index):
            return index
        elif t != list and t != tuple:
            return Index(index)
        else:
            return MultiIndex(index)
    def to_pandas(self):
        val = aku.util.convert_if_categorical(self.index)
        return val.to_ndarray()
    def set_dtype(self,dtype):
        """Change the data type of the index
        
        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = dtype(self.index)
        self.index = new_idx
        return self
    def register(self,label):

        aku.register(self.index, "{}_key".format(label))
        return 1
    
    def to_dict(self,label):
        data = {}
        if label == None:
            label = "idx"
        elif type(label) == list:
            label = label[0]
        data [ label] = self.index
        return data
    
    def _check_types(self,other):
        if type(self) != type(other):
            raise TypeError("Index Types must match")
    
    def _merge(self,other):
        self._check_types(other)
        
        callback = aku.get_callback(self.index)
        idx = aku.concatenate([self.index,other.index],ordered=False)
        return Index(callback(ak.unique(idx)))

    def _merge_all(self,array):
        
        idx = self.index
        callback = aku.get_callback(idx)
        
        for other in array:
            
            self._check_types(other)
            idx = aku.concatenate([idx,other.index],ordered=False)
            
        return Index(callback(ak.unique(idx)))

    def _check_aligned(self,other):
        self._check_types(other)
        l = len(self)
        return len(other) == l  and (self == other.index).sum() == l
    
    def argsort(self, ascending=True):
        if not ascending:
            if isinstance(self.index, ak.pdarray) and self.index.dtype in (ak.int64, ak.float64):
                i = ak.argsort(-self.index)
            else:
                i = ak.argsort(self.index)[ak.arange(self.index.size-1, -1, -1)]
        else:
            i = ak.argsort(self.index)
        return i
    
    def concat(self,other):
        self._check_types(other)
        
        idx = aku.concatenate([self.index,other.index],ordered=True)
        return Index(idx)
    
    def lookup(self,key):
        if not isinstance(key,ak.pdarrayclass.pdarray):
            raise TypeError("Lookup must be on an arkouda array")
            
        return ak.in1d(self.index,key)


    def __getitem__(self,key):
        if type(key) == aku.Series:
            key = key.values
        return Index(self.index[key])
    def __repr__(self):
        return repr(self.index)
    def __len__(self):
        return len(self.index)
    def __eq__(self,v):
        return self.index == v
        
class MultiIndex(Index):
    def __init__(self,index):
        if not(isinstance(index,list) or isinstance(index,tuple)):
            raise TypeError("MultiIndex shuold be an iterable")    
        self.index = index
        first = True
        for col in self.index:
            if first:
                self.size = col.size
            else:
                if col.size != self.size:
                    raise ValueError("All columns in MultiIndex must have same length")
        self.levels = len(self.index)
    def to_pandas(self):
        idx = [ aku.util.convert_if_categorical(i) for i in self.index ]
        mi = [ i.to_ndarray() for i in idx ]
        return pd.Series(index=mi,dtype='float64').index 
    def set_dtype(self,dtype):
        """Change the data type of the index
        
        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = [ dtype(i) for i in self.index]
        self.index = new_idx
        return self
    def register(self,label):
 
        for i,arr in enumerate(self.index):
            aku.register(arr, "{}_key_{}".format(label,i))
        return len(self.index)
    
    def to_dict(self,labels):
        data = {}
        if labels == None:
            labels = [ "idx_{}".format(i) for i in range(len(self.index))]
        for i,value in enumerate(self.index):
            data [ labels[i]] = value
        return data 

    def _merge(self,other):
        self._check_types(other)
        
        idx = [ aku.concatenate( [ix1, ix2 ],ordered=False ) for ix1,ix2 in zip(self.index,other.index) ]

        return MultiIndex(ak.GroupBy(idx).unique_keys)
    
    def _merge_all(self,array):
        
        idx = self.index
        
        for other in array:
            self._check_types(other)
            idx = [ aku.concatenate( [ix1, ix2 ],ordered=False ) for ix1,ix2 in zip(idx,other.index) ]
            
        return MultiIndex(ak.GroupBy(idx).unique_keys)

    def argsort(self, ascending=True):
        
        i = ak.coargsort(self.index)
        if not ascending:
            i = i[ak.arange(self.size-1, -1, -1)]
        return i
    
    def concat (self,other):
        self._check_types(other)
        
        idx = [ aku.concatenate( [ix1, ix2 ],ordered=True ) for ix1,ix2 in zip(self.index,other.index) ]

        return MultiIndex(idx)
    
    def lookup(self,key):
        if type(key) != list and type(key) != tuple:
            raise TypeError("MultiIndex lookup failure")
            
        return aku.in1dmulti(self.index,key)


    def __getitem__(self,key):
        if type(key) == aku.Series:
            key = key.values
        return MultiIndex([ i[key] for i in self.index])
    def __len__(self):
        return len(self.index[0])
    def __eq__(self,v):
        if type(v) != list and type(v) != tuple:
            raise TypeError("Cannot compare MultiIndex to a scalar")
        retval = ak.ones(len(self),dtype=ak.bool )
        for a,b in zip(self.index,v) :
            retval &= (a==b)
        return retval


