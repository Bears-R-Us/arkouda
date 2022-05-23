import pandas as pd  # type: ignore
from typing import Optional
import json
from typing import cast as typecast

from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import arange, ones
from arkouda.pdarraysetops import argsort, in1d
from arkouda.sorting import coargsort
from arkouda.dtypes import int64 as akint64, float64 as akfloat64, bool as akbool
from arkouda.util import register, convert_if_categorical, concatenate, get_callback
from arkouda.groupbyclass import unique, GroupBy
from arkouda.alignment import in1dmulti
from arkouda.infoclass import list_registry
from arkouda.client import generic_msg


class Index:
    def __init__(self, index):
        self.index = index
        self.size = index.size
        self.name: Optional[str] = None

    def __getitem__(self,key):
        from arkouda.series import Series
        if type(key) == Series:
            key = key.values
        return Index(self.index[key])

    def __repr__(self):
        return repr(self.index)

    def __len__(self):
        return len(self.index)

    def __eq__(self,v):
        if isinstance(v, Index):
            return self.index == v.index
        return self.index == v

    @staticmethod
    def factory(index):
        t = type(index)
        if isinstance(index, Index):
            return index
        elif t != list and t != tuple:
            return Index(index)
        else:
            return MultiIndex(index)

    def to_pandas(self):
        val = convert_if_categorical(self.index)
        return val.to_ndarray()

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = dtype(self.index)
        self.index = new_idx
        return self

    def register(self, label):
        register(self.index, "{}_key".format(label))
        self.name = label
        return 1

    def is_registered(self):
        """
        Return True if the object is contained in the registry

        Returns
        -------
        bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown
        """
        return f"{self.name}_key" in list_registry()

    def to_dict(self, label):
        data = {}
        if label is None:
            label = "idx"
        elif type(label) == list:
            label = label[0]
        data[label] = self.index
        return data

    def _check_types(self, other):
        if type(self) != type(other):
            raise TypeError("Index Types must match")

    def _merge(self, other):
        self._check_types(other)

        callback = get_callback(self.index)
        idx = concatenate([self.index, other.index], ordered=False)
        return Index(callback(unique(idx)))

    def _merge_all(self, array):
        idx = self.index
        callback = get_callback(idx)

        for other in array:
            self._check_types(other)
            idx = concatenate([idx, other.index], ordered=False)

        return Index(callback(unique(idx)))

    def _check_aligned(self, other):
        self._check_types(other)
        l = len(self)
        return len(other) == l and (self == other.index).sum() == l

    def argsort(self, ascending=True):
        if not ascending:
            if isinstance(self.index, pdarray) and self.index.dtype in (akint64, akfloat64):
                i = argsort(-self.index)
            else:
                i = argsort(self.index)[arange(self.index.size - 1, -1, -1)]
        else:
            i = argsort(self.index)
        return i

    def concat(self, other):
        self._check_types(other)

        idx = concatenate([self.index, other.index], ordered=True)
        return Index(idx)

    def lookup(self, key):
        if not isinstance(key, pdarray):
            raise TypeError("Lookup must be on an arkouda array")

        return in1d(self.index, key)

    def save(self, prefix_path: str, dataset: str = 'index', mode: str = 'truncate',
             compressed: bool = False, file_format: str = 'HDF5') -> str:
        """
        Save the index to HDF5 or Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compressed : bool
            Defaults to False. When True, files will be written with Snappy compression
            and RLE bit packing. This is currently only supported on Parquet files and will
            not impact the generated files when writing HDF5 files.
        file_format : str {'HDF5', 'Parquet'}
            By default, saved files will be written to the HDF5 file format. If
            'Parquet', the files will be written to the Parquet file format. This
            is case insensitive.

        Returns
        -------
        string message indicating result of save operation

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        ValueError
            Raised if there is an error in parsing the prefix path pointing to
            file write location or if the mode parameter is neither truncate
            nor append
        TypeError
            Raised if any one of the prefix_path, dataset, or mode parameters
            is not a string

        See Also
        --------
        save_all, load, read

        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must
        have write permission.

        Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        Previously all files saved in Parquet format were saved with a ``.parquet`` file extension.
        This will require you to use load as if you saved the file with the extension. Try this if
        an older file is not being found.

        Any file extension can be used. The file I/O does not rely on the extension to determine the file format.
        """
        if mode.lower() in ['a', 'app', 'append']:
            m = 1
        elif mode.lower() in ['t', 'trunc', 'truncate']:
            m = 0
        else:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        if file_format.lower() == 'hdf5':
            cmd = "tohdf"
        elif file_format.lower() == 'parquet':
            cmd = "writeParquet"
        else:
            raise ValueError("Supported file formats are 'HDF5' and 'Parquet'")

        """
        If offsets are provided, add to the json_array as the offsets will be used to 
        retrieve the array elements from the hdf5 files.
        """
        try:
            json_array = json.dumps([prefix_path])
        except Exception as e:
            raise ValueError(e)
        strings_placeholder = False

        return typecast(str, generic_msg(cmd=cmd, args=f"{self.index.name} {dataset} {m} {json_array} "
                                                   f"{self.index.dtype} {strings_placeholder} {compressed}"))


class MultiIndex(Index):
    def __init__(self,index):
        if not(isinstance(index,list) or isinstance(index,tuple)):
            raise TypeError("MultiIndex should be an iterable")
        self.index = index
        first = True
        for col in self.index:
            if first:
                self.size = col.size
                first = False
            else:
                if col.size != self.size:
                    raise ValueError("All columns in MultiIndex must have same length")
        self.levels = len(self.index)

    def __getitem__(self,key):
        from arkouda.series import Series
        if type(key) == Series:
            key = key.values
        return MultiIndex([ i[key] for i in self.index])

    def __len__(self):
        return len(self.index[0])

    def __eq__(self,v):
        if type(v) != list and type(v) != tuple:
            raise TypeError("Cannot compare MultiIndex to a scalar")
        retval = ones(len(self), dtype=akbool)
        for a,b in zip(self.index, v):
            retval &= (a == b)
        return retval

    def to_pandas(self):
        idx = [convert_if_categorical(i) for i in self.index]
        mi = [i.to_ndarray() for i in idx]
        return pd.Series(index=mi, dtype='float64').index

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = [dtype(i) for i in self.index]
        self.index = new_idx
        return self

    def register(self, label):
        for i, arr in enumerate(self.index):
            register(arr, "{}_key_{}".format(label, i))
        return len(self.index)

    def to_dict(self, labels):
        data = {}
        if labels is None:
            labels = ["idx_{}".format(i) for i in range(len(self.index))]
        for i, value in enumerate(self.index):
            data[labels[i]] = value
        return data

    def _merge(self, other):
        self._check_types(other)
        idx = [concatenate([ix1, ix2], ordered=False) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(GroupBy(idx).unique_keys)

    def _merge_all(self, array):
        idx = self.index

        for other in array:
            self._check_types(other)
            idx = [concatenate([ix1, ix2], ordered=False) for ix1, ix2 in zip(idx, other.index)]

        return MultiIndex(GroupBy(idx).unique_keys)

    def argsort(self, ascending=True):
        i = coargsort(self.index)
        if not ascending:
            i = i[arange(self.size - 1, -1, -1)]
        return i

    def concat(self, other):
        self._check_types(other)
        idx = [concatenate([ix1, ix2], ordered=True) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(idx)

    def lookup(self, key):
        if type(key) != list and type(key) != tuple:
            raise TypeError("MultiIndex lookup failure")

        return in1dmulti(self.index, key)
