from typing import List, Optional, Union
from typing import cast as typecast
from warnings import warn

import pandas as pd  # type: ignore
from typeguard import typechecked

from arkouda import Strings
from arkouda.client import generic_msg
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.groupbyclass import GroupBy, unique
from arkouda.infoclass import list_registry
from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import arange, array, ones
from arkouda.pdarraysetops import argsort, in1d
from arkouda.sorting import coargsort
from arkouda.util import convert_if_categorical, generic_concat, get_callback, register


class Index:
    @typechecked
    def __init__(
        self, values: Union[List, pdarray, Strings, pd.Index, "Index"], name: Optional[str] = None
    ):
        if isinstance(values, Index):
            self.values = values.values
            self.size = values.size
            self.dtype = values.dtype
            self.name = name if name else values.name
            return
        elif isinstance(values, pd.Index):
            self.values = array(values.values)
            self.size = values.size
            self.dtype = self.values.dtype
            self.name = name if name else values.name
            return
        elif isinstance(values, List):
            values = array(values)

        self.values = values
        self.size = self.values.size
        self.dtype = self.values.dtype
        self.name = name

    def __getitem__(self, key):
        from arkouda.series import Series

        if isinstance(key, Series):
            key = key.values

        if isinstance(key, int):
            return self.values[key]

        return Index(self.values[key])

    def __repr__(self):
        # Configured to match pandas
        return f"Index({repr(self.index)}, dtype='{self.dtype}')"

    def __len__(self):
        return len(self.index)

    def __eq__(self, v):
        if isinstance(v, Index):
            return self.index == v.index
        return self.index == v

    @property
    def index(self):
        """
        This is maintained to support older code
        """
        return self.values

    @property
    def shape(self):
        return (self.size,)

    @property
    def is_unique(self):
        """
        Property indicating if all values in the index are unique
        Returns
        -------
            bool - True if all values are unique, False otherwise.
        """
        g = GroupBy(self.values)
        key, ct = g.count()
        return (ct == 1).all()

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
        val = convert_if_categorical(self.values).to_ndarray()
        return pd.Index(data=val, dtype=val.dtype, name=self.name)

    def to_ndarray(self):
        val = convert_if_categorical(self.values)
        return val.to_ndarray()

    def to_list(self):
        return self.to_ndarray().tolist()

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = dtype(self.values)
        self.values = new_idx
        return self

    def register(self, label):
        register(self.values, f"{label}_key")
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

        callback = get_callback(self.values)
        idx = generic_concat([self.values, other.values], ordered=False)
        return Index(callback(unique(idx)))

    def _merge_all(self, idx_list):
        idx = self.values
        callback = get_callback(idx)

        for other in idx_list:
            self._check_types(other)
            idx = generic_concat([idx, other.values], ordered=False)

        return Index(callback(unique(idx)))

    def _check_aligned(self, other):
        self._check_types(other)
        length = len(self)
        return len(other) == length and (self == other.values).sum() == length

    def argsort(self, ascending=True):
        if not ascending:
            if isinstance(self.values, pdarray) and self.dtype in (akint64, akfloat64):
                i = argsort(-self.values)
            else:
                i = argsort(self.values)[arange(self.size - 1, -1, -1)]
        else:
            i = argsort(self.values)
        return i

    def concat(self, other):
        self._check_types(other)

        idx = generic_concat([self.values, other.values], ordered=True)
        return Index(idx)

    def lookup(self, key):
        if not isinstance(key, pdarray):
            raise TypeError("Lookup must be on an arkouda array")

        return in1d(self.values, key)

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        return self.values.to_hdf(prefix_path, dataset=dataset, mode=mode, file_type=file_type)

    def to_parquet(
        self, prefix_path: str, dataset: str = "index", mode: str = "truncate", compressed: bool = False
    ):
        return self.values.to_parquet(prefix_path, dataset=dataset, mode=mode, compressed=compressed)

    def save(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        compressed: bool = False,
        file_format: str = "HDF5",
        file_type: str = "distribute",
    ) -> str:
        """
        DEPRECATED
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
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.

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

        Any file extension can be used. The file I/O does not rely on the extension to determine the
        file format.
        """
        warn(
            "ak.Index.save has been deprecated. Please use ak.Index.to_parquet or ak.Index.to_hdf",
            DeprecationWarning,
        )
        from arkouda.io import file_type_to_int, mode_str_to_int

        if file_format.lower() == "hdf5":
            return typecast(
                str,
                generic_msg(
                    cmd="tohdf",
                    args={
                        "values": self.values,
                        "dset": dataset,
                        "write_mode": mode_str_to_int(mode),
                        "filename": prefix_path,
                        "dtype": self.dtype,
                        "objType": "pdarray",
                        "file_format": file_type_to_int(file_type),
                    },
                ),
            )
        elif file_format.lower() == "parquet":
            return typecast(
                str,
                generic_msg(
                    cmd="writeParquet",
                    args={
                        "values": self.values,
                        "dset": dataset,
                        "mode": mode_str_to_int(mode),
                        "prefix": prefix_path,
                        "dtype": self.dtype,
                        "save_offsets": False,  # only used by strings
                        "compressed": compressed,
                    },
                ),
            )
        else:
            raise ValueError("Supported file formats are 'HDF5' and 'Parquet'")


class MultiIndex(Index):
    def __init__(self, values):
        if not (isinstance(values, list) or isinstance(values, tuple)):
            raise TypeError("MultiIndex should be an iterable")
        self.values = values
        first = True
        for col in self.values:
            if first:
                # we are implicitly assuming values contains arkouda types and not python lists
                # because we are using obj.size/obj.dtype instead of len(obj)/type(obj)
                # this should be made explict using typechecking
                self.size = col.size
                first = False
            else:
                if col.size != self.size:
                    raise ValueError("All columns in MultiIndex must have same length")
        self.levels = len(self.values)

    def __getitem__(self, key):
        from arkouda.series import Series

        if type(key) == Series:
            key = key.values
        return MultiIndex([i[key] for i in self.index])

    def __repr__(self):
        return f"MultiIndex({repr(self.index)})"

    def __len__(self):
        return len(self.index[0])

    def __eq__(self, v):
        if not isinstance(v, (list, tuple, MultiIndex)):
            raise TypeError("Cannot compare MultiIndex to a scalar")
        retval = ones(len(self), dtype=akbool)
        if isinstance(v, MultiIndex):
            v = v.index
        for a, b in zip(self.index, v):
            retval &= a == b

        return retval

    @property
    def index(self):
        return self.values

    def to_pandas(self):
        idx = [convert_if_categorical(i) for i in self.index]
        mi = [i.to_ndarray() for i in idx]
        return pd.Series(index=mi, dtype="float64").index

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = [dtype(i) for i in self.index]
        self.index = new_idx
        return self

    def register(self, label):
        for i, arr in enumerate(self.index):
            register(arr, f"{label}_key_{i}")
        return len(self.index)

    def to_dict(self, labels):
        data = {}
        if labels is None:
            labels = [f"idx_{i}" for i in range(len(self.index))]
        for i, value in enumerate(self.index):
            data[labels[i]] = value
        return data

    def _merge(self, other):
        self._check_types(other)
        idx = [generic_concat([ix1, ix2], ordered=False) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(GroupBy(idx).unique_keys)

    def _merge_all(self, array):
        idx = self.index

        for other in array:
            self._check_types(other)
            idx = [generic_concat([ix1, ix2], ordered=False) for ix1, ix2 in zip(idx, other.index)]

        return MultiIndex(GroupBy(idx).unique_keys)

    def argsort(self, ascending=True):
        i = coargsort(self.index)
        if not ascending:
            i = i[arange(self.size - 1, -1, -1)]
        return i

    def concat(self, other):
        self._check_types(other)
        idx = [generic_concat([ix1, ix2], ordered=True) for ix1, ix2 in zip(self.index, other.index)]
        return MultiIndex(idx)

    def lookup(self, key):
        if type(key) != list and type(key) != tuple:
            raise TypeError("MultiIndex lookup failure")

        return in1d(self.index, key)
