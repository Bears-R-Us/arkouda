from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd  # type: ignore
from numpy import array as ndarray
from typeguard import typechecked

from arkouda import Categorical, Strings
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import float64 as akfloat64
from arkouda.dtypes import int64 as akint64
from arkouda.groupbyclass import GroupBy, unique
from arkouda.numeric import cast as akcast
from arkouda.pdarrayclass import RegistrationError, pdarray
from arkouda.pdarraycreation import arange, array, create_pdarray, ones
from arkouda.pdarraysetops import argsort, in1d
from arkouda.sorting import coargsort
from arkouda.util import convert_if_categorical, generic_concat, get_callback

if TYPE_CHECKING:
    from arkouda.series import Series


class Index:
    objType = "Index"
    """
    Sequence used for indexing and alignment.

    The basic object storing axis labels for all DataFrame objects.

    Parameters
    ----------
    values: List, pdarray, Strings, Categorical, pandas.Index, or Index
    name : str, default=None
        Name to be stored in the index.
    allow_list = False,
        If False, list values will be converted to a pdarray.
        If True, list values will remain as a list, provided the data length is less than max_list_size.
    max_list_size = 1000
        This is the maximum allowed data length for the values to be stored as a list object.

    Raises
    ------
    ValueError
        Raised if allow_list=True and the size of values is > max_list_size.

    See Also
    --------
    MultiIndex

    Examples
    --------
    >>> ak.Index([1, 2, 3])
    Index(array([1 2 3]), dtype='int64')

    >>> ak.Index(list('abc'))
    Index(array(['a', 'b', 'c']), dtype='<U0')

    >>> ak.Index([1, 2, 3], allow_list=True)
    Index([1, 2, 3], dtype='int64')

    """

    @typechecked
    def __init__(
        self,
        values: Union[List, pdarray, Strings, Categorical, pd.Index, "Index"],
        name: Optional[str] = None,
        allow_list=False,
        max_list_size=1000,
    ):
        self.max_list_size = max_list_size
        self.registered_name: Optional[str] = None
        if isinstance(values, Index):
            self.values = values.values
            self.size = values.size
            self.dtype = values.dtype
            self.name = name if name else values.name
        elif isinstance(values, pd.Index):
            self.values = array(values.values)
            self.size = values.size
            self.dtype = self.values.dtype
            self.name = name if name else values.name
        elif isinstance(values, List):
            if allow_list is True:
                if len(values) <= max_list_size:
                    self.values = values
                    self.size = len(values)
                    self.dtype = self._dtype_of_list_values(values)
                else:
                    raise ValueError(
                        f"Cannot create Index because list size {len(values)} "
                        f"exceeds max_list_size {self.max_list_size}."
                    )
            else:
                values = array(values)
                self.values = values
                self.size = self.values.size
                self.dtype = self.values.dtype
            self.name = name
        elif isinstance(values, (pdarray, Strings, Categorical)):
            self.values = values
            self.size = self.values.size
            self.dtype = self.values.dtype
            self.name = name
        else:
            raise TypeError(f"Unable to create Index from type {type(values)}")

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

    def _dtype_of_list_values(self, lst):
        from arkouda.dtypes import dtype

        if isinstance(lst, list):
            d = dtype(type(lst[0]))
            for item in lst:
                assert dtype(type(item)) == d, (
                    f"Values of Index must all be same type.  "
                    f"Types {d} and {dtype(type(item))} do not match."
                )
            return d
        else:
            raise TypeError("Index Types must match")

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
        if isinstance(self.values, list):
            return len(set(self.values)) == self.size
        else:
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

    @classmethod
    def from_return_msg(cls, rep_msg):
        data = json.loads(rep_msg)

        idx = []
        for d in data:
            i_comps = d.split("+|+")
            if i_comps[0].lower() == pdarray.objType.lower():
                idx.append(create_pdarray(i_comps[1]))
            elif i_comps[0].lower() == Strings.objType.lower():
                idx.append(Strings.from_return_msg(i_comps[1]))
            elif i_comps[0].lower() == Categorical.objType.lower():
                idx.append(Categorical.from_return_msg(i_comps[1]))

        return cls.factory(idx) if len(idx) > 1 else cls.factory(idx[0])

    def memory_usage(self, unit="B"):
        """
        Return the memory usage of the Index values.

        Parameters
        ----------
        unit : str, default = "B"
            Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        arkouda.pdarrayclass.nbytes
        arkouda.index.MultiIndex.memory_usage
        arkouda.series.Series.memory_usage
        arkouda.dataframe.DataFrame.memory_usage

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> idx = Index(ak.array([1, 2, 3]))
        >>> idx.memory_usage()
        24

        """
        from arkouda.util import convert_bytes

        return convert_bytes(self.values.nbytes, unit=unit)

    def to_pandas(self):
        if isinstance(self.values, list):
            val = ndarray(self.values)
        else:
            val = convert_if_categorical(self.values).to_ndarray()
        return pd.Index(data=val, dtype=val.dtype, name=self.name)

    def to_ndarray(self):
        if isinstance(self.values, list):
            return ndarray(self.values)
        else:
            val = convert_if_categorical(self.values)
            return val.to_ndarray()

    def to_list(self):
        if isinstance(self.values, list):
            return self.values
        else:
            return self.to_ndarray().tolist()

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = dtype(self.values)
        self.values = new_idx
        return self

    def register(self, user_defined_name):
        """
        Register this Index object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Index is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Index
            The same Index which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Indexes with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Index with the user_defined_name

        See also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        if isinstance(self.values, list):
            raise TypeError("Index cannot be registered when values are list type.")

        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "num_idxs": 1,
                "idx_names": [
                    json.dumps(
                        {
                            "codes": self.values.codes.name,
                            "categories": self.values.categories.name,
                            "NA_codes": self.values._akNAcode.name,
                            **(
                                {"permutation": self.values.permutation.name}
                                if self.values.permutation is not None
                                else {}
                            ),
                            **(
                                {"segments": self.values.segments.name}
                                if self.values.segments is not None
                                else {}
                            ),
                        }
                    )
                    if isinstance(self.values, Categorical)
                    else self.values.name
                ],
                "idx_types": [self.values.objType],
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this Index object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self):
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            if not isinstance(self.values, Categorical):
                return is_registered(self.values.name, as_component=True)
            else:
                result = True
                result &= is_registered(self.values.codes.name, as_component=True)
                result &= is_registered(self.values.categories.name, as_component=True)
                result &= is_registered(self.values._akNAcode.name, as_component=True)
                if self.values.permutation is not None and self.values.segments is not None:
                    result &= is_registered(self.values.permutation.name, as_component=True)
                    result &= is_registered(self.values.segments.name, as_component=True)
                return result
        else:
            return is_registered(self.registered_name)

    def to_dict(self, label):
        data = {}
        if label is None:
            label = "idx"
        elif isinstance(label, list):
            label = label[0]
        data[label] = self.index
        return data

    def _check_types(self, other):
        if type(self) is not type(other):
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
        if isinstance(self.values, list):
            reverse = not ascending
            return sorted(range(self.size), key=self.values.__getitem__, reverse=reverse)

        if not ascending:
            if isinstance(self.values, pdarray) and self.dtype in (akint64, akfloat64):
                i = argsort(-self.values)
            else:
                i = argsort(self.values)[arange(self.size - 1, -1, -1)]
        else:
            i = argsort(self.values)
        return i

    def map(self, arg: Union[dict, "Series"]) -> "Index":
        """
        Map values of Index according to an input mapping.

        Parameters
        ----------
        arg : dict or Series
            The mapping correspondence.

        Returns
        -------
        arkouda.index.Index
            A new index with the values transformed by the mapping correspondence.

        Raises
        ------
        TypeError
            Raised if arg is not of type dict or arkouda.Series.
            Raised if index values not of type pdarray, Categorical, or Strings.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> idx = ak.Index(ak.array([2, 3, 2, 3, 4]))
        >>> display(idx)
        Index(array([2 3 2 3 4]), dtype='int64')
        >>> idx.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})
        Index(array([30.00000000000000000 5.00000000000000000 30.00000000000000000
        5.00000000000000000 25.00000000000000000]), dtype='float64')
        >>> s2 = ak.Series(ak.array(["a","b","c","d"]), index = ak.array([4,2,1,3]))
        >>> idx.map(s2)
        Index(array(['b', 'b', 'd', 'd', 'a']), dtype='<U0')

        """
        from arkouda.util import map

        return Index(map(self.values, arg))

    def concat(self, other):
        self._check_types(other)

        idx = generic_concat([self.values, other.values], ordered=True)
        return Index(idx)

    def lookup(self, key):
        if not isinstance(key, pdarray):
            # try to handle single value
            try:
                key = array([key])
            except Exception:
                raise TypeError("Lookup must be on an arkouda array")

        return in1d(self.values, key)

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        """
        Save the Index to HDF5.
        The object can be saved to a collection of files or single file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        TypeError
            Raised if the Index values are a list.
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`. Otherwise,
        the file name will be `prefix_path`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        from typing import cast as typecast

        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        if isinstance(self.values, list):
            raise TypeError("Unable to write Index to hdf when values are a list.")

        index_data = [
            self.values.name
            if not isinstance(self.values, (Categorical_))
            else json.dumps(
                {
                    "codes": self.values.codes.name,
                    "categories": self.values.categories.name,
                    "NA_codes": self.values._akNAcode.name,
                    **(
                        {"permutation": self.values.permutation.name}
                        if self.values.permutation is not None
                        else {}
                    ),
                    **(
                        {"segments": self.values.segments.name}
                        if self.values.segments is not None
                        else {}
                    ),
                }
            )
        ]
        return typecast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "filename": prefix_path,
                    "dset": dataset,
                    "file_format": _file_type_to_int(file_type),
                    "write_mode": _mode_str_to_int(mode),
                    "objType": self.objType,
                    "num_idx": 1,
                    "idx": index_data,
                    "idx_objTypes": [self.values.objType],  # this will be pdarray, strings, or cat
                    "idx_dtypes": [str(self.values.dtype)],
                },
            ),
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        repack: bool = True,
    ):
        """
        Overwrite the dataset with the name provided with this Index object. If
        the dataset does not exist it is added.

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the index

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, this will create a copy of the
          file with the new data
        """
        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        index_data = [
            self.values.name
            if not isinstance(self.values, (Categorical_))
            else json.dumps(
                {
                    "codes": self.values.codes.name,
                    "categories": self.values.categories.name,
                    "NA_codes": self.values._akNAcode.name,
                    **(
                        {"permutation": self.values.permutation.name}
                        if self.values.permutation is not None
                        else {}
                    ),
                    **(
                        {"segments": self.values.segments.name}
                        if self.values.segments is not None
                        else {}
                    ),
                }
            )
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "filename": prefix_path,
                "dset": dataset,
                "file_format": _file_type_to_int(file_type),
                "write_mode": _mode_str_to_int("append"),
                "objType": self.objType,
                "num_idx": 1,
                "idx": index_data,
                "idx_objTypes": [self.values.objType],  # this will be pdarray, strings, or cat
                "idx_dtypes": [str(self.values.dtype)],
                "overwrite": True,
            },
        ),

        if repack:
            _repack_hdf(prefix_path)

    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        compression: Optional[str] = None,
    ):
        """
        Save the Index to Parquet. The result is a collection of files,
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
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
        Returns
        -------
        string message indicating result of save operation
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        TypeError
            Raised if the Index values are a list.
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`.
        - 'append' write mode is supported, but is not efficient.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        if isinstance(self.values, list):
            raise TypeError("Unable to write Index to parquet when values are a list.")

        return self.values.to_parquet(prefix_path, dataset=dataset, mode=mode, compression=compression)

    @typechecked
    def to_csv(
        self,
        prefix_path: str,
        dataset: str = "index",
        col_delim: str = ",",
        overwrite: bool = False,
    ):
        """
        Write Index to CSV file(s). File will contain a single column with the pdarray data.
        All CSV Files written by Arkouda include a header denoting data types of the columns.

        Parameters
        -----------
        prefix_path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        dataset: str
            Column name to save the pdarray under. Defaults to "array".
        col_delim: str
            Defaults to ",". Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool
            Defaults to False. If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        --------
        str reponse message

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist.
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server.
            Raised if the Index values are a list.

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (`\n`) at this time.
        """
        if isinstance(self.values, list):
            raise TypeError("Unable to write Index to csv when values are a list.")

        return self.values.to_csv(prefix_path, dataset=dataset, col_delim=col_delim, overwrite=overwrite)

    def save(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        compression: Optional[str] = None,
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
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
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
            is not a string.
            Raised if the Index values are a list.
        See Also
        --------
        save_all, load, read, to_parquet, to_hdf
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
        from warnings import warn

        warn(
            "ak.Index.save has been deprecated. Please use ak.Index.to_parquet or ak.Index.to_hdf",
            DeprecationWarning,
        )

        if isinstance(self.values, list):
            raise TypeError("Unable to save Index when values are a list.")

        if mode.lower() not in ["append", "truncate"]:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        if file_format.lower() == "hdf5":
            return self.to_hdf(prefix_path, dataset=dataset, mode=mode, file_type=file_type)
        elif file_format.lower() == "parquet":
            return self.to_parquet(prefix_path, dataset=dataset, mode=mode, compression=compression)
        else:
            raise ValueError("Valid file types are HDF5 or Parquet")


class MultiIndex(Index):
    objType = "MultiIndex"

    def __init__(self, values, name=None, names=None):
        self.registered_name: Optional[str] = None
        if not (isinstance(values, list) or isinstance(values, tuple)):
            raise TypeError("MultiIndex should be an iterable")
        self.values = values
        first = True
        self.names = names
        self.name = name
        for col in self.values:
            # col can be a python int which doesn't have a size attribute
            col_size = col.size if not isinstance(col, int) else 0
            if first:
                # we are implicitly assuming values contains arkouda types and not python lists
                # because we are using obj.size/obj.dtype instead of len(obj)/type(obj)
                # this should be made explict using typechecking
                self.size = col_size
                first = False
            else:
                if col_size != self.size:
                    raise ValueError("All columns in MultiIndex must have same length")
        self.levels = len(self.values)

    def __getitem__(self, key):
        from arkouda.series import Series

        if isinstance(key, Series):
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

    def memory_usage(self, unit="B"):
        """
        Return the memory usage of the MultiIndex values.

        Parameters
        ----------
        unit : str, default = "B"
            Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        arkouda.pdarrayclass.nbytes
        arkouda.index.Index.memory_usage
        arkouda.series.Series.memory_usage
        arkouda.dataframe.DataFrame.memory_usage

        Examples
        --------

        >>> import arkouda as ak
        >>> ak.connect()
        >>> m = ak.index.MultiIndex([ak.array([1,2,3]),ak.array([4,5,6])])
        >>> m.memory_usage()
        48

        """
        from arkouda.util import convert_bytes

        nbytes = 0
        for item in self.values:
            nbytes += item.nbytes

        return convert_bytes(nbytes, unit=unit)

    def to_pandas(self):
        idx = [convert_if_categorical(i) for i in self.index]
        mi = pd.MultiIndex.from_arrays([i.to_ndarray() for i in idx], names=self.names)
        return pd.Series(index=mi, dtype="float64", name=self.name).index

    def set_dtype(self, dtype):
        """Change the data type of the index

        Currently only aku.ip_address and ak.array are supported.
        """
        new_idx = [dtype(i) for i in self.index]
        self.index = new_idx
        return self

    def to_ndarray(self):
        return ndarray([convert_if_categorical(val).to_ndarray() for val in self.values])

    def to_list(self):
        return self.to_ndarray().tolist()

    def register(self, user_defined_name):
        """
        Register this Index object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Index is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        MultiIndex
            The same Index which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Indexes with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Index with the user_defined_name

        See also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "num_idxs": len(self.values),
                "idx_names": [
                    json.dumps(
                        {
                            "codes": v.codes.name,
                            "categories": v.categories.name,
                            "NA_codes": v._akNAcode.name,
                            **({"permutation": v.permutation.name} if v.permutation is not None else {}),
                            **({"segments": v.segments.name} if v.segments is not None else {}),
                        }
                    )
                    if isinstance(v, Categorical)
                    else v.name
                    for v in self.values
                ],
                "idx_types": [v.objType for v in self.values],
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        from arkouda.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self):
        from arkouda.util import is_registered

        if self.registered_name is None:
            return False
        return is_registered(self.registered_name)

    def to_dict(self, labels=None):
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
        if not isinstance(key, list) and not isinstance(key, tuple):
            raise TypeError("MultiIndex lookup failure")
        # if individual vals convert to pdarrays
        if not isinstance(key[0], pdarray):
            dt = self.values[0].dtype if isinstance(self.values[0], pdarray) else akint64
            key = [akcast(array([x]), dt) for x in key]

        return in1d(self.index, key)

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        """
        Save the Index to HDF5.
        The object can be saved to a collection of files or single file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray.

        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`. Otherwise,
        the file name will be `prefix_path`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        from typing import cast as typecast

        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        index_data = [
            obj.name
            if not isinstance(obj, (Categorical_))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            for obj in self.values
        ]
        return typecast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "filename": prefix_path,
                    "dset": dataset,
                    "file_format": _file_type_to_int(file_type),
                    "write_mode": _mode_str_to_int(mode),
                    "objType": self.objType,
                    "num_idx": len(self.values),
                    "idx": index_data,
                    "idx_objTypes": [obj.objType for obj in self.values],
                    "idx_dtypes": [str(obj.dtype) for obj in self.values],
                },
            ),
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "index",
        repack: bool = True,
    ):
        """
        Overwrite the dataset with the name provided with this Index object. If
        the dataset does not exist it is added.

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the index
        TypeError
            Raised if the Index values are a list.

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, this will create a copy of the
          file with the new data
        """
        from arkouda.categorical import Categorical as Categorical_
        from arkouda.client import generic_msg
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        if isinstance(self.values, list):
            raise TypeError("Unable update hdf when Index values are a list.")

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        index_data = [
            obj.name
            if not isinstance(obj, (Categorical_))
            else json.dumps(
                {
                    "codes": obj.codes.name,
                    "categories": obj.categories.name,
                    "NA_codes": obj._akNAcode.name,
                    **({"permutation": obj.permutation.name} if obj.permutation is not None else {}),
                    **({"segments": obj.segments.name} if obj.segments is not None else {}),
                }
            )
            for obj in self.values
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "filename": prefix_path,
                "dset": dataset,
                "file_format": _file_type_to_int(file_type),
                "write_mode": _mode_str_to_int("append"),
                "objType": self.objType,
                "num_idx": len(self.values),
                "idx": index_data,
                "idx_objTypes": [obj.objType for obj in self.values],
                "idx_dtypes": [str(obj.dtype) for obj in self.values],
                "overwrite": True,
            },
        ),

        if repack:
            _repack_hdf(prefix_path)
