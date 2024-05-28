import glob
import json
import os
from typing import Dict, List, Mapping, Optional, Union, cast
from warnings import warn

import pandas as pd
from typeguard import typechecked

from arkouda.array_view import ArrayView
from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.client_dtypes import IPv4
from arkouda.dataframe import DataFrame
from arkouda.groupbyclass import GroupBy
from arkouda.index import Index, MultiIndex
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import arange, array
from arkouda.segarray import SegArray
from arkouda.strings import Strings
from arkouda.timeclass import Datetime, Timedelta

__all__ = [
    "get_filetype",
    "ls",
    "ls_csv",
    "get_null_indices",
    "get_datasets",
    "get_columns",
    "read_hdf",
    "read_parquet",
    "read_csv",
    "read",
    "read_tagged_data",
    "import_data",
    "export",
    "to_hdf",
    "to_parquet",
    "to_csv",
    "save_all",
    "load",
    "load_all",
    "update_hdf",
    "snapshot",
    "restore",
    "receive",
    "receive_dataframe",
]

ARKOUDA_HDF5_FILE_METADATA_GROUP = "_arkouda_metadata"


def get_filetype(filenames: Union[str, List[str]]) -> str:
    """
    Get the type of a file accessible to the server. Supported
    file types and possible return strings are 'HDF5' and 'Parquet'.

    Parameters
    ----------
    filenames : Union[str, List[str]]
        A file or list of files visible to the arkouda server

    Returns
    -------
    str
        Type of the file returned as a string, either 'HDF5', 'Parquet' or 'CSV

    Raises
    ------
    ValueError
        Raised if filename is empty or contains only whitespace

    Notes
    -----
    - When list provided, it is assumed that all files are the same type
    - CSV Files without the Arkouda Header are not supported

    See Also
    --------
    read_parquet, read_hdf
    """
    if isinstance(filenames, list):
        fname = filenames[0]
    else:
        fname = filenames
    if not (fname and fname.strip()):
        raise ValueError("filename cannot be an empty string")

    return cast(str, generic_msg(cmd="getfiletype", args={"filename": fname}))


def ls(filename: str, col_delim: str = ",", read_nested: bool = True) -> List[str]:
    """
    This function calls the h5ls utility on a HDF5 file visible to the
    arkouda server or calls a function that imitates the result of h5ls
    on a Parquet file.

    Parameters
    ----------
    filename : str
        The name of the file to pass to the server
    col_delim : str
        The delimiter used to separate columns if the file is a csv
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Only used for Parquet files.

    Returns
    -------
    str
        The string output of the datasets from the server

    Raises
    ------
    TypeError
        Raised if filename is not a str
    ValueError
        Raised if filename is empty or contains only whitespace
    RuntimeError
        Raised if error occurs in executing ls on an HDF5 file
    Notes
        - This will need to be updated because Parquet will not technically support this when we update.
            Similar functionality will be added for Parquet in the future
        - For CSV files without headers, please use ls_csv

    See Also
    ---------
    ls_csv
    """
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    cmd = "lsany"
    return json.loads(
        cast(
            str,
            generic_msg(
                cmd=cmd,
                args={"filename": filename, "col_delim": col_delim, "read_nested": read_nested},
            ),
        )
    )


def get_null_indices(
    filenames: Union[str, List[str]], datasets: Optional[Union[str, List[str]]] = None
) -> Union[pdarray, Mapping[str, pdarray]]:
    """
    Get null indices of a string column in a Parquet file.

    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read. Each dataset must be a string
        column. There is no default value for this function, the datasets to be
        read must be specified.

    Returns
    -------
    returns a dictionary of Arkouda pdarrays
        Dictionary of {datasetName: pdarray}

    Raises
    ------
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    See Also
    --------
    get_datasets, ls
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(datasets, str):
        datasets = [datasets]
    rep_msg = generic_msg(
        cmd="getnullparquet",
        args={
            "dset_size": len(datasets) if datasets is not None else 0,  # if needed for mypy
            "filename_size": len(filenames),
            "dsets": datasets,
            "filenames": filenames,
        },
    )
    rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllMsgJson for json structure
    # ignore the type here because we are returning a specific case
    return _build_objects(rep)  # type: ignore


@typechecked
def _file_type_to_int(file_type: str) -> int:
    """
    Convert a string to integer representing the format to save the file in

    Parameters
    ----------
    file_type: str (single | distribute)
        The string representation of the format for saving the file

    Returns
    -------
    int representing the format

    Raises
    ------
    ValueError
        If mode is not 'single' or 'distribute'
    """
    if file_type.lower() == "single":
        return 0
    elif file_type.lower() == "distribute":
        return 1
    else:
        raise ValueError(f"File Type expected to be 'single' or 'distributed'. Got {file_type}")


@typechecked
def _mode_str_to_int(mode: str) -> int:
    """
    Convert string to integer representing the mode to write

    Parameters
    ----------
    mode: str (truncate | append)
        The string representation of the write mode to be converted to integer

    Returns
    -------
    int representing the mode

    Raises
    ------
    ValueError
        If mode is not 'truncate' or 'append'
    """
    if mode.lower() == "truncate":
        return 0
    elif mode.lower() == "append":
        return 1
    else:
        raise ValueError(f"Write Mode expected to be 'truncate' or 'append'. Got {mode}.")


def get_datasets(
    filenames: Union[str, List[str]],
    allow_errors: bool = False,
    column_delim: str = ",",
    read_nested: bool = True,
) -> List[str]:
    """
    Get the names of the datasets in the provide files

    Parameters
    ----------
    filenames: str or List[str]
        Name of the file/s from which to return datasets
    allow_errors: bool
        Default: False
        Whether or not to allow errors while accessing datasets
    column_delim : str
        Column delimiter to be used if dataset is CSV. Otherwise, unused.
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Only used for Parquet Files.


    Returns
    -------
    List[str] of names of the datasets

    Raises
    ------
    RuntimeError
        - If no datasets are returned

    Notes
    -----
    - This function currently supports HDF5 and Parquet formats.
    - Future updates to Parquet will deprecate this functionality on that format,
    but similar support will be added for Parquet at that time.
    - If a list of files is provided, only the datasets in the first file will be returned

    See Also
    --------
    ls
    """
    datasets = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for fname in filenames:
        try:
            datasets = ls(fname, col_delim=column_delim, read_nested=read_nested)
            if datasets:
                break
        except RuntimeError:
            if allow_errors:
                pass
            else:
                raise

    if not datasets:  # empty
        raise RuntimeError("Unable to identify datasets.")
    return datasets


def ls_csv(filename: str, col_delim: str = ",") -> List[str]:
    """
    Used for identifying the datasets within a file when a CSV does not
    have a header.

    Parameters
    ----------
    filename : str
        The name of the file to pass to the server
    col_delim : str
        The delimiter used to separate columns if the file is a csv

    Returns
    -------
    str
        The string output of the datasets from the server

    See Also
    ---------
    ls
    """
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    return json.loads(
        cast(
            str,
            generic_msg(
                cmd="lscsv",
                args={"filename": filename, "col_delim": col_delim},
            ),
        )
    )


def get_columns(
    filenames: Union[str, List[str]], col_delim: str = ",", allow_errors: bool = False
) -> List[str]:
    """
    Get a list of column names from CSV file(s).
    """
    datasets = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for fname in filenames:
        try:
            datasets = ls_csv(fname, col_delim)
            if datasets:
                break
        except RuntimeError:
            if allow_errors:
                pass
            else:
                raise

    if not datasets:  # empty
        raise RuntimeError("Unable to identify datasets.")
    return datasets


def _prep_datasets(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    allow_errors: bool = False,
    read_nested: bool = True,
) -> List[str]:
    """
    Prepare a list of datasets to be read

    Parameters
    ----------
    filenames: str or List[str]
        Names of the files for which datasets are being prepped.
        Used to call get_datasets()
    datasets: Optional str or List[str]
        datasets to be accessed
    allow_errors: bool
        Default: False
        Whether or not to allow errors during access operations
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Only used for Parquet Files

    Returns
    -------
    List[str] of dataset names to access

    Raises
    ------
    ValueError
        - If one or more datasets cannot be found
    """
    if datasets is None:
        # get datasets. We know they exist because we pulled from the file
        datasets = get_datasets(filenames, allow_errors, read_nested=read_nested)
    else:
        if isinstance(datasets, str):
            # TODO - revisit this and enable checks that support things like "strings/values"
            # old logic did not check existence for single string dataset.
            return [datasets]
        # ensure dataset(s) exist
        # read_nested always true because when user supplies datasets, it is ignored
        nonexistent = set(datasets) - set(get_datasets(filenames, allow_errors, read_nested=True))
        if len(nonexistent) > 0:
            raise ValueError(f"Dataset(s) not found: {nonexistent}")
    return datasets


def _parse_errors(rep_msg, allow_errors: bool = False):
    """
    Helper function to parse error messages from a read operation

    Parameters
    ----------
    rep_msg
        The server response from a read operation
    allow_errors: bool
        Default: False
        Whether or not errors are to be allowed during read operation
    """
    file_errors = rep_msg["file_errors"] if "file_errors" in rep_msg else []
    if allow_errors and file_errors:
        file_error_count = rep_msg["file_error_count"] if "file_error_count" in rep_msg else -1
        warn(
            f"There were {file_error_count} errors reading files on the server. "
            + f"Sample error messages {file_errors}",
            RuntimeWarning,
        )


def _parse_obj(
    obj: Dict,
) -> Union[
    Strings,
    pdarray,
    ArrayView,
    SegArray,
    Categorical,
    DataFrame,
    IPv4,
    Datetime,
    Timedelta,
    Index,
    MultiIndex,
]:
    """
    Helper function to create an Arkouda object from read response

    Parameters
    ----------
    obj : Dict
        The response data used to create an Arkouda object

    Returns
    -------
    Strings, pdarray, or ArrayView Arkouda object

    Raises
    ------
    TypeError
        - If return object is an unsupported type
    """
    if Strings.objType.upper() == obj["arkouda_type"]:
        return Strings.from_return_msg(obj["created"])
    elif SegArray.objType.upper() == obj["arkouda_type"]:
        return SegArray.from_return_msg(obj["created"])
    elif pdarray.objType.upper() == obj["arkouda_type"]:
        return create_pdarray(obj["created"])
    elif IPv4.special_objType.upper() == obj["arkouda_type"]:
        return IPv4(create_pdarray(obj["created"]))
    elif Datetime.special_objType.upper() == obj["arkouda_type"]:
        return Datetime(create_pdarray(obj["created"]))
    elif Timedelta.special_objType.upper() == obj["arkouda_type"]:
        return Timedelta(create_pdarray(obj["created"]))
    elif ArrayView.objType.upper() == obj["arkouda_type"]:
        components = obj["created"].split("+")
        flat = create_pdarray(components[0])
        shape = create_pdarray(components[1])
        return ArrayView(flat, shape)
    elif Categorical.objType.upper() == obj["arkouda_type"]:
        return Categorical.from_return_msg(obj["created"])
    elif GroupBy.objType.upper() == obj["arkouda_type"]:
        return GroupBy.from_return_msg(obj["created"])
    elif DataFrame.objType.upper() == obj["arkouda_type"]:
        return DataFrame.from_return_msg(obj["created"])
    elif (
        obj["arkouda_type"].lower() == Index.objType.lower()
        or obj["arkouda_type"].lower() == MultiIndex.objType.lower()
    ):
        return Index.from_return_msg(obj["created"])
    else:
        raise TypeError(f"Unknown arkouda type:{obj['arkouda_type']}")


def _dict_recombine_segarrays_categoricals(df_dict):
    # this assumes segments will always have corresponding values.
    # This should happen due to save config
    seg_cols = ["_".join(col.split("_")[:-1]) for col in df_dict.keys() if col.endswith("_segments")]
    cat_cols = [".".join(col.split(".")[:-1]) for col in df_dict.keys() if col.endswith(".categories")]
    df_dict_keys = {
        "_".join(col.split("_")[:-1])
        if col.endswith("_segments") or col.endswith("_values")
        else ".".join(col.split(".")[:-1])
        if col.endswith("._akNAcode")
        or col.endswith(".categories")
        or col.endswith(".codes")
        or col.endswith(".permutation")
        or col.endswith(".segments")
        else col
        for col in df_dict.keys()
    }

    # update dict to contain segarrays where applicable if any exist
    if len(seg_cols) > 0 or len(cat_cols) > 0:
        df_dict = {
            col: SegArray(df_dict[col + "_segments"], df_dict[col + "_values"])
            if col in seg_cols
            else Categorical.from_codes(
                df_dict[f"{col}.codes"],
                df_dict[f"{col}.categories"],
                permutation=df_dict[f"{col}.permutation"]
                if f"{col}.permutation" in df_dict_keys
                else None,
                segments=df_dict[f"{col}.segments"] if f"{col}.segments" in df_dict_keys else None,
                _akNAcode=df_dict[f"{col}._akNAcode"],
            )
            if col in cat_cols
            else df_dict[col]
            for col in df_dict_keys
        }
    return df_dict


def _build_objects(
    rep_msg: Dict,
) -> Union[
    Mapping[
        str,
        Union[
            Strings,
            pdarray,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Helper function to create the Arkouda objects from a read operation

    Parameters
    ----------
    rep_msg: Dict
        rep_msg to create objects from

    Returns
    -------
    Strings, pdarray, or ArrayView Arkouda object or Dictionary mapping the dataset name to the object

    Raises
    ------
    RuntimeError
        - If no objects were returned
    """
    items = json.loads(rep_msg["items"]) if "items" in rep_msg else []
    if len(items) >= 1:
        return _dict_recombine_segarrays_categoricals(
            {item["dataset_name"]: _parse_obj(item) for item in items}
        )
    else:
        raise RuntimeError("No items were returned")


def read_hdf(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strict_types: bool = True,
    allow_errors: bool = False,
    calc_string_offsets: bool = False,
    tag_data=False,
) -> Union[
    Mapping[
        str,
        Union[
            pdarray,
            Strings,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Read Arkouda objects from HDF5 file/s

    Parameters
    ----------
    filenames : str, List[str]
        Filename/s to read objects from
    datasets : Optional str, List[str]
        datasets to read from the provided files
    iterative : bool
        Iterative (True) or Single (False) function call(s) to server
    strict_types: bool
        If True (default), require all dtypes of a given dataset to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one
        file contains a uint32 dataset and another contains an int64
        dataset with the same name, the contents of both will be read
        into an int64 pdarray.
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.
    calc_string_offsets: bool
        Default False, if True this will tell the server to calculate the
        offsets/segments array on the server versus loading them from HDF5 files.
        In the future this option may be set to True as the default.
    tagData: bool
        Default False, if True tag the data with the code associated with the filename
        that the data was pulled from.

    Returns
    -------
    Returns a dictionary of Arkouda pdarrays, Arkouda Strings, Arkouda Segarrays,
     or Arkouda ArrayViews.
        Dictionary of {datasetName: pdarray, String, SegArray, or ArrayView}

    Raises
    ------
    ValueError
        Raised if all datasets are not present in all hdf5 files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files.

    If iterative == True each dataset name and file names are passed to
    the server as independent sequential strings while if iterative == False
    all dataset names and file names are passed to the server in a single
    string.

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets
    to HDF5 files.

    See Also
    ---------
    read_tagged_data

    Examples
    --------
    >>>
    # Read with file Extension
    >>> x = ak.read_hdf('path/name_prefix.h5') # load HDF5
    # Read Glob Expression
    >>> x = ak.read_hdf('path/name_prefix*') # Reads HDF5
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = _prep_datasets(filenames, datasets, allow_errors)

    if iterative:
        if tag_data:
            raise RuntimeError("Cannot tag data with iterative read.")
        return {
            dset: read_hdf(
                filenames,
                datasets=dset,
                strict_types=strict_types,
                allow_errors=allow_errors,
                calc_string_offsets=calc_string_offsets,
                tag_data=tag_data,
            )[dset]
            for dset in datasets
        }
    else:
        rep_msg = generic_msg(
            cmd="readAllHdf",
            args={
                "strict_types": strict_types,
                "dset_size": len(datasets),
                "filename_size": len(filenames),
                "allow_errors": allow_errors,
                "calc_string_offsets": calc_string_offsets,
                "dsets": datasets,
                "filenames": filenames,
                "tag_data": tag_data,
            },
        )
        rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllMsgJson for json structure
        _parse_errors(rep, allow_errors)
        return _build_objects(rep)


def read_parquet(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strict_types: bool = True,
    allow_errors: bool = False,
    tag_data: bool = False,
    read_nested: bool = True,
    has_non_float_nulls: bool = False,
) -> Union[
    Mapping[
        str,
        Union[
            pdarray,
            Strings,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Read Arkouda objects from Parquet file/s

    Parameters
    ----------
    filenames : str, List[str]
        Filename/s to read objects from
    datasets : Optional str, List[str]
        datasets to read from the provided files
    iterative : bool
        Iterative (True) or Single (False) function call(s) to server
    strict_types: bool
        If True (default), require all dtypes of a given dataset to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one
        file contains a uint32 dataset and another contains an int64
        dataset with the same name, the contents of both will be read
        into an int64 pdarray.
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.
    tagData: bool
        Default False, if True tag the data with the code associated with the filename
        that the data was pulled from.
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        If datasets is not None, this will be ignored.
    has_non_float_nulls: bool
        Default False. This flag must be set to True to read non-float parquet columns
        that contain null values.

    Returns
    -------
    Returns a dictionary of Arkouda pdarrays, Arkouda Strings, Arkouda Segarrays,
     or Arkouda ArrayViews.
        Dictionary of {datasetName: pdarray, String, SegArray, or ArrayView}

    Raises
    ------
    ValueError
        Raised if all datasets are not present in all parquet files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files.

    If iterative == True each dataset name and file names are passed to
    the server as independent sequential strings while if iterative == False
    all dataset names and file names are passed to the server in a single
    string.

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets
    to Parquet files.

    Parquet always recomputes offsets at this time
    This will need to be updated once parquets workflow is updated

    See Also
    ---------
    read_tagged_data

    Examples
    --------
    Read without file Extension
    >>> x = ak.read_parquet('path/name_prefix.parquet') # load Parquet
    Read Glob Expression
    >>> x = ak.read_parquet('path/name_prefix*') # Reads Parquet
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = _prep_datasets(filenames, datasets, read_nested=read_nested)

    if iterative:
        if tag_data:
            raise RuntimeError("Cannot tag data with iterative read.")
        return {
            dset: read_parquet(
                filenames,
                datasets=dset,
                strict_types=strict_types,
                allow_errors=allow_errors,
                tag_data=tag_data,
                read_nested=read_nested,
                has_non_float_nulls=has_non_float_nulls,
            )[dset]
            for dset in datasets
        }
    else:
        rep_msg = generic_msg(
            cmd="readAllParquet",
            args={
                "strict_types": strict_types,
                "dset_size": len(datasets),
                "filename_size": len(filenames),
                "allow_errors": allow_errors,
                "dsets": datasets,
                "filenames": filenames,
                "tag_data": tag_data,
                "has_non_float_nulls": has_non_float_nulls,
            },
        )
        rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllMsgJson for json structure
        _parse_errors(rep, allow_errors)
        return _build_objects(rep)


def read_csv(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    column_delim: str = ",",
    allow_errors: bool = False,
) -> Union[
    Mapping[
        str,
        Union[
            pdarray,
            Strings,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Read CSV file(s) into Arkouda objects. If more than one dataset is found, the objects
    will be returned in a dictionary mapping the dataset name to the Arkouda object
    containing the data. If the file contains the appropriately formatted header, typed
    data will be returned. Otherwise, all data will be returned as a Strings object.

    Parameters
    -----------
    filenames: str or List[str]
        The filenames to read data from
    datasets: str or List[str] (Optional)
        names of the datasets to read. When `None`, all datasets will be read.
    column_delim: str
        The delimiter for column names and data. Defaults to ",".
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.

    Returns
    --------
    Returns a dictionary of Arkouda pdarrays, Arkouda Strings, Arkouda Segarrays,
     or Arkouda ArrayViews.
        Dictionary of {datasetName: pdarray, String, SegArray, or ArrayView}

    Raises
    ------
    ValueError
        Raised if all datasets are not present in all parquet files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    See Also
    ---------
    to_csv

    Notes
    ------
    - CSV format is not currently supported by load/load_all operations
    - The column delimiter is expected to be the same for column names and data
    - Be sure that column delimiters are not found within your data.
    - All CSV files must delimit rows using newline (``\\n``) at this time.
    - Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
      bytes as uint(8).
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    if isinstance(datasets, str):
        datasets = [datasets]
    elif datasets is None:
        datasets = get_columns(filenames, col_delim=column_delim, allow_errors=allow_errors)

    rep_msg = generic_msg(
        cmd="readcsv",
        args={
            "filenames": filenames,
            "nfiles": len(filenames),
            "datasets": datasets,
            "num_dsets": len(datasets),
            "col_delim": column_delim,
            "allow_errors": allow_errors,
        },
    )
    rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllMsgJson for json structure
    _parse_errors(rep, allow_errors)
    return _build_objects(rep)


def import_data(
    read_path: str, write_file: Optional[str] = None, return_obj: bool = True, index: bool = False
):
    """
    Import data from a file saved by Pandas (HDF5/Parquet) to Arkouda object and/or
    a file formatted to be read by Arkouda.

    Parameters
    __________
    read_path: str
        path to file where pandas data is stored. This can be glob expression for parquet formats.
    write_file: str, optional
        path to file to write arkouda formatted data to. Only write file if provided
    return_obj: bool, optional
        Default True. When True return the Arkouda DataFrame object, otherwise return None
    index: bool, optional
        Default False. When True, maintain the indexes loaded from the pandas file

    Raises
    ______
    RuntimeWarning
        - Export attempted on Parquet file. Arkouda formatted Parquet files are readable by pandas.
    RuntimeError
        - Unsupported file type

    Returns
    _______
    pd.DataFrame
        When `return_obj=True`

    See Also
    ________
    pandas.DataFrame.to_parquet, pandas.DataFrame.to_hdf,
    pandas.DataFrame.read_parquet, pandas.DataFrame.read_hdf,
    ak.export

    Notes
    _____
    - Import can only be performed from hdf5 or parquet files written by pandas.
    """
    from arkouda.dataframe import DataFrame

    # verify file path
    is_glob = not os.path.isfile(read_path)
    file_list = glob.glob(read_path)
    if len(file_list) == 0:
        raise FileNotFoundError(f"Invalid read_path, {read_path}. No files found.")

    # access the file type - multiple files valid here because parquet supports glob. Check first listed.
    file = read_path if not is_glob else glob.glob(read_path)[0]
    filetype = get_filetype(file)
    # Note - in the future if we support more than pandas here, we should verify attributes.
    if filetype == "HDF5":
        if is_glob:
            raise RuntimeError(
                "Pandas HDF5 import supports valid file path only. Only supports the local file system,"
                " remote URLs and file-like objects are not supported."
            )
        df_def = pd.read_hdf(read_path)
    elif filetype == "Parquet":
        # parquet supports glob input in pandas
        df_def = pd.read_parquet(read_path)
    else:
        raise RuntimeError(
            "File type not supported. Import is only supported for HDF5 and Parquet file formats."
        )
    df = DataFrame(df_def)

    if write_file:
        df.to_hdf(write_file, index=index) if filetype == "HDF5" else df.to_parquet(
            write_file, index=index
        )

    if return_obj:
        return df


def export(
    read_path: str,
    dataset_name: str = "ak_data",
    write_file: Optional[str] = None,
    return_obj: bool = True,
    index: bool = False,
):
    """
    Export data from Arkouda file (Parquet/HDF5) to Pandas object or file formatted to be
    readable by Pandas

    Parameters
    __________
    read_path: str
        path to file where arkouda data is stored.
    dataset_name: str
        name to store dataset under
    index: bool
        Default False. When True, maintain the indexes loaded from the pandas file
    write_file: str, optional
        path to file to write pandas formatted data to. Only write the file if this is set
    return_obj: bool, optional
        Default True. When True return the Pandas DataFrame object, otherwise return None


    Raises
    ______
    RuntimeError
        - Unsupported file type

    Returns
    _______
    pd.DataFrame
        When `return_obj=True`

    See Also
    ________
    pandas.DataFrame.to_parquet, pandas.DataFrame.to_hdf,
    pandas.DataFrame.read_parquet, pandas.DataFrame.read_hdf,
    ak.import_data

    Notes
    _____
    - If Arkouda file is exported for pandas, the format will not change. This mean parquet files
      will remain parquet and hdf5 will remain hdf5.
    - Export can only be performed from hdf5 or parquet files written by Arkouda. The result will be
      the same file type, but formatted to be read by Pandas.
    """
    from arkouda.dataframe import DataFrame

    # get the filetype
    prefix, extension = os.path.splitext(read_path)
    first_file = f"{prefix}_LOCALE0000{extension}"
    filetype = get_filetype(first_file)

    if filetype not in ["HDF5", "Parquet"]:
        raise RuntimeError(
            "File type not supported. Import is only supported for HDF5 and Parquet file formats."
        )

    akdf = DataFrame.load(read_path, file_format=filetype)
    df = akdf.to_pandas(retain_index=index)

    if write_file:
        if filetype == "HDF5":
            # write to fixed format as this should be the most efficient
            df.to_hdf(write_file, key=dataset_name, format="fixed", mode="w", index=index)
        else:
            # we know this is parquet because otherwise we would have errored at the type check
            df.to_parquet(write_file, index=index)

    if return_obj:
        return df


def _bulk_write_prep(
    columns: Union[
        Mapping[str, Union[pdarray, Strings, SegArray, ArrayView]],
        List[Union[pdarray, Strings, SegArray, ArrayView]],
    ],
    names: Optional[List[str]] = None,
    convert_categoricals: bool = False,
):
    datasetNames = []
    if names is not None:
        if len(names) != len(columns):
            raise ValueError("Number of names does not match number of columns")
        else:
            datasetNames = names

    data = []  # init to avoid undefined errors
    if isinstance(columns, dict):
        data = list(columns.values())
        if names is None:
            datasetNames = list(columns.keys())
    elif isinstance(columns, list):
        data = cast(List[pdarray], columns)
        if names is None:
            datasetNames = [str(column) for column in range(len(columns))]

    if len(data) == 0:
        raise RuntimeError("No data was found.")

    if convert_categoricals:
        for i, val in enumerate(data):
            if isinstance(val, Categorical):
                data[i] = val.categories[val.codes]

    col_objtypes = [c.objType for c in data]

    return datasetNames, data, col_objtypes


def to_parquet(
    columns: Union[
        Mapping[str, Union[pdarray, Strings, SegArray, ArrayView]],
        List[Union[pdarray, Strings, SegArray, ArrayView]],
    ],
    prefix_path: str,
    names: Optional[List[str]] = None,
    mode: str = "truncate",
    compression: Optional[str] = None,
    convert_categoricals: bool = False,
) -> None:
    """
    Save multiple named pdarrays to Parquet files.

    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    prefix_path : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist.
        If 'append', attempt to create new dataset in existing files.
        'append' is deprecated, please use the multi-column write
    compression : str (Optional)
            Default None
            Provide the compression type to use when writing the file.
            Supported values: snappy, gzip, brotli, zstd, lz4
        convert_categoricals: bool
            Defaults to False
            Parquet requires all columns to be the same size and Categoricals
            don't satisfy that requirement.
            if set, write the equivalent Strings in place of any Categorical columns.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised if (1) the lengths of columns and values differ or (2) the mode
        is not 'truncate' or 'append'
    RuntimeError
            Raised if a server-side error is thrown saving the pdarray

    See Also
    --------
    to_hdf, load, load_all, read

    Notes
    -----
    Creates one file per locale containing that locale's chunk of each pdarray.
    If columns is a dictionary, the keys are used as the Parquet column names.
    Otherwise, if no names are supplied, 0-up integers are used. By default,
    any existing files at path_prefix will be overwritten, unless the user
    specifies the 'append' mode, in which case arkouda will attempt to add
    <columns> as new datasets to existing files. If the wrong number of files
    is present or dataset names already exist, a RuntimeError is raised.

    Examples
    --------
    >>> a = ak.arange(25)
    >>> b = ak.arange(25)

    >>> # Save with mapping defining dataset names
    >>> ak.to_parquet({'a': a, 'b': b}, 'path/name_prefix')

    >>> # Save using names instead of mapping
    >>> ak.to_parquet([a, b], 'path/name_prefix', names=['a', 'b'])
    """
    if mode.lower() not in ["append", "truncate"]:
        raise ValueError("Allowed modes are 'truncate' and 'append'")
    if mode.lower() == "append":
        warn(
            "Append has been deprecated when writing Parquet files. "
            "Please write all columns to the file at once.",
            DeprecationWarning,
        )

    datasetNames, data, col_objtypes = _bulk_write_prep(columns, names, convert_categoricals)
    # append or single column use the old logic
    if mode.lower() == "append" or len(data) == 1:
        for arr, name in zip(data, cast(List[str], datasetNames)):
            arr.to_parquet(prefix_path=prefix_path, dataset=name, mode=mode, compression=compression)
    else:
        print(
            cast(
                str,
                generic_msg(
                    cmd="toParquet_multi",
                    args={
                        "columns": data,
                        "col_names": datasetNames,
                        "col_objtypes": col_objtypes,
                        "filename": prefix_path,
                        "num_cols": len(data),
                        "compression": compression,
                    },
                ),
            )
        )


def to_hdf(
    columns: Union[
        Mapping[str, Union[pdarray, Strings, SegArray, ArrayView]],
        List[Union[pdarray, Strings, SegArray, ArrayView]],
    ],
    prefix_path: str,
    names: Optional[List[str]] = None,
    mode: str = "truncate",
    file_type: str = "distribute",
) -> None:
    """
    Save multiple named pdarrays to HDF5 files.

    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    prefix_path : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist.
        If 'append', attempt to create new dataset in existing files.
    file_type : str ("single" | "distribute")
            Default: distribute
            Single writes the dataset to a single file
            Distribute writes the dataset to a file per locale

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised if (1) the lengths of columns and values differ or (2) the mode
        is not 'truncate' or 'append'
    RuntimeError
            Raised if a server-side error is thrown saving the pdarray

    See Also
    --------
    to_parquet, load, load_all, read

    Notes
    -----
    Creates one file per locale containing that locale's chunk of each pdarray.
    If columns is a dictionary, the keys are used as the HDF5 dataset names.
    Otherwise, if no names are supplied, 0-up integers are used. By default,
    any existing files at path_prefix will be overwritten, unless the user
    specifies the 'append' mode, in which case arkouda will attempt to add
    <columns> as new datasets to existing files. If the wrong number of files
    is present or dataset names already exist, a RuntimeError is raised.

    Examples
    --------
    >>> a = ak.arange(25)
    >>> b = ak.arange(25)

    >>> # Save with mapping defining dataset names
    >>> ak.to_hdf({'a': a, 'b': b}, 'path/name_prefix')

    >>> # Save using names instead of mapping
    >>> ak.to_hdf([a, b], 'path/name_prefix', names=['a', 'b'])
    """
    if mode.lower() not in ["append", "truncate"]:
        raise ValueError("Allowed modes are 'truncate' and 'append'")

    datasetNames, pdarrays, _ = _bulk_write_prep(columns, names)

    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        arr.to_hdf(
            prefix_path=prefix_path,
            dataset=name,
            mode=mode,
            file_type=file_type,
        )
        if mode.lower() == "truncate":
            mode = "append"


def _get_hdf_filetype(filename: str) -> str:
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    cmd = "hdffileformat"
    return cast(
        str,
        generic_msg(
            cmd=cmd,
            args={"filename": filename},
        ),
    )


def _repack_hdf(prefix_path: str):
    """
    Overwrites the existing hdf5 file with a copy that removes any inaccessible datasets
    """
    file_type = _get_hdf_filetype(prefix_path + "*")
    dset_list = ls(prefix_path + "*")
    if len(dset_list) == 1:
        # early out because when overwriting only one value, hdf5 automatically releases memory
        return
    data = read_hdf(prefix_path + "*")
    if not isinstance(data, dict):
        # handles the case of reading only 1 dataset
        data = [data]  # type: ignore
    to_hdf(data, prefix_path, names=dset_list, file_type=file_type)  # type: ignore


def update_hdf(
    columns: Union[
        Mapping[str, Union[pdarray, Strings, SegArray, ArrayView]],
        List[Union[pdarray, Strings, SegArray, ArrayView]],
    ],
    prefix_path: str,
    names: Optional[List[str]] = None,
    repack: bool = True,
):
    """
    Overwrite the datasets with name appearing in names or keys in columns if columns
    is a dictionary

    Parameters
    -----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    prefix_path : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    repack: bool
        Default: True
        HDF5 does not release memory on delete. When True, the inaccessible
        data (that was overwritten) is removed. When False, the data remains, but is
        inaccessible. Setting to false will yield better performance, but will cause
        file sizes to expand.

    Raises
    -------
    RuntimeError
        Raised if a server-side error is thrown saving the datasets

    Notes
    -----
    - If file does not contain File_Format attribute to indicate how it was saved,
      the file name is checked for _LOCALE#### to determine if it is distributed.
    - If the datasets provided do not exist, they will be added
    - Because HDF5 deletes do not release memory, this will create a copy of the
      file with the new data
    - This workflow is slightly different from `to_hdf` to prevent reading and
      creating a copy of the file for each dataset
    """
    datasetNames, pdarrays, _ = _bulk_write_prep(columns, names)

    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        # overwrite the data without repacking. Repack done once at end if set
        arr.update_hdf(prefix_path, dataset=name, repack=False)

    if repack:
        _repack_hdf(prefix_path)


def to_csv(
    columns: Union[Mapping[str, Union[pdarray, Strings]], List[Union[pdarray, Strings]]],
    prefix_path: str,
    names: Optional[List[str]] = None,
    col_delim: str = ",",
    overwrite: bool = False,
):
    """
    Write Arkouda object(s) to CSV file(s). All CSV Files written by Arkouda
    include a header denoting data types of the columns.

    Parameters
    -----------
    columns: Mapping[str, pdarray] or List[pdarray]
        The objects to be written to CSV file. If a mapping is used and `names` is None
        the keys of the mapping will be used as the dataset names.
    prefix_path: str
        The filename prefix to be used for saving files. Files will have _LOCALE#### appended
        when they are written to disk.
    names: List[str] (Optional)
        names of dataset to be written. Order should correspond to the order of data
        provided in `columns`.
    col_delim: str
        Defaults to ",". Value to be used to separate columns within the file.
        Please be sure that the value used DOES NOT appear in your dataset.
    overwrite: bool
        Defaults to False. If True, any existing files matching your provided prefix_path will
        be overwritten. If False, an error will be returned if existing files are found.

    Returns
    --------
    None

    Raises
    ------
    ValueError
        Raised if any datasets are present in all csv files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    See Also
    ---------
    read_csv

    Notes
    ------
    - CSV format is not currently supported by load/load_all operations
    - The column delimiter is expected to be the same for column names and data
    - Be sure that column delimiters are not found within your data.
    - All CSV files must delimit rows using newline (``\\n``) at this time.
    - Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
      bytes as uint(8).
    """
    datasetNames, pdarrays, _ = _bulk_write_prep(columns, names)  # type: ignore
    dtypes = [a.dtype.name for a in pdarrays]

    generic_msg(
        cmd="writecsv",
        args={
            "datasets": pdarrays,
            "col_names": datasetNames,
            "filename": prefix_path,
            "num_dsets": len(pdarrays),
            "col_delim": col_delim,
            "dtypes": dtypes,
            "row_count": pdarrays[0].size,  # all columns should have equal number of entries
            "overwrite": overwrite,
        },
    )


def save_all(
    columns: Union[
        Mapping[str, Union[pdarray, Strings, SegArray, ArrayView]],
        List[Union[pdarray, Strings, SegArray, ArrayView]],
    ],
    prefix_path: str,
    names: Optional[List[str]] = None,
    file_format="HDF5",
    mode: str = "truncate",
    file_type: str = "distribute",
    compression: Optional[str] = None,
) -> None:
    """
    DEPRECATED
    Save multiple named pdarrays to HDF5/Parquet files.
    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    prefix_path : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    file_format : str
        'HDF5' or 'Parquet'. Defaults to hdf5
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist.
        If 'append', attempt to create new dataset in existing files.
    file_type : str ("single" | "distribute")
        Default: distribute
        Single writes the dataset to a single file
        Distribute writes the dataset to a file per locale
        Only used with HDF5
    compression: str (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
        Optional
        Select the compression to use with Parquet files.
        Only used with Parquet.

    Returns
    -------
    None
    Raises
    ------
    ValueError
        Raised if (1) the lengths of columns and values differ or (2) the mode
        is not 'truncate' or 'append'
    See Also
    --------
    save, load_all, to_parquet, to_hdf
    Notes
    -----
    Creates one file per locale containing that locale's chunk of each pdarray.
    If columns is a dictionary, the keys are used as the HDF5 dataset names.
    Otherwise, if no names are supplied, 0-up integers are used. By default,
    any existing files at path_prefix will be overwritten, unless the user
    specifies the 'append' mode, in which case arkouda will attempt to add
    <columns> as new datasets to existing files. If the wrong number of files
    is present or dataset names already exist, a RuntimeError is raised.
    Examples
    --------
    >>> a = ak.arange(25)
    >>> b = ak.arange(25)
    >>> # Save with mapping defining dataset names
    >>> ak.save_all({'a': a, 'b': b}, 'path/name_prefix', file_format='Parquet')
    >>> # Save using names instead of mapping
    >>> ak.save_all([a, b], 'path/name_prefix', names=['a', 'b'], file_format='Parquet')
    """
    warn(
        "ak.save_all has been deprecated. Please use ak.to_hdf or ak.to_parquet",
        DeprecationWarning,
    )
    if file_format.lower() == "hdf5":
        to_hdf(columns, prefix_path, names=names, mode=mode, file_type=file_type)
    elif file_format.lower() == "parquet":
        to_parquet(columns, prefix_path, names=names, mode=mode, compression=compression)
    else:
        raise ValueError("Arkouda only supports HDF5 and Parquet files.")


@typechecked
def load(
    path_prefix: str,
    file_format: str = "INFER",
    dataset: str = "array",
    calc_string_offsets: bool = False,
    column_delim: str = ",",
) -> Union[
    Mapping[
        str,
        Union[
            pdarray,
            Strings,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Load a pdarray previously saved with ``pdarray.save()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    file_format : str
        'INFER', 'HDF5' or 'Parquet'. Defaults to 'INFER'. Used to indicate the file type being loaded.
        If INFER, this will be detected during processing
    dataset : str
        Dataset name where the pdarray was saved, defaults to 'array'
    calc_string_offsets : bool
        If True the server will ignore Segmented Strings 'offsets' array and derive
        it from the null-byte terminators.  Defaults to False currently
    column_delim : str
        Column delimiter to be used if dataset is CSV. Otherwise, unused.

    Returns
    -------
    Mapping[str, Union[pdarray, Strings, SegArray, Categorical]]
        Dictionary of {datsetName: Union[pdarray, Strings, SegArray, Categorical]}
        with the previously saved pdarrays, Strings, SegArrays, or Categoricals

    Raises
    ------
    TypeError
        Raised if either path_prefix or dataset is not a str
    ValueError
        Raised if invalid file_format or if the dataset is not present in all hdf5 files or if the
        path_prefix does not correspond to files accessible to Arkouda
    RuntimeError
        Raised if the hdf5 files are present but there is an error in opening
        one or more of them

    See Also
    --------
    to_parquet, to_hdf, load_all, read

    Notes
    -----
    If you have a previously saved Parquet file that is raising a FileNotFound error, try loading it
    with a .parquet appended to the prefix_path.
    Parquet files were previously ALWAYS stored with a ``.parquet`` extension.

    ak.load does not support loading a single file.
    For loading single HDF5 files without the _LOCALE#### suffix please use ak.read().

    CSV files without the Arkouda Header are not supported.

    Examples
    --------
    >>> # Loading from file without extension
    >>> obj = ak.load('path/prefix')
    Loads the array from numLocales files with the name ``cwd/path/name_prefix_LOCALE####``.
    The file type is inferred during processing.

    >>> # Loading with an extension (HDF5)
    >>> obj = ak.load('path/prefix.test')
    Loads the object from numLocales files with the name ``cwd/path/name_prefix_LOCALE####.test`` where
    #### is replaced by each locale numbers. Because filetype is inferred during processing,
    the extension is not required to be a specific format.
    """
    if "*" in path_prefix:
        raise ValueError(
            "Glob expressions not supported by ak.load(). "
            "To read files using a glob expression, please use ak.read()"
        )
    prefix, extension = os.path.splitext(path_prefix)
    globstr = f"{prefix}_LOCALE*{extension}"
    try:
        file_format = get_filetype(globstr) if file_format.lower() == "infer" else file_format
        if file_format.lower() == "hdf5":
            return read_hdf(globstr, dataset, calc_string_offsets=calc_string_offsets)
        elif file_format.lower() == "parquet":
            return read_parquet(globstr, dataset)
        else:
            return read_csv(globstr, dataset, column_delim=column_delim)
    except RuntimeError as re:
        if "does not exist" in str(re):
            raise ValueError(
                f"There are no files corresponding to the path_prefix {path_prefix} in"
                " a location accessible to Arkouda"
            )
        else:
            raise RuntimeError(re)


@typechecked
def load_all(
    path_prefix: str, file_format: str = "INFER", column_delim: str = ",", read_nested=True
) -> Mapping[str, Union[pdarray, Strings, SegArray, Categorical]]:
    """
    Load multiple pdarrays, Strings, SegArrays, or Categoricals previously
    saved with ``save_all()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    file_format: str
        'INFER', 'HDF5', 'Parquet', or 'CSV'. Defaults to 'INFER'. Indicates the format being loaded.
        When 'INFER' the processing will detect the format
        Defaults to 'INFER'
    column_delim : str
        Column delimiter to be used if dataset is CSV. Otherwise, unused.
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Parquet files only

    Returns
    -------
    Mapping[str, Union[pdarray, Strings, SegArray, Categorical]]
        Dictionary of {datsetName: Union[pdarray, Strings, SegArray, Categorical]}
        with the previously saved pdarrays, Strings, SegArrays, or Categoricals


    Raises
    ------
    TypeError:
        Raised if path_prefix is not a str
    ValueError
        Raised if file_format/extension is encountered that is not hdf5 or parquet or
        if all datasets are not present in all hdf5/parquet files or if the
        path_prefix does not correspond to files accessible to Arkouda
    RuntimeError
        Raised if the hdf5 files are present but there is an error in opening
        one or more of them

    See Also
    --------
    to_parquet, to_hdf, load, read

    Notes
    _____
    This function has been updated to determine the file extension based on the file format variable

    This function will be deprecated when glob flags are added to read_* methods

    CSV files without the Arkouda Header are not supported.
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = f"{prefix}_LOCALE0000{extension}"
    try:
        result = dict()
        for dataset in get_datasets(firstname, column_delim=column_delim, read_nested=read_nested):
            result[dataset] = load(prefix, file_format=file_format, dataset=dataset)[dataset]

        result = _dict_recombine_segarrays_categoricals(result)
        # Check for Categoricals and remove if necessary
        removal_names, categoricals = Categorical.parse_hdf_categoricals(result)
        if removal_names:
            result.update(categoricals)
            for n in removal_names:
                result.pop(n)

        return result

    except RuntimeError as re:
        # enables backwards compatibility with previous naming convention
        if "does not exist" in str(re):
            try:
                firstname = f"{prefix}_LOCALE0{extension}"
                return {dataset: load(prefix, dataset=dataset) for dataset in get_datasets(firstname)}
            except RuntimeError as re:
                if "does not exist" in str(re):
                    raise ValueError(
                        f"There are no files corresponding to the path_prefix {prefix} and "
                        f"file_format {file_format} in location accessible to Arkouda"
                    )
                else:
                    raise RuntimeError(re)
        else:
            raise RuntimeError(
                f"Could not open one or more files with path_prefix {prefix} and "
                f"file_format {file_format} in location accessible to Arkouda"
            )


def read(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strictTypes: bool = True,
    allow_errors: bool = False,
    calc_string_offsets=False,
    column_delim: str = ",",
    read_nested: bool = True,
    has_non_float_nulls: bool = False,
) -> Union[
    Mapping[
        str,
        Union[
            pdarray,
            Strings,
            SegArray,
            ArrayView,
            Categorical,
            DataFrame,
            IPv4,
            Datetime,
            Timedelta,
            Index,
        ],
    ],
]:
    """
    Read datasets from files.
    File Type is determined automatically.

    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read (default: all available)
    iterative : bool
        Iterative (True) or Single (False) function call(s) to server
    strictTypes: bool
        If True (default), require all dtypes of a given dataset to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one
        file contains a uint32 dataset and another contains an int64
        dataset with the same name, the contents of both will be read
        into an int64 pdarray.
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.
    calc_string_offsets: bool
        Default False, if True this will tell the server to calculate the
        offsets/segments array on the server versus loading them from HDF5 files.
        In the future this option may be set to True as the default.
    column_delim : str
        Column delimiter to be used if dataset is CSV. Otherwise, unused.
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Ignored if datasets is not None
        Parquet Files only.
    has_non_float_nulls: bool
        Default False. This flag must be set to True to read non-float parquet columns
        that contain null values.

    Returns
    -------
    Returns a dictionary of Arkouda pdarrays, Arkouda Strings, Arkouda Segarrays,
     or Arkouda ArrayViews.
        Dictionary of {datasetName: pdarray, String, SegArray, or ArrayView}

    Raises
    ------
    RuntimeError
        If invalid filetype is detected

    See Also
    --------
    get_datasets, ls, read_parquet, read_hdf

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files.

    If iterative == True each dataset name and file names are passed to
    the server as independent sequential strings while if iterative == False
    all dataset names and file names are passed to the server in a single
    string.

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets
    to HDF5/Parquet files.

    CSV files without the Arkouda Header are not supported.

    Examples
    --------
    Read with file Extension
    >>> x = ak.read('path/name_prefix.h5') # load HDF5 - processing determines file type not extension
    Read without file Extension
    >>> x = ak.read('path/name_prefix.parquet') # load Parquet
    Read Glob Expression
    >>> x = ak.read('path/name_prefix*') # Reads HDF5
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    ftype = get_filetype(filenames)
    if ftype.lower() == "hdf5":
        return read_hdf(
            filenames,
            datasets=datasets,
            iterative=iterative,
            strict_types=strictTypes,
            allow_errors=allow_errors,
            calc_string_offsets=calc_string_offsets,
        )
    elif ftype.lower() == "parquet":
        return read_parquet(
            filenames,
            datasets=datasets,
            iterative=iterative,
            strict_types=strictTypes,
            allow_errors=allow_errors,
            read_nested=read_nested,
            has_non_float_nulls=has_non_float_nulls,
        )
    elif ftype.lower() == "csv":
        return read_csv(
            filenames, datasets=datasets, column_delim=column_delim, allow_errors=allow_errors
        )
    else:
        raise RuntimeError(f"Invalid File Type detected, {ftype}")


def read_tagged_data(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    strictTypes: bool = True,
    allow_errors: bool = False,
    calc_string_offsets=False,
    read_nested: bool = True,
    has_non_float_nulls: bool = False,
):
    """
    Read datasets from files and tag each record to the file it was read from.
    File Type is determined automatically.

    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read (default: all available)
    strictTypes: bool
        If True (default), require all dtypes of a given dataset to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one
        file contains a uint32 dataset and another contains an int64
        dataset with the same name, the contents of both will be read
        into an int64 pdarray.
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.
    calc_string_offsets: bool
        Default False, if True this will tell the server to calculate the
        offsets/segments array on the server versus loading them from HDF5 files.
        In the future this option may be set to True as the default.
    read_nested: bool
        Default True, when True, SegArray objects will be read from the file. When False,
        SegArray (or other nested Parquet columns) will be ignored.
        Ignored if datasets is not `None`
        Parquet Files only.
    has_non_float_nulls: bool
        Default False. This flag must be set to True to read non-float parquet columns
        that contain null values.

    Notes
    ------
    Not currently supported for Categorical or GroupBy datasets

    Examples
    ---------
    Read files and return data with tagging corresponding to the Categorical returned
    cat.codes will link the codes in data to the filename. Data will contain the code `Filename_Codes`
    >>> data, cat = ak.read_tagged_data('path/name')
    >>> data
    {'Filname_Codes': array([0 3 6 9 12]), 'col_name': array([0 0 0 1])}
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    # handle glob expansion
    j_str = generic_msg(
        cmd="globExpansion",
        args={"file_count": len(filenames), "filenames": filenames},
    )
    file_list = array(json.loads(j_str))
    file_cat = Categorical.from_codes(
        arange(file_list.size), file_list
    )  # create a categorical from the ak.Strings representation of the file list

    ftype = get_filetype(filenames)
    if ftype.lower() == "hdf5":
        return (
            read_hdf(
                filenames,
                datasets=datasets,
                iterative=False,
                strict_types=strictTypes,
                allow_errors=allow_errors,
                calc_string_offsets=calc_string_offsets,
                tag_data=True,
            ),
            file_cat,
        )
    elif ftype.lower() == "parquet":
        return (
            read_parquet(
                filenames,
                datasets=datasets,
                iterative=False,  # hard-coded because iterative not supported
                strict_types=strictTypes,
                allow_errors=allow_errors,
                tag_data=True,
                read_nested=read_nested,
                has_non_float_nulls=has_non_float_nulls,
            ),
            file_cat,
        )
    elif ftype.lower() == "csv":
        raise RuntimeError("CSV does not support tagging data with file name associated.")
    else:
        raise RuntimeError(f"Invalid File Type detected, {ftype}")


def snapshot(filename):
    """
    Create a snapshot of the current Arkouda namespace. All currently accessible variables containing
    Arkouda objects will be written to an HDF5 file.

    Unlike other save/load functions, this maintains the integrity of dataframes.

    Current Variable names are used as the dataset name when saving.

    Parameters
    ----------
    filename: str
    Name to use when storing file

    Returns
    --------
    None

    See Also
    ---------
    ak.restore
    """
    import inspect
    from types import ModuleType

    from arkouda.dataframe import DataFrame

    filename = filename + "_SNAPSHOT"
    mode = "TRUNCATE"
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for name, val in [
        (n, v) for n, v in callers_local_vars if not n.startswith("__") and not isinstance(v, ModuleType)
    ]:
        if isinstance(val, (pdarray, Categorical, SegArray, Strings, DataFrame, GroupBy)):
            if isinstance(val, DataFrame):
                val._to_hdf_snapshot(filename, dataset=name, mode=mode)
            else:
                val.to_hdf(filename, dataset=name, mode=mode)
            mode = "APPEND"


def restore(filename):
    """
    Return data saved using `ak.snapshot`

    Parameters
    ----------
    filename: str
    Name used to create snapshot to be read

    Returns
    --------
    Dict

    Notes
    ------
    Unlike other save/load methods using snapshot restore will save DataFrames alongside other
    objects in HDF5. Thus, they are returned within the dictionary as a dataframe.
    """
    restore_files = glob.glob(f"{filename}_SNAPSHOT_LOCALE*")
    return read_hdf(sorted(restore_files))


def receive(hostname: str, port):
    """
    Receive a pdarray sent by `pdarray.transfer()`.

    Parameters
    ----------
    hostname : str
        The hostname of the pdarray that sent the array
    port : int_scalars
        The port to send the array over. This needs to be an
        open port (i.e., not one that the Arkouda server is
        running on). This will open up `numLocales` ports,
        each of which in succession, so will use ports of the
        range {port..(port+numLocales)} (e.g., running an
        Arkouda server of 4 nodes, port 1234 is passed as
        `port`, Arkouda will use ports 1234, 1235, 1236,
        and 1237 to send the array data).
        This port much match the port passed to the call to
        `pdarray.transfer()`.

    Returns
    -------
    pdarray
        The pdarray sent from the sending server to the current
        receiving server.

    Raises
    ------
    ValueError
        Raised if the op is not within the pdarray.BinOps set
    TypeError
        Raised if other is not a pdarray or the pdarray.dtype is not
        a supported dtype
    """
    rep_msg = generic_msg(cmd="receiveArray", args={"hostname": hostname, "port": port})
    rep = json.loads(rep_msg)
    return _build_objects(rep)


def receive_dataframe(hostname: str, port):
    """
    Receive a pdarray sent by `dataframe.transfer()`.

    Parameters
    ----------
    hostname : str
        The hostname of the dataframe that sent the array
    port : int_scalars
        The port to send the dataframe over. This needs to be an
        open port (i.e., not one that the Arkouda server is
        running on). This will open up `numLocales` ports,
        each of which in succession, so will use ports of the
        range {port..(port+numLocales)} (e.g., running an
        Arkouda server of 4 nodes, port 1234 is passed as
        `port`, Arkouda will use ports 1234, 1235, 1236,
        and 1237 to send the array data).
        This port much match the port passed to the call to
        `pdarray.send_array()`.

    Returns
    -------
    pdarray
        The dataframe sent from the sending server to the
        current receiving server.

    Raises
    ------
    ValueError
        Raised if the op is not within the pdarray.BinOps set
    TypeError
        Raised if other is not a pdarray or the pdarray.dtype is not
        a supported dtype
    """
    rep_msg = generic_msg(cmd="receiveDataframe", args={"hostname": hostname, "port": port})
    rep = json.loads(rep_msg)
    return DataFrame(_build_objects(rep))
