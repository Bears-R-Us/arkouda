import glob
import json
import os
import warnings
from typing import Dict, List, Mapping, Optional, Union, cast
from warnings import warn

import pandas as pd  # type: ignore
from typeguard import typechecked

import arkouda.array_view
from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.strings import Strings

__all__ = [
    "get_filetype",
    "ls",
    "get_null_indices",
    "get_datasets",
    "read_hdf",
    "read_parquet",
    "import_data",
    "export",
    "to_hdf",
    "to_parquet",
    "load",
    "load_all",
    "read",
    "read_hdf5_multi_dim",
    "write_hdf5_multi_dim",
    "save_all",
    "file_type_to_int",
    "mode_str_to_int",
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
        Type of the file returned as a string, either 'HDF5' or 'Parquet'

    Raises
    ------
    ValueError
        Raised if filename is empty or contains only whitespace

    Notes
    -----
    When list provided, it is assumed that all files are the same type

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


def ls(filename: str) -> List[str]:
    """
    This function calls the h5ls utility on a HDF5 file visible to the
    arkouda server or calls a function that imitates the result of h5ls
    on a Parquet file.

    Parameters
    ----------
    filename : str
        The name of the file to pass to the server

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
    """
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    cmd = "lsany"
    return json.loads(
        cast(
            str,
            generic_msg(
                cmd=cmd,
                args={
                    "filename": filename,
                },
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
        column. There is no default value for this funciton, the datasets to be
        read must be specified.

    Returns
    -------
    For a single dataset returns an Arkouda pdarray and for multiple datasets
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
def file_type_to_int(file_type: str) -> int:
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
        - If mode is not 'single' or 'distribute'
    """
    if file_type.lower() == "single":
        return 0
    elif file_type.lower() == "distribute":
        return 1
    else:
        raise ValueError(f"File Type expected to be 'single' or 'distributed'. Got {file_type}")


@typechecked
def mode_str_to_int(mode: str) -> int:
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
        - If mode is not 'truncate' or 'append'
    """
    if mode.lower() == "truncate":
        return 0
    elif mode.lower() == "append":
        return 1
    else:
        raise ValueError(f"Write Mode expected to be 'truncate' or 'append'. Got {mode}.")


def get_datasets(filenames: Union[str, List[str]], allow_errors: bool = False) -> List[str]:
    """
    Get the names of the datasets in the provide files

    Parameters
    ----------
    filenames: str or List[str]
        Name of the file/s from which to return datasets
    allow_errors: bool
        Default: False
        Whether or not to allow errors while accessing datasets

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
            datasets = ls(fname)
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
        datasets = get_datasets(filenames, allow_errors)
    else:
        if isinstance(datasets, str):
            # TODO - revisit this and enable checks that support things like "strings/values"
            # old logic did not check existence for single string dataset.
            return [datasets]
        # ensure dataset(s) exist
        nonexistent = set(datasets) - set(get_datasets(filenames, allow_errors))
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
        warnings.warn(
            f"There were {file_error_count} errors reading files on the server. "
            + f"Sample error messages {file_errors}",
            RuntimeWarning,
        )


def _parse_obj(obj: Dict) -> Union[Strings, pdarray, arkouda.array_view.ArrayView]:
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
    if "seg_string" == obj["arkouda_type"]:
        return Strings.from_return_msg(obj["created"])
    elif "pdarray" == obj["arkouda_type"]:
        return create_pdarray(obj["created"])
    elif "ArrayView" == obj["arkouda_type"]:
        components = obj["created"].split("+")
        flat = create_pdarray(components[0])
        shape = create_pdarray(components[1])
        return arkouda.array_view.ArrayView(flat, shape)
    else:
        raise TypeError(f"Unknown arkouda type:{obj['arkouda_type']}")


def _build_objects(
    rep_msg: Dict,
) -> Union[
    Strings,
    pdarray,
    arkouda.array_view.ArrayView,
    Mapping[str, Union[Strings, pdarray, arkouda.array_view.ArrayView]],
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
    items = rep_msg["items"] if "items" in rep_msg else []
    # We have a couple possible return conditions
    # 1. We have multiple items returned i.e. multi pdarrays, multi strings, multi pdarrays & strings
    # 2. We have a single pdarray
    # 3. We have a single strings object
    if len(items) > 1:  # DataSets condition
        return {item["dataset_name"]: _parse_obj(item) for item in items}
    elif len(items) == 1:
        return _parse_obj(items[0])
    else:
        raise RuntimeError("No items were returned")


def read_hdf(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strict_types: bool = True,
    allow_errors: bool = False,
    calc_string_offsets: bool = False,
) -> Union[
    pdarray,
    Strings,
    arkouda.array_view.ArrayView,
    Mapping[str, Union[pdarray, Strings, arkouda.array_view.ArrayView]],
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

    Returns
    -------
    For a single dataset returns an Arkouda pdarray, Arkouda Strings, or Arkouda ArrayView object
    and for multiple datasets returns a dictionary of Arkouda pdarrays,
    Arkouda Strings or Arkouda ArrayView.
        Dictionary of {datasetName: pdarray or String}

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

    Examples
    --------
    Read with file Extension
    >>> x = ak.read_hdf('path/name_prefix.h5') # load HDF5
    Read Glob Expression
    >>> x = ak.read_hdf('path/name_prefix*') # Reads HDF5
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = _prep_datasets(filenames, datasets)

    if iterative:
        return {
            dset: read_hdf(
                filenames,
                datasets=dset,
                strict_types=strict_types,
                allow_errors=allow_errors,
                calc_string_offsets=calc_string_offsets,
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
) -> Union[
    pdarray,
    Strings,
    arkouda.array_view.ArrayView,
    Mapping[str, Union[pdarray, Strings, arkouda.array_view.ArrayView]],
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

        Returns
        -------
        For a single dataset returns an Arkouda pdarray, Arkouda Strings, or Arkouda ArrayView object
        and for multiple datasets returns a dictionary of Arkouda pdarrays,
        Arkouda Strings or Arkouda ArrayView.
            Dictionary of {datasetName: pdarray or String}

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

        Examples
        --------
        Read without file Extension
        >>> x = ak.read_parquet('path/name_prefix.parquet') # load Parquet
        Read Glob Expression
        >>> x = ak.read_parquet('path/name_prefix*') # Reads Parquet
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = _prep_datasets(filenames, datasets)

    if iterative:
        return {
            dset: read_parquet(
                filenames,
                datasets=dset,
                strict_types=strict_types,
                allow_errors=allow_errors,
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
            },
        )
        rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllMsgJson for json structure
        _parse_errors(rep, allow_errors)
        return _build_objects(rep)


def import_data(read_path: str, write_file: str = None, return_obj: bool = True, index: bool = False):
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
    is_glob = False if os.path.isfile(read_path) else True
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
        df.save(write_file, index=index, file_format=filetype)

    if return_obj:
        return df


def export(
    read_path: str,
    dataset_name: str = "ak_data",
    write_file: str = None,
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


def _bulk_write_prep(columns: Union[Mapping[str, pdarray], List[pdarray]], names: List[str] = None):
    datasetNames = []
    if names is not None:
        if len(names) != len(columns):
            raise ValueError("Number of names does not match number of columns")
        else:
            datasetNames = names
    if isinstance(columns, dict):
        pdarrays = list(columns.values())
        if names is None:
            datasetNames = list(columns.keys())
    elif isinstance(columns, list):
        pdarrays = cast(List[pdarray], columns)
        if names is None:
            datasetNames = [str(column) for column in range(len(columns))]

    return datasetNames, pdarrays


def to_parquet(
    columns: Union[Mapping[str, pdarray], List[pdarray]],
    prefix_path: str,
    names: List[str] = None,
    mode: str = "truncate",
    compressed: bool = False,
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

    See Also
    --------
    save, load_all

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

    datasetNames, pdarrays = _bulk_write_prep(columns, names)

    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        arr.to_parquet(prefix_path=prefix_path, dataset=name, mode=mode, compressed=compressed)
        if mode.lower() == "truncate":
            mode = "append"


def to_hdf(
    columns: Union[Mapping[str, pdarray], List[pdarray]],
    prefix_path: str,
    names: List[str] = None,
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

    See Also
    --------
    save, load_all

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

    datasetNames, pdarrays = _bulk_write_prep(columns, names)

    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        arr.to_hdf(
            prefix_path=prefix_path,
            dataset=name,
            mode=mode,
            file_type=file_type,
        )
        if mode.lower() == "truncate":
            mode = "append"


@typechecked
def load(
    path_prefix: str,
    file_format: str = "INFER",
    dataset: str = "array",
    calc_string_offsets: bool = False,
) -> Union[
    pdarray,
    Strings,
    arkouda.array_view.ArrayView,
    Mapping[str, Union[pdarray, Strings, arkouda.array_view.ArrayView]],
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

    Returns
    -------
    Union[pdarray, Strings]
        The pdarray or Strings that was previously saved

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
    save, load_all, read

    Notes
    -----
    If you have a previously saved Parquet file that is raising a FileNotFound error, try loading it
    with a .parquet appended to the prefix_path.
    Parquet files were previously ALWAYS stored with a ``.parquet`` extension.

    This function will be deprecated when glob flags are added to read_* functions

    Examples
    --------
    >>> # Loading from file without extension
    >>> obj = a.load('path/prefix')
    Loads the array from numLocales files with the name ``cwd/path/name_prefix_LOCALE####``.
    The file type is inferred during processing.

    >>> # Loading with an extension (HDF5)
    >>> obj = a.load('path/prefix.test')
    Loads the object from numLocales files with the name ``cwd/path/name_prefix_LOCALE####.test`` where
    #### is replaced by each locale numbers. Because filetype is inferred during processing,
    the extension is not required to be a specific format.
    """
    prefix, extension = os.path.splitext(path_prefix)
    globstr = f"{prefix}*{extension}"
    try:
        file_format = get_filetype(globstr) if file_format.lower() == "infer" else file_format
        if file_format.lower() == "hdf5":
            return read_hdf(globstr, dataset, calc_string_offsets=calc_string_offsets)
        else:
            return read_parquet(globstr, dataset)
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
    path_prefix: str, file_format: str = "INFER"
) -> Mapping[str, Union[pdarray, Strings, Categorical]]:
    """
    Load multiple pdarrays or Strings previously saved with ``save_all()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    file_format: str
        'INFER', 'HDF5' or 'Parquet'. Defaults to 'INFER'. Indicates the format being loaded.
        When 'INFER' the processing will detect the format
        Defaults to 'HDF5'

    Returns
    -------
    Mapping[str,pdarray]
        Dictionary of {datsetName: pdarray} with the previously saved pdarrays


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
    save_all, load, read

    Notes
    _____
    This function has been updated to determine the file extension based on the file format variable

    This function will be deprecated when glob flags are added to read_* methods
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = f"{prefix}_LOCALE0000{extension}"
    try:
        result = {
            dataset: load(prefix, file_format=file_format, dataset=dataset)
            for dataset in get_datasets(firstname)
        }

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


"""
    The following functionality will be deprecated in a later release
"""


def read(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strictTypes: bool = True,
    allow_errors: bool = False,
    calc_string_offsets=False,
    file_format: str = "infer",
) -> Union[
    pdarray,
    Strings,
    arkouda.array_view.ArrayView,
    Mapping[str, Union[pdarray, Strings, arkouda.array_view.ArrayView]],
]:
    """
    DEPRECATED
    Read datasets from HDF5 or Parquet files.

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
    file_format: str
        Default 'infer', if 'HDF5' or 'Parquet' (case insensitive), the file
        type checking will be skipped and will execute expecting all files in
        filenames to be of the specified type. Otherwise, will infer filetype
        based off of first file in filenames, expanded if a glob expression.

    Returns
    -------
    For a single dataset returns an Arkouda pdarray, Arkouda Strings, or Arkouda ArrayView object
    and for multiple datasets returns a dictionary of Arkouda pdarrays,
    Arkouda Strings or Arkouda ArrayView.
        Dictionary of {datasetName: pdarray or String}

    Raises
    ------
    ValueError
        Raised if all datasets are not present in all hdf5/parquet files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    See Also
    --------
    read, get_datasets, ls

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

    Examples
    --------
    Read with file Extension
    >>> x = ak.read('path/name_prefix.h5') # load HDF5 - processing determines file type not extension
    Read without file Extension
    >>> x = ak.read('path/name_prefix.parquet', file_format='Parquet') # load Parquet
    Read Glob Expression
    >>> x = ak.read('path/name_prefix*') # Reads HDF5
    """
    warn(
        "ak.pdarrayIO.read has been deprecated. Please use ak.IO.read_parquet or ak.IO.read_hdf",
        DeprecationWarning,
    )
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = _prep_datasets(filenames, datasets, allow_errors)

    file_format = file_format.lower()
    if file_format == "infer":
        cmd = "readany"
    elif file_format == "hdf5":
        cmd = "readAllHdf"
    elif file_format == "parquet":
        cmd = "readAllParquet"
    else:
        warnings.warn(f"Unrecognized file format string: {file_format}. Inferring file type")
        cmd = "readany"
    if iterative:  # iterative calls to server readhdf
        return {
            dset: read(
                filenames,
                dset,
                strictTypes=strictTypes,
                allow_errors=allow_errors,
                iterative=False,
                calc_string_offsets=calc_string_offsets,
            )[dset]
            for dset in datasets
        }
    else:
        rep_msg = generic_msg(
            cmd=cmd,
            args={
                "strict_types": strictTypes,
                "dset_size": len(datasets),
                "filename_size": len(filenames),
                "allow_errors": allow_errors,
                "calc_string_offsets": calc_string_offsets,
                "dsets": datasets,
                "filenames": filenames,
            },
        )
        rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllHdfMsgJson for json structure
        items = rep["items"] if "items" in rep else []
        file_errors = rep["file_errors"] if "file_errors" in rep else []
        if allow_errors and file_errors:
            file_error_count = rep["file_error_count"] if "file_error_count" in rep else -1
            warnings.warn(
                f"There were {file_error_count} errors reading files on the server. "
                + f"Sample error messages {file_errors}",
                RuntimeWarning,
            )

        # We have a couple possible return conditions
        # 1. We have multiple items returned i.e. multi pdarrays, multi strings, multi pdarrays & strings
        # 2. We have a single pdarray
        # 3. We have a single strings object
        if len(items) > 1:  # DataSets condition
            d: Dict[str, Union[pdarray, Strings, arkouda.array_view.ArrayView]] = {}
            for item in items:
                if "seg_string" == item["arkouda_type"]:
                    d[item["dataset_name"]] = Strings.from_return_msg(item["created"])
                elif "pdarray" == item["arkouda_type"]:
                    d[item["dataset_name"]] = create_pdarray(item["created"])
                elif "ArrayView" == item["arkouda_type"]:
                    objs = item["created"].split("+")
                    flat = create_pdarray(objs[0])
                    shape = create_pdarray(objs[1])

                    d[item["dataset_name"]] = arkouda.array_view.ArrayView(flat, shape)
                else:
                    raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
            return d
        elif len(items) == 1:
            item = items[0]
            if "pdarray" == item["arkouda_type"]:
                return create_pdarray(item["created"])
            elif "seg_string" == item["arkouda_type"]:
                return Strings.from_return_msg(item["created"])
            elif "ArrayView" == item["arkouda_type"]:
                objs = item["created"].split("+")
                flat = create_pdarray(objs[0])
                shape = create_pdarray(objs[1])

                return arkouda.array_view.ArrayView(flat, shape)
            else:
                raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
        else:
            raise RuntimeError("No items were returned")


@typechecked
def read_hdf5_multi_dim(file_path: str, dset: str) -> arkouda.array_view.ArrayView:
    """
    DEPRECATED
    This function is being mantained to allow reading from files written in Arkouda v2022.10.13
    or earlier. If used, save the object to update formatting.
    Read a multi-dimensional object from an HDF5 file

    Parameters
    ----------
    file_path: str
        path to the file to read from
    dset: str
        name of the dataset to read

    Returns
    -------
    ArrayView object representing the data read from file

    See Also
    --------
    ak.write_hdf5_multi_dim

    Notes
    -----
        - Error handling done on server to prevent multiple server calls
        - This is an initial implementation and updates will be coming soon
        - dset currently only reading a single dataset is supported
        - file_path will need to support list[str] and str for glob
        - Currently, order is always assumed to be row major
    """
    warn(
        "ak.pdarrayIO.read_hdf5_multi_dim has been deprecated. Please use ak.pdarrayIO.read",
        DeprecationWarning,
    )
    args = {"filename": file_path, "dset": dset}
    rep_msg = cast(
        str,
        generic_msg(
            cmd="readhdf_multi",
            args=args,
        ),
    )

    objs = rep_msg.split("+")

    shape = create_pdarray(objs[0])
    flat = create_pdarray(objs[1])

    arr = arkouda.array_view.ArrayView(flat, shape)
    return arr


@typechecked
def write_hdf5_multi_dim(
    obj: arkouda.array_view.ArrayView,
    file_path: str,
    dset: str,
    mode: str = "truncate",
    file_type: str = "distribute",
):
    """
    DEPRECATED
    Write a multi-dimensional ArrayView object to an HDF5 file

    Parameters
    ----------
    obj: ArrayView
        The object that will be written to the file
    file_path: str
        Path to the file to write the dataset to
    dset: str
        Name of the dataset to write
    mode: str (truncate | append)
        Default: truncate
        Mode to write the dataset in. Truncate will overwrite any existing files.
        Append will add the dataset to an existing file.
    file_type: str (single|distribute)
        Default: distribute
        Indicates the format to save the file. Single will store in a single file.
        Distribute will store the date in a file per locale.

    See Also
    --------
    ak.read_hdf5_multi_dim

    Notes
    -----
    - All ArrayView/Multi-Dimensional objects are stored as flattened arrays
    - If a file does not exist, it will be created regardless of the mode value
    - This function is currently standalone functionality for multi-dimensional datasets
    - Error handling done on server to prevent multiple server calls
    """
    from arkouda.io import file_type_to_int, mode_str_to_int

    warn(
        "ak.write_hdf5_multi_dim has been deprecated. Please use ak.ArrayView.to_hdf",
        DeprecationWarning,
    )
    args = {
        "values": obj.base,
        "shape": obj.shape,
        "order": obj.order,
        "filename": file_path,
        "file_format": file_type_to_int(file_type),
        "dset": dset,
        "write_mode": mode_str_to_int(mode),
        "objType": "ArrayView",
    }

    generic_msg(
        cmd="tohdf",
        args=args,
    )


def save_all(
    columns: Union[Mapping[str, pdarray], List[pdarray]],
    prefix_path: str,
    names: List[str] = None,
    file_format="HDF5",
    mode: str = "truncate",
    file_type: str = "distribute",
) -> None:
    """
    DEPRECATED
    Save multiple named pdarrays to HDF5 files.

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
    save, load_all

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
    if names is not None:
        if len(names) != len(columns):
            raise ValueError("Number of names does not match number of columns")
        else:
            datasetNames = names
    if isinstance(columns, dict):
        pdarrays = list(columns.values())
        if names is None:
            datasetNames = list(columns.keys())
    elif isinstance(columns, list):
        pdarrays = cast(List[pdarray], columns)
        if names is None:
            datasetNames = [str(column) for column in range(len(columns))]
    if mode.lower() not in ["append", "truncate"]:
        raise ValueError("Allowed modes are 'truncate' and 'append'")

    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        arr.save(
            prefix_path=prefix_path,
            dataset=name,
            file_format=file_format,
            mode=mode,
            file_type=file_type,
        )
        if mode.lower() == "truncate":
            mode = "append"
