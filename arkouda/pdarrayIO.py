from __future__ import annotations

import glob
import json
import os
import warnings
from typing import Dict, List, Mapping, Optional, Union, cast

import pandas as pd  # type: ignore
from typeguard import typechecked

import arkouda.array_view
from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.strings import Strings

__all__ = [
    "ls",
    "read",
    "load",
    "get_datasets",
    "load_all",
    "save_all",
    "get_filetype",
    "get_null_indices",
    "import_data",
    "export",
    "read_hdf5_multi_dim",
    "write_hdf5_multi_dim",
]

ARKOUDA_HDF5_FILE_METADATA_GROUP = "_arkouda_metadata"


@typechecked
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
    """
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    cmd = "lsany"
    return json.loads(cast(str, generic_msg(cmd=cmd, args="{}".format(json.dumps([filename])))))


def read(
    filenames: Union[str, List[str]],
    datasets: Optional[Union[str, List[str]]] = None,
    iterative: bool = False,
    strictTypes: bool = True,
    allow_errors: bool = False,
    calc_string_offsets=False,
    file_format: str = "infer",
) -> Union[pdarray, Strings, Mapping[str, Union[pdarray, Strings]]]:
    """
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
    For a single dataset returns an Arkouda pdarray or Arkouda Strings object
    and for multiple datasets returns a dictionary of Arkouda pdarrays or
    Arkouda Strings.
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
    if isinstance(filenames, str):
        filenames = [filenames]
    if datasets is None:
        datasets = get_datasets_allow_errors(filenames) if allow_errors else get_datasets(filenames[0])
    if isinstance(datasets, str):
        datasets = [datasets]
    else:  # ensure dataset(s) exist
        if isinstance(datasets, str):
            datasets = [datasets]
        nonexistent = set(datasets) - (
            set(get_datasets_allow_errors(filenames))
            if allow_errors
            else set(get_datasets(filenames[0]))
        )
        if len(nonexistent) > 0:
            raise ValueError(f"Dataset(s) not found: {nonexistent}")

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
            args=f"{strictTypes} {len(datasets)} {len(filenames)} {allow_errors} {calc_string_offsets} "
            f"{json.dumps(datasets)} | {json.dumps(filenames)}",
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
            d: Dict[str, Union[pdarray, Strings]] = {}
            for item in items:
                if "seg_string" == item["arkouda_type"]:
                    d[item["dataset_name"]] = Strings.from_return_msg(item["created"])
                elif "pdarray" == item["arkouda_type"]:
                    d[item["dataset_name"]] = create_pdarray(item["created"])
                else:
                    raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
            return d
        elif len(items) == 1:
            item = items[0]
            if "pdarray" == item["arkouda_type"]:
                return create_pdarray(item["created"])
            elif "seg_string" == item["arkouda_type"]:
                return Strings.from_return_msg(item["created"])
            else:
                raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
        else:
            raise RuntimeError("No items were returned")


@typechecked
def get_null_indices(filenames, datasets) -> Union[pdarray, Mapping[str, pdarray]]:
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
        args=f"{len(datasets)} {len(filenames)} {json.dumps(datasets)} | {json.dumps(filenames)}",
    )
    rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllHdfMsgJson for json structure
    items = rep["items"] if "items" in rep else []

    # We have a couple possible return conditions
    # 1. We have multiple items returned i.e. multi pdarrays
    # 2. We have a single pdarray
    if len(items) > 1:  # DataSets condition
        d: Dict[str, pdarray] = {}
        for item in items:
            if "pdarray" == item["arkouda_type"]:
                d[item["dataset_name"]] = create_pdarray(item["created"])
            else:
                raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
        return d
    elif len(items) == 1:
        item = items[0]
        if "pdarray" == item["arkouda_type"]:
            return create_pdarray(item["created"])
        else:
            raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
    else:
        raise RuntimeError("No items were returned")


@typechecked
def load(
    path_prefix: str,
    file_format: str = "INFER",
    dataset: str = "array",
    calc_string_offsets: bool = False,
) -> Union[pdarray, Strings, Mapping[str, Union[pdarray, Strings]]]:
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
    globstr = f"{prefix}_LOCALE*{extension}"

    try:
        return read(globstr, dataset, calc_string_offsets=calc_string_offsets, file_format=file_format)
    except RuntimeError as re:
        if "does not exist" in str(re):
            raise ValueError(
                f"There are no files corresponding to the path_prefix {path_prefix} in"
                " location accessible to Arkouda"
            )
        else:
            raise RuntimeError(re)


@typechecked
def get_datasets(filename: str) -> List[str]:
    """
    Get the names of datasets in an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of an HDF5/Parquet file visible to the arkouda server

    Returns
    -------
    List[str]
        Names of the datasets in the file

    Raises
    ------
    TypeError
        Raised if filename is not a str
    ValueError
        Raised if filename is empty or contains only whitespace
    RuntimeError
        Raised if error occurs in executing ls on an HDF5 file

    See Also
    --------
    ls
    """
    datasets = ls(filename)
    # We can skip/remove the _arkouda_metadata group since it is an internal only construct
    if ARKOUDA_HDF5_FILE_METADATA_GROUP in datasets:
        datasets.remove(ARKOUDA_HDF5_FILE_METADATA_GROUP)
    return datasets


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

    See Also
    --------
    read
    """
    if isinstance(filenames, list):
        fname = filenames[0]
    else:
        fname = filenames
    if not (fname and fname.strip()):
        raise ValueError("filename cannot be an empty string")

    return cast(str, generic_msg(cmd="getfiletype", args="{}".format(json.dumps([fname]))))


@typechecked
def get_datasets_allow_errors(filenames: List[str]) -> List[str]:
    """
    Get the names of datasets in an HDF5 file
    Allow file read errors until success

    Parameters
    ----------
    filenames : List[str]
        A list of HDF5 files visible to the arkouda server

    Returns
    -------
    List[str]
        Names of the datasets in the file

    Raises
    ------
    TypeError
        Raised if filenames is not a List[str]
    FileNotFoundError
        If none of the files could be read successfully

    See Also
    --------
    get_datasets, ls
    """
    datasets = []
    for filename in filenames:
        try:
            datasets = get_datasets(filename)
            break
        except RuntimeError:
            pass
    if not datasets:  # empty
        raise FileNotFoundError("Could not read any of the requested files")
    return datasets


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


def save_all(
    columns: Union[Mapping[str, pdarray], List[pdarray]],
    prefix_path: str,
    names: List[str] = None,
    file_format="HDF5",
    mode: str = "truncate",
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
    file_format : str
        'HDF5' or 'Parquet'. Defaults to hdf5
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist.
        If 'append', attempt to create new dataset in existing files.

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
        arr.save(prefix_path=prefix_path, dataset=name, file_format=file_format, mode=mode)
        if mode.lower() == "truncate":
            mode = "append"


@typechecked
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


@typechecked
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


@typechecked
def read_hdf5_multi_dim(file_path: str, dset: str) -> arkouda.array_view.ArrayView:
    """
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
    rep_msg = cast(
        str,
        generic_msg(
            cmd="readhdf_multi",
            args=f"{file_path} {dset}",
        ),
    )

    objs = rep_msg.split("+")

    shape = create_pdarray(objs[0])
    flat = create_pdarray(objs[1])

    arr = arkouda.array_view.ArrayView(flat, shape)
    return arr


@typechecked
def _storage_str_to_int(method: str) -> int:
    """
    Convert string to integer representing the storage method

    Parameters
    ----------
    method: str (flat | multi)
        The string representation of the storage format to be converted to integer

    Returns
    -------
    int representing the storage method

    Raises
    ------
    ValueError
        - If mode is not 'flat' or 'multi'
    """
    if method.lower() == "flat":
        return 0
    elif method.lower() == "multi":
        return 1
    else:
        raise ValueError(f"Storage method expected to be 'flat' or 'multi'. Got {method}.")


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
        - If mode is not 'truncate' or 'append'
    """
    if mode.lower() == "truncate":
        return 0
    elif mode.lower() == "append":
        return 1
    else:
        raise ValueError(f"Write Mode expected to be 'truncate' or 'append'. Got {mode}.")


@typechecked
def write_hdf5_multi_dim(
    obj: arkouda.array_view.ArrayView,
    file_path: str,
    dset: str,
    mode: str = "truncate",
    storage: str = "Flat",
):
    """
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
    storage: str (Flat | Multi)
        Default: Flat
        Method to use when storing the dataset.
        Flat - flatten the multi-dimensional object into a 1-D array of values
        Multi - Store the object in the multidimensional presentation.

    See Also
    --------
    ak.read_hdf5_multi_dim

    Notes
    -----
    - If a file does not exist, it will be created regardless of the mode value
    - This function is currently standalone functionality for multi-dimensional datasets
    - Error handling done on server to prevent multiple server calls
    """
    # error handling is done in the conversion functions
    storage_int = _storage_str_to_int(storage)
    mode_int = _mode_str_to_int(mode)
    generic_msg(
        cmd="writehdf_multi",
        args=f"{obj.base.name} {obj.shape.name} {obj.order} {file_path} {dset} {mode_int} {storage_int}",
    )
