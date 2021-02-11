from typeguard import typechecked
import json, os
from typing import cast, Dict, List, Mapping, Optional, Union
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.strings import Strings

__all__ = ["ls_hdf", "read_hdf", "read_all", "load", "get_datasets",
           "load_all", "save_all"]

@typechecked
def ls_hdf(filename : str) -> str:
    """
    This function calls the h5ls utility on a filename visible to the
    arkouda server.

    Parameters
    ----------
    filename : str
        The name of the file to pass to h5ls

    Returns
    -------
    str
        The string output of `h5ls <filename>` from the server
    """
    return cast(str,generic_msg(cmd="lshdf", args="{}".format(json.dumps([filename]))))

@typechecked
def read_hdf(dsetName : str, filenames : Union[str,List[str]],
             strictTypes: bool=True) \
          -> Union[pdarray, Strings]:
    """
    Read a single dataset from multiple HDF5 files into an Arkouda
    pdarray or Strings object.

    Parameters
    ----------
    dsetName : str
        The name of the dataset (must be the same across all files)
    filenames : list or str
        Either a list of filenames or shell expression
    strictTypes: bool
        If True (default), require all dtypes in all files to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one 
        file contains a uint32 dataset and another contains an int64
        dataset, the contents of both will be read into an int64 pdarray.
        
    Returns
    -------
    Union[pdarray,Strings] 
        A pdarray or Strings instance pointing to the server-side data

    Raises
    ------
    TypeError 
        Raised if dsetName is not a str or if filenames is neither a string
        nor a list of strings
    ValueError 
        Raised if all datasets are not present in all hdf5 files    

    See Also
    --------
    get_datasets, ls_hdf, read_all, load, save

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files. Use ``get_datasets`` to
    show the names of datasets in HDF5 files.

    If dsetName is not present in all files, a TypeError is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    # rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    # # This is a hack to detect a string return type
    # # In the future, we should put the number and type into the return message
    # if '+' in rep_msg:
    #     return Strings(*rep_msg.split('+'))
    # else:
    #     return create_pdarray(rep_msg)
    return cast(Union[pdarray, Strings], 
                read_all(filenames, datasets=dsetName, strictTypes=strictTypes))

def read_all(filenames : Union[str,List[str]],
             datasets : Optional[Union[str,List[str]]]=None,
             iterative : bool=False,
             strictTypes: bool=True) \
             -> Union[pdarray, Strings, Mapping[str,Union[pdarray,Strings]]]:
    """
    Read datasets from HDF5 files.

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

    Returns
    -------
    For a single dataset returns an Arkouda pdarray or Arkouda Strings object
    and for multiple datasets returns a dictionary of Arkouda pdarrays or
    Arkouda Strings.
        Dictionary of {datasetName: pdarray or String}

    Raises
    ------
    ValueError 
        Raised if all datasets are not present in all hdf5 files

    See Also
    --------
    read_hdf, get_datasets, ls_hdf

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
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if datasets is None:
        datasets = get_datasets(filenames[0])
    if isinstance(datasets, str):
        datasets = [datasets]
    else: # ensure dataset(s) exist
        if isinstance(datasets, str):
            datasets = [datasets]
        nonexistent = set(datasets) - set(get_datasets(filenames[0]))
        if len(nonexistent) > 0:
            raise ValueError("Dataset(s) not found: {}".format(nonexistent))
    if iterative == True: # iterative calls to server readhdf
        return {dset:read_hdf(dset, filenames, strictTypes=strictTypes) for dset in datasets}
    else:  # single call to server readAllHdf
        rep_msg = generic_msg(cmd="readAllHdf", args="{} {:n} {:n} {} | {}".\
                format(strictTypes, len(datasets), len(filenames), json.dumps(datasets), 
                       json.dumps(filenames)))
        if ',' in rep_msg:
            rep_msgs = cast(str,rep_msg).split(' , ')
            d : Dict[str,Union[pdarray,Strings]] = dict()
            for dset, rm in zip(datasets, rep_msgs):
                if('+' in cast(str,rm)): #String
                    d[dset]=Strings(*cast(str,rm).split('+'))
                else:
                    d[dset]=create_pdarray(cast(str,rm))
            return d
        elif '+' in rep_msg:
            return Strings(*cast(str,rep_msg).split('+'))
        else:
            return create_pdarray(cast(str,rep_msg))

@typechecked
def load(path_prefix : str, dataset : str='array') -> Union[pdarray,Strings]:
    """
    Load a pdarray previously saved with ``pdarray.save()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    dataset : str
        Dataset name where the pdarray was saved, defaults to 'array'

    Returns
    -------
    Union[pdarray, Strings]
        The pdarray or Strings that was previously saved

    Raises
    ------
    TypeError 
        Raised if dataset is not a str 
    ValueError 
        Raised if all datasets are not present in all hdf5 files     

    See Also
    --------
    save, load_all, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    globstr = "{}_LOCALE*{}".format(prefix, extension)
    return read_hdf(dataset, globstr)

@typechecked
def get_datasets(filename : str) -> List[str]:
    """
    Get the names of datasets in an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of an HDF5 file visible to the arkouda server

    Returns
    -------
    List[str]
        Names of the datasets in the file
        
    Raises
    ------
    TypeError
        Raised if filename is not a str

    See Also
    --------
    ls_hdf
    """
    rep_msg = ls_hdf(filename)
    datasets = [line.split()[0] for line in rep_msg.splitlines()]
    return datasets

@typechecked
def load_all(path_prefix : str) -> Mapping[str,Union[pdarray,Strings]]:
    """
    Load multiple pdarrays or Strings previously saved with ``save_all()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray

    Returns
    -------
    Mapping[str,pdarray]
        Dictionary of {datsetName: pdarray} with the previously saved pdarrays
        
        
    Raises
    ------
    ValueError 
        Raised if all datasets are not present in all hdf5 files    

    See Also
    --------
    save_all, load, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = "{}_LOCALE0000{}".format(prefix, extension)
    try:
        return {dataset: load(path_prefix, dataset=dataset) \
                                       for dataset in get_datasets(firstname)}
    except RuntimeError:
        # enables backwards compatibility with previous naming convention
        firstname = "{}_LOCALE0{}".format(prefix, extension)
        return {dataset: load(path_prefix, dataset=dataset) \
                                       for dataset in get_datasets(firstname)}

def save_all(columns : Union[Mapping[str,pdarray],List[pdarray]], prefix_path : str, 
             names : List[str]=None, mode : str='truncate') -> None:
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
        pdarrays = cast(List[pdarray],columns)
        if names is None:
            datasetNames = [str(column) for column in range(len(columns))]
    if (mode.lower() not in 'append') and (mode.lower() not in 'truncate'):
        raise ValueError("Allowed modes are 'truncate' and 'append'")
    first_iter = True
    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        '''Append all pdarrays to existing files as new datasets EXCEPT the first one, 
           and only if user requests truncation'''
        if mode.lower() not in 'append' and first_iter:
            arr.save(prefix_path=prefix_path, dataset=name, mode='truncate')
            first_iter = False
        else:
            arr.save(prefix_path=prefix_path, dataset=name, mode='append')
