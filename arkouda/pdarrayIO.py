import json, os

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray

__all__ = ["ls_hdf", "read_hdf", "read_all", "load", "get_datasets",
           "load_all", "save_all"]

def ls_hdf(filename):
    """
    This function calls the h5ls utility on a filename visible to the arkouda 
    server.

    Parameters
    ----------
    filename : str
        The name of the file to pass to h5ls

    Returns
    -------
    str 
        The string output of `h5ls <filename>` from the server
    """
    return generic_msg("lshdf {}".format(json.dumps([filename])))

def read_hdf(dsetName, filenames):
    """
    Read a single dataset from multiple HDF5 files into an arkouda pdarray. 

    Parameters
    ----------
    dsetName : str
        The name of the dataset (must be the same across all files)
    filenames : list or str
        Either a list of filenames or shell expression

    Returns
    -------
    pdarray
        A pdarray instance pointing to the server-side data read in

    See Also
    --------
    get_datasets, ls_hdf, read_all, load, save

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression 
    (a single filename is a valid expression, so it will work) and is 
    expanded with glob to read all matching files. Use ``get_datasets`` to 
    show the names of datasets in HDF5 files.

    If dsetName is not present in all files, a RuntimeError is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    return create_pdarray(rep_msg)

def read_all(filenames, datasets=None):
    """
    Read multiple datasets from multiple HDF5 files.
    
    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read (default: all available)

    Returns
    -------
    dict of pdarrays
        Dictionary of {datasetName: pdarray}

    See Also
    --------
    read_hdf, get_datasets, ls_hdf
    
    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression 
    (a single filename is a valid expression, so it will work) and is 
    expanded with glob to read all matching files. This is done separately
    for each dataset, so if new matching files appear during ``read_all``,
    some datasets will contain more data than others. 

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets in 
    HDF5 files.

    If not all datasets are present in all HDF5 files, a RuntimeError
    is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    alldatasets = get_datasets(filenames[0])
    if datasets is None:
        datasets = alldatasets
    else: # ensure dataset(s) exist
        if isinstance(datasets, str):
            datasets = [datasets]
        nonexistent = set(datasets) - set(get_datasets(filenames[0]))
        if len(nonexistent) > 0:
            raise ValueError("Dataset(s) not found: {}".format(nonexistent))
    return {dset:read_hdf(dset, filenames) for dset in datasets}

def load(path_prefix, dataset='array'):
    """
    Load a pdarray previously saved with ``pdarray.save()``. 
    
    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    dataset : str
        Dataset name where the pdarray was saved

    Returns
    -------
    pdarray
        The pdarray that was previously saved

    See Also
    --------
    save, load_all, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    globstr = "{}_LOCALE*{}".format(prefix, extension)
    return read_hdf(dataset, globstr)

def get_datasets(filename):
    """
    Get the names of datasets in an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of an HDF5 file visible to the arkouda server

    Returns
    -------
    list of str
        Names of the datasets in the file
    
    See Also
    --------
    ls_hdf
    """
    rep_msg = ls_hdf(filename)
    datasets = [line.split()[0] for line in rep_msg.splitlines()]
    return datasets
            
def load_all(path_prefix):
    """
    Load multiple pdarray previously saved with ``save_all()``. 
    
    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray

    Returns
    -------
    dict of pdarrays
        Dictionary of {datsetName: pdarray} with the previously saved pdarrays

    See Also
    --------
    save_all, load, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = "{}_LOCALE0{}".format(prefix, extension)
    return {dataset: load(path_prefix, dataset=dataset) for dataset in get_datasets(firstname)}

def save_all(columns, path_prefix, names=None, mode='truncate'):
    """
    Save multiple named pdarrays to HDF5 files.

    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    path_prefix : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist. 
        If 'append', attempt to create new dataset in existing files.

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
    if names is not None and len(names) != len(columns):
        raise ValueError("Number of names does not match number of columns")
    if isinstance(columns, dict):
        pdarrays = columns.values()
        if names is None:
            names = columns.keys()
    elif isinstance(columns, list):
        pdarrays = columns
        if names is None:
            names = range(len(columns))
    if (mode.lower() not in 'append') and (mode.lower() not in 'truncate'):
        raise ValueError("Allowed modes are 'truncate' and 'append'")
    first_iter = True
    for arr, name in zip(pdarrays, names):
        # Append all pdarrays to existing files as new datasets EXCEPT the first one, and only if user requests truncation
        if mode.lower() not in 'append' and first_iter:
            arr.save(path_prefix, dataset=name, mode='truncate')
            first_iter = False
        else:
            arr.save(path_prefix, dataset=name, mode='append')
