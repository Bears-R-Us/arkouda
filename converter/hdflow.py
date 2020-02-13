import pandas as pd
import numpy as np
import h5py
import os
from itertools import repeat
from tqdm import tqdm
import multiprocessing as mp

def _normalize_dtype(col, dtype):
    if dtype == np.str_ or dtype == str or (isinstance(dtype, str) and dtype in 'strings'):
        normdtype = np.dtype('str')
    elif dtype is not None:
        # data = col.astype(dtype).values
        if callable(dtype):
            normdtype = dtype().dtype
    # Convert datetime64 to int64, but save original dtype for round-trip conversion
    elif col.dtype.kind == 'M':
        # data = col.astype(np.int64).values
        normdtype = col.dtype
    else:
        # data = col.values
        normdtype = col.dtype
    return normdtype

def pack_strings(column):
    lengths = column.apply(len).values
    offsets = lengths.cumsum() + np.arange(lengths.shape[0]) - lengths
    totalbytes = lengths.sum() + lengths.shape[0]
    packed = np.zeros(shape=(totalbytes,), dtype=np.uint8)
    for (o, s) in zip(offsets, column.values):
        for i, b in enumerate(s.encode()):
            packed[o+i] = b
    return packed, offsets

def _cast_values(col, dtype):
    if dtype == np.str_ or dtype == str or (isinstance(dtype, str) and dtype in 'strings'):
        data = pack_strings(col)
    elif dtype is not None:
        data = col.astype(dtype).values
    elif col.dtype.kind == 'M':
        data = col.astype(np.int64).values
    else:
        data = col.values
    return data

def write_strings(f, group, packed, offsets):
    g = f.create_group(group)
    g.attrs["segmented_string"] = np.bool(True)
    g.create_dataset('values', data=packed)
    g.create_dataset('segments', data=offsets)

def col2dset(name, col, f, dtype=None, compression='gzip'):
    '''Write a pandas Series <col> to dataset <name> in HDF5 file <f>.
    Optionally, specify a dtype to convert to before writing. Compression
    with gzip is also supported.'''
    normdtype = _normalize_dtype(col, dtype)
    data = _cast_values(col, dtype)
    try:
        if normdtype == np.dtype('str'):
            packed, offsets = data
            write_strings(f, name, packed, offsets)
        else:
            dset = f.create_dataset(name, data=data, compression=compression)
            # Store the normalized dtype for conversion back to a pd.Series
            dset.attrs['dtype'] = np.string_(normdtype.str)
    except TypeError:
        print(f"Column {col}")


def df2hdf(filename, df, dtypes={}, compression='gzip'):
    '''Write a pandas DataFrame <df> to a HDF5 file <filename>. Optionally,
    specify dtypes for converting the columns and a compression to use.'''
    with h5py.File(filename, 'w') as f:
        for colname in df.columns:
            col2dset(colname, df[colname], f, dtype=dtypes.get(colname, None), compression=compression)

def convert_file(args):
    filename, outdir, extension, options = args
    try:
        df = pd.read_csv(filename, **options)
    except Exception as e:
        print(f"Error converting {filename}:")
        print(e)
        return
    newname = os.path.splitext(os.path.basename(filename))[0] + extension
    df2hdf(os.path.join(outdir, newname), df, options.get('dtype', {}))

def _get_valid_columns(filename, options):
    #print(filename)
    #print(options)
    try:
        #**options causing error
        df = pd.read_csv(filename, nrows=1000, **options)
    except Exception as e:
        print(f"Error reading sample DataFrame: ",e)
        #print(e)
    validcols = []
    for i, col in enumerate(df.columns):
        userdtype = options.get('dtype', {}).get(col, None)
        normdtype = _normalize_dtype(df[col], userdtype)
        data = _cast_values(df[col], userdtype)
        try:
            if normdtype != np.str_:
                h5py.h5t.py_create(data.dtype, logical=True)
            validcols.append(i)
        except TypeError:
            print(f'Ignoring column {col} because dtype "{df[col].values.dtype}" has no HDF5 equivalent.')
    print("Columns to be extracted:")
    print(df[df.columns[validcols]].info())
    used = options.get('usecols', list(range(df.shape[1])))
    new_usecols = [used[v] for v in validcols]
    return new_usecols

def convert_files(filenames, outdir, extension, options, nprocs):
    if not os.path.isdir(outdir) or not os.access(outdir, os.W_OK):
        raise OSError(f"Permission denied: {outdir}")
    usecols = _get_valid_columns(filenames[0], options)
    if len(usecols) == 0:
        raise TypeError("No columns found with HDF5-compatible dtype")
    options['usecols'] = usecols
    arglist = zip(filenames, repeat(outdir), repeat(extension), repeat(options))
    nprocs = min((nprocs, len(filenames)))
    if nprocs <= 1:
        _ = list(tqdm(map(convert_file, arglist), total=len(filenames)))
    else:
        with mp.Pool(nprocs) as pool:
            _ = list(tqdm(pool.imap_unordered(convert_file, arglist), total=len(filenames)))


def read_hdf(filenames):
    df, offsets, lengths = _hdf_alloc(filenames)
    for fname, ind in tqdm(zip(filenames, offsets), total=len(filenames)):
        _hdf_insert(fname, df, ind)
    return df

def _hdf_alloc(filenames):
    if len(filenames) == 0:
        raise ValueError("Need at least one file to allocate a DataFrame")
    # Use first file as a refernce for column dtypes
    dtypes, _ = _get_column_metadata(filenames[0])
    offsets = [0]
    lengths = []
    for fn in filenames:
        # Ensure each file has same number of columns and same dtypes as reference file
        thisdtypes, thislength = _get_column_metadata(fn)
        if len(thisdtypes) != len(dtypes):
            raise ValueError("Number of columns must be constant across all files")
        if thisdtypes != dtypes:
            raise ValueError("Columns have inhomogeneous names or dtypes between files")
        offsets.append(offsets[-1] + thislength)
        lengths.append(thislength)
    # Last entry is total length; remove it so it won't be used as offset
    total = offsets.pop()
    # Allocate uninitialized memory for concatenated columns
    columns = {col:pd.Series(np.empty(shape=(total,), dtype=dtypes[col])) for col in dtypes}
    return pd.DataFrame(columns), offsets, lengths

def _get_column_metadata(filename, dtype_attr='dtype'):
    dtypes = {}
    length = -1
    with h5py.File(filename, 'r') as f:
        # Use HDF5 dataset names as column names
        for colname in f.keys():
            dset = f[colname]
            # Check for user-specified dtype as dset attribute
            # Otherwise, use native HDF5 dtype
            dtypes[colname] = dset.attrs.get(dtype_attr, dset.dtype)
            # Set length on th efirst column, and test that other columns have same length
            if length == -1:
                length = dset.shape[0]
            else:
                assert length == dset.shape[0], f"Columns of unequal length in {filename}"
    return dtypes, length

def _hdf_insert(filename, df, index):
    with h5py.File(filename, 'r') as f:
        # _hdf_alloc has already guaranteed that file has correct column names
        for colname in f.keys():
            dset = f[colname]
            size = dset.shape[0]
            # Read the dataset directly into the uninitialized column
            df[colname].values[index:index+size] = dset[:]
