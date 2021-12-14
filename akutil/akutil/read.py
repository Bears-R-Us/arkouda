import arkouda as ak
import akutil as aku

def _read_chunk(files, start, end, filterfunc=None, **kwargs):
    chunk = ak.read_all(files[start:min((end, len(files)))], **kwargs)
    # Single datasets will get squeezed into a pdarray, so stuff back into a dict
    if type(chunk) != dict:
        if 'datasets' in kwargs:
            if isinstance(kwargs['datasets'], str):
                dataset = kwargs['datasets']
            else:
                dataset = kwargs['datasets'][0]
        else:
            dataset = ak.get_datasets(files[start])[0]
        chunk = {dataset: chunk}
    if filterfunc is not None:
        f = filterfunc(chunk)
        print(f'Filtering out {f.size - f.sum():,} records')
        chunk = {k:v[f] for k, v in chunk.items()}
    return chunk

def read_checkpointed(files, filterfunc=None, prior_data=None, prefix=None, 
                      chunksize=1000, checkpoint=0, clear=False, asframe=True, 
                      converters={}, strictTypes=False, **kwargs):
    '''Read files in chunks in a recoverable fashion, optionally appending to existing 
    data and/or performing aggressive memory conservation. If initial data argument is
    supplied, any chunks read will be available to the user regardless of errors in
    later chunks. Function can be called again from checkpoint to resume reading.

    Parameters
    ----------
    files : list
        List of filenames to read
    filterfunc : function
        Function that accepts a data dictionary and returns a boolean array indicating
        which rows of data to keep. By default, no filtering is performed.
    data : dict of pdarray
        Initial data dictionary, to which new data will be appended in-place. Even if
        errors occur, chunks that are successfully read will exist in data.
    prefix : str
        Prefix with which to register data arrays in arkouda. Can be used with
        ak.attach_pdarray() to recover data.
    chunksize : int
        Number of files to read in each chunk
    checkpoint : int
        Index in files list for restarting. If an error occurs, the message will
        specify the checkpoint value to use.
    clear : bool
        If True (default: False), call ak.clear() after reading each chunk. This will
        aggressively conserve memory by deleting all arrays that have not been
        registered. WARNING: before using this option, be sure to register all
        non-temporary arrays!
    asframe : bool
        If False, return a dictionary of arkouda arrays. By default, return 
        a DataFrame.
    converters : dict-like
        A mapping of column name to function that will be called on that column
        after it is read. If a column is not present, no error is raised.
    kwargs
        Passed to ak.read_all()

    Returns
    -------
    data : dict of pdarray
        Dictionary emulating a dataframe of all data in files that passes filter
    '''
    if prefix is None and clear:
        raise ValueError("Must supply a registration prefix (prefix=) with clear=True")
    if prior_data is None:
        data = {}
    else:
        data = prior_data
    if len(data) == 0:
        size = 0
    else:
        size = list(data.values())[0].size
    for i in range(checkpoint, len(files), chunksize):
        try:
            print(f'Reading files {i}:{min((len(files), i + chunksize))}')
            chunk = _read_chunk(files, i, i+chunksize, filterfunc=filterfunc, strictTypes=strictTypes, **kwargs)
            s = list(chunk.values())[0].size
            print(f'{s:,} records read')
        except Exception as e:
            raise RuntimeError(f'Error encountered: restart with checkpoint={i}') from e
        if len(data) > 0:
            if (set(chunk.keys()) != set(data.keys())):
                raise ValueError(f"Incompatible chunk: mismatched columns: {chunk.keys()} vs. {data.keys()}")
            for k in chunk:
                # Append to the data dict in-place
                # In-place update ensures data survives any errors raised
                data[k] = ak.concatenate((data[k], chunk[k]))
        else:
            for k in chunk:
                # Update the data dict in-place
                data[k] = chunk[k]
        if prefix is not None:
            data = aku.register_all(data, prefix=prefix)
        size += s
        if clear:
            # Clear to stay under memory ceiling
            ak.clear()
        if i > checkpoint:
            print(f'{size:,} total records')
    for col, convert in converters.items():
        if col in data:
            data[col] = convert(data[col])
    if asframe:
        data = aku.DataFrame(data)
    return data
