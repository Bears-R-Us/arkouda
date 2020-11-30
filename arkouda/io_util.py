from pathlib import Path
from typing import Any, Dict, Mapping

def get_directory(path : str) -> Path:
    '''
    Creates the directory if it does not exist and then
    returns the corresponding Path object

    Parameters
    ----------
    path : str
        The path to the directory
    
    Returns
    -------
    str
        Path object corresponding to the directory
        
    Raises
    ------
    ValueError
        Raised if there's an error in reading an
        existing directory or creating a new one
    '''
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path) 
    except Exception as e:
        raise ValueError(e)

def write_line_to_file(path : str, line : str) -> None:
    """
    Writes a line to the requested file. Note: if the file
    does not exist, the file is created first and then
    the specified line is written to it.

    Parameters
    ----------
    path : str
        Path to the target file
    line : str
        Line to be written to the file

    Returns
    -------
    None

    Raises
    ------
    UnsupportedOption
        Raised if there's an error in creating or 
        writing to the file
    """
    with open(path, 'a') as f:
        f.write(''.join([line,'\n']))

def delimited_file_to_dict(path : str, 
                      delimiter : str=',') -> Dict[str,str]: 
    """
    Returns a dictionary populated by lines from a file where 
    the first delimited element of each line is the key and
    the second delimited element is the value. 

    Parameters
    ----------
    path : str
        Path to the file
    delimiter : str
        Delimiter separating key and value

    Returns
    -------
    Mapping[str,str]
        Dictionary containing key,value pairs derived from each
        line of delimited strings

    Raises
    ------
    UnsupportedOperation 
        Raised if there's an error in reading the file
    """
    values : Dict[str,str] = {}
    
    with open(path,'a+') as f:
        f.seek(0)
        for line in f:
            line = line.rstrip()
            key,value = line.split(delimiter)
            values[key] = value
    return values

def dict_to_delimited_file(path : str, values : Mapping[Any,Any],
                      delimiter : str=',') -> None:
    """
    Writes a dictionary to delimited lines in a file where
    the first delimited element of each line is the dict key
    and the second delimited element is the dict value. If the 
    file does not exist, it is created and then written to.

    Parameters
    ----------
    path : str
        Path to the file
    delimiter
        Delimiter separating key and value
        
    Returns
    -------
    None
    
    Raises
    ------
    OError 
        Raised if there's an error opening or writing to the 
        specified file
    ValueError 
        Raised if the delimiter is not supported
    """
    if ',' == delimiter:
        with open(path, 'w+') as f:
            for key,value in values.items():
                f.write('{},{}\n'.format(key,value))
    else:
        raise ValueError('the delimiter {} is not supported'.\
                                       format(delimiter))
