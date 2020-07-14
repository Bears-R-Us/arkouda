from pathlib import Path
from typing import Any, Mapping
import csv

def get_directory(path : str) -> Path:
    '''
    Creates the directory if it does not exist and then
    returns the corresponding Path object

    :param str path: the path to the directory
    :return: Path object corresponding to the directory
    :rtype: Path
    :raise: ValueError if there's an error in reading
            existing directory or creating new one
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

    :param str path: path to the target file
    :param str line: line to be written to file
    :return: None
    :raise: UnsupportedOption if there's an error creating
            or writing to the file
    """
    with open(path, 'a') as f:
        f.write(''.join([line,'\n']))

def delimited_file_to_dict(path : str, 
                      delimiter : str=',') -> Mapping[str,str]: 
    """
    Returns a dictionary populated by lines from a file where 
    the first delimited element of each line is the key and
    the second delimited element is the value. 
  
    :param str path: path to file
    :param str delimiter: delimiter separating key and value
    :return: dict containing key -> value
    :rtype: Mapping[str,str]
    :raise: UnsupportedOperation if there's an error in reading
            the file
    """
    values : Mapping[str,str] = {}
    
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

    :param str path: path to file
    :param str delimiter: delimiter separating key and value
    :return: dict containing key -> value
    :rtype: Mapping[str,str]
    :raise: IOError if there's an error opening or writing to
            the specified file, ValueError if the delimiter is
            not supported
    """
    if ',' == delimiter:
        with open(path, 'w+') as f:
            for key,value in values.items():
                f.write('{},{}\n'.format(key,value))
    else:
        raise ValueError('the delimiter {} is not supported'.\
                                       format(delimiter))
