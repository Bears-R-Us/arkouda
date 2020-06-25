from pathlib import Path
from typing import Mapping

def get_directory(path : str) -> Path:
    '''
    Creates the directory if it does not exist and then
    returns the corresponding Path object

    :param str path: the path to the directory
    :return: Path object corresponding to the directory
    :rtype: Path
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
    the specified line is written to i.

    :param str path: path to the target file
    :param str line: line to be written to file
    :return: None
    """
    with open(path, 'a') as f:
        f.write(''.join([line,'\n']))

def delimited_file_to_dict(path : str, 
                      delimiter : str=',') -> Mapping[str,str]: 
    """
    Returns a dict populated by lines from a file where 
    the first delmited element of each line is the key and
    the second delimited element is the value.
    
    :param str path: path to file
    :param str delimiter: delimiter separating key and value
    :return: dict containing key -> value
    :rtype: Mapping[str,str]
    """
    try:
        values : Mapping[str,str] = {}

        with open(path) as f:
            for line in f:
                line = line.rstrip()
                key,value = line.split(delimiter)
                values[key] = value
        return values
    except Exception as e:
        raise ValueError(e)
