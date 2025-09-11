"""
File and directory utility functions for Arkouda's pandas module.

This module provides a set of helper functions for performing common I/O operations
needed during data processing, configuration, and metadata management. It includes
utilities for creating and deleting directories, reading and writing key-value pairs
from/to delimited text files, and appending lines to files.


Functions
---------
get_directory(path: str) -> Path
    Create a directory if it doesn't exist and return its `Path` object.

write_line_to_file(path: str, line: str) -> None
    Append a single line to a file, creating it if necessary.

delimited_file_to_dict(path: str, delimiter: str = ",") -> Dict[str, str]
    Load key-value pairs from a delimited file into a dictionary.

dict_to_delimited_file(path: str, values: Mapping[Any, Any], delimiter: str = ",") -> None
    Write a dictionary to a file as delimited key-value lines.

delete_directory(dir: str) -> None
    Recursively delete a directory if it exists.

Notes
-----
- `dict_to_delimited_file` currently only supports a comma (`,`) as a delimiter.
- Errors encountered during I/O are either printed or raised, depending on context.
- This module is intended for internal use within Arkouda's pandas compatibility layer.

Examples
--------
>>> from arkouda.io_util import get_directory, write_line_to_file
>>> path = get_directory("tmp/output")
>>> write_line_to_file(path / "log.txt", "Computation completed")

"""

from os.path import isdir
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping

from arkouda.logger import getArkoudaLogger


__all__ = [
    "delete_directory",
    "delimited_file_to_dict",
    "dict_to_delimited_file",
    "get_directory",
    "write_line_to_file",
]

logger = getArkoudaLogger("io_util Logger")


def get_directory(path: str) -> Path:
    """
    Create the directory if it does not exist and returns the corresponding Path object.

    Parameters
    ----------
    path : str
        The path to the directory

    Returns
    -------
    Path
        Object corresponding to the directory

    Raises
    ------
    ValueError
        Raised if there's an error in reading an
        existing directory or creating a new one

    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)
    except Exception as e:
        raise ValueError(e)


def write_line_to_file(path: str, line: str) -> None:
    """
    Write a line to the requested file.

    Note: if the file
    does not exist, the file is created first and then
    the specified line is written to it.

    Parameters
    ----------
    path : str
        Path to the target file
    line : str
        Line to be written to the file

    Raises
    ------
    UnsupportedOption
        Raised if there's an error in creating or
        writing to the file

    """
    with open(path, "a") as f:
        f.write("".join([line, "\n"]))


def delimited_file_to_dict(path: str, delimiter: str = ",") -> Dict[str, str]:
    """
    Return a dictionary populated by lines from a file.

    Return a dictionary populated by lines from a file where
    the first delimited element of each line is the key and
    the second delimited element is the value.
    If the file does not exist, return an empty dictionary.

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
    ValueError
        Raised if a line has more or fewer than two delimited elements

    """
    values: Dict[str, str] = {}

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                key, value = line.split(delimiter)
                values[key] = value
    except FileNotFoundError:
        pass  # return an empty dictionary

    return values


def dict_to_delimited_file(path: str, values: Mapping[Any, Any], delimiter: str = ",") -> None:
    """
    Write a dictionary to delimited lines in a file.

    Write a dictionary to delimited lines in a file where
    the first delimited element of each line is the dict key
    and the second delimited element is the dict value. If the
    file does not exist, it is created and then written to.

    Parameters
    ----------
    path : str
        Path to the file
    delimiter : str
        Delimiter separating key and value

    Raises
    ------
    OError
        Raised if there's an error opening or writing to the
        specified file
    ValueError
        Raised if the delimiter is not supported

    """
    if "," == delimiter:
        with open(path, "w+") as f:
            for key, value in values.items():
                f.write(f"{key},{value}\n")
    else:
        raise ValueError(f"the delimiter {delimiter} is not supported")


def delete_directory(dir: str) -> None:
    """
    Delete the directory if it exists.

    Parameters
    ----------
    dir : str
        The path to the directory

    Raises
    ------
    OSError
        Raised if there's an error in deleting the directory.

    """
    if isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError as e:
            logger.error("Error: %s - %s." % (e.filename, e.strerror))


def directory_exists(dir: str) -> bool:
    """
    Return True if the directory exists.

    Parameters
    ----------
    dir : str
        The path to the directory

    Returns
    -------
    True if the directory exists, False otherwise.

    """
    return isdir(dir)
