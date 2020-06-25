from pathlib import Path


def get_directory(path : str) -> Path:
    '''
    Creates the directory if it does not exist and then
    returns the corresponding Path object

    :param str path: the path to the directory
    :return: Path object corresponding to the directory
    :rtype: Path
    '''
    return Path(path).mkdir(parents=True, exist_ok=True)

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
