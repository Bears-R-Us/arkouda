from string import ascii_lowercase, ascii_uppercase, digits
import random
import numpy as np
import h5py
from context import arkouda as ak

ALPHABET = ascii_lowercase
UPPERCASE_ALPHABET = ascii_uppercase
ALPHANUMERIC = ascii_lowercase + digits
UPPERCASE_ALPHANUMERIC = ascii_uppercase + digits

def generate_alpha_string(length : int=10, uppercase : bool=False) -> str:
    """
    Generates a random string of a specified length composed of either 
    uppercase or lowercase English alphabet letters
    
    :param int length: length of string
    :param bool uppercase: indicates whether the letters are upper case
    :return: string composed of English alphabet letters
    :retype: str
    """
    if uppercase:
        return ''.join(random.choice(UPPERCASE_ALPHABET) for _ in range(length))
    else:
        return ''.join(random.choice(ALPHABET) for _ in range(length))

def generate_alphanumeric_string(length : int=10, uppercase : bool=False) -> str:
    """
    Generates a random string of a specified length composed of 
    either uppercase or lowercase English alphabet letters and digits
    
    :param int length: length of string
    :param bool uppercase: indicates whether the characters are upper case
    :return: string composed of English alphabet letters and digits
    :retype: str
    """  
    if uppercase:
        return ''.join(random.choice(UPPERCASE_ALPHANUMERIC) for _ in range(length))
    else:
        return ''.join(random.choice(ALPHANUMERIC)  for _ in range(length))
      
def generate_alpha_string_array(string_length : int=10, array_length : 
                    int=100, uppercase : bool=False) -> np.ndarray:
    """
    Generates a Numpy ndarray containing random strings composed of either
    lowercase or uppercase English alphabet letters
    
    :param int string_length: length of string
    :param int array_length: length of Numpy array
    :param bool uppercase: indicates whether the characters are upper case
    :return: Numpy array containing strings composed of English alphabet characters
    :retype: np.ndarray
    """
    return np.array([generate_alpha_string(length=string_length, 
                                  uppercase=uppercase) for _ in range(array_length)], dtype='str') 
  
def generate_alphanumeric_string_array(string_length : int=10, array_length : int=100, 
                                       uppercase : bool=False) -> np.ndarray:
    """
    Generates a Numpy ndarray containing random strings of a specified length composed of 
    either uppercase or lowercase English alphabet letters and digits
    
    :param int string_length: length of string
    :param int array_length: length of Numpy array
    :param bool uppercase: indicates whether the characters are upper case
    :return: Numpy array containing strings composed of English alphabet letters and digits
    :retype: np.ndarray
    """  
    return np.array([generate_alphanumeric_string(length=string_length, 
                                  uppercase=uppercase) for _ in range(array_length)])


def generate_hdf5_file_with_datasets(dataset : ak.pdarray, filepath : str) -> None:

    """
    Creates an hdf5 file and populates it with a dataset in the form of an Akrouda pdarray

    :oaram ak.pdarray dataset: the Arkouda pdarray to be persisted
    :param str filepath: filepath to hdf5 file
    """
    if not datasets or not filepath:
        raise ValueError('both datasets and filepath must be not None')
    
    h_file = h5py.File(filepath)
    h_file.create_dataset(dataset)
