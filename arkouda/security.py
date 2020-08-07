import os, platform, secrets, json
from os.path import expanduser
from collections import defaultdict 
from arkouda import io_util

username_tokenizer = defaultdict(lambda x : x.split('/'))
username_tokenizer['Windows'] = lambda x : x.split('\\')
username_tokenizer['Linux'] = lambda x : x.split('/')
username_tokenizer['Darwin'] = lambda x : x.split('/')

def generate_token(length : int=32) -> str:
    """
    Uses the secrets.token_hex() method to generate a
    a hexidecimal token

    :param int length: desired length of token
    :return: hexidecimal string
    :rtype: str
    """
    return secrets.token_hex(length//2)

def get_home_directory() -> str:
    """
    A platform-independent means of finding path to
    the current user's home directory    

    :return: string corresponding to home directory path
    :rtype: str
    """
    return expanduser("~")

def get_arkouda_client_directory() -> str:
    """
    A platform-independent means of finding path to
    the current user's .arkouda directory where artifacts
    such as server access tokens are stored. 

    The default implementation is to place the .arkouda 
    directory in the current user's home directory. The
    default can be overridden by seting the ARKOUDA_HOME
    environment variable.

    :return: string corresponding to .arkouda directory path
    :rtype: str
    """
    arkouda_parent_dir = os.getenv('ARKOUDA_CLIENT_DIRECTORY')
    if not arkouda_parent_dir:
        arkouda_parent_dir = get_home_directory()
    return io_util.get_directory('{}{}.arkouda'.\
                format(arkouda_parent_dir,os.sep)).absolute()

def get_username() -> str:
    """
    A platform-independent means of retrieving the current user's 
    username for the host system.

    :return: the username string
    :rtype: str
    :raise: EnvironmentError if the host OS is unsupported
    """
    try:
        u_tokens = \
          username_tokenizer[platform.system()](get_home_directory())
    except KeyError as ke:
        return EnvironmentError('Unsupported OS')
    return u_tokens[-1]

def generate_username_token_json(token : str) -> str:
    """
    Generates a JSON object encapsulating the user's username
    and token for connecting to an arkouda server with basic
    authentication enabled

    :param str token: the token to be used to access arkouda server
    :return: JSON-formatted string
    :rtype: str
    """
    return json.dumps({'username' : get_username(),
                       'token' : token})
