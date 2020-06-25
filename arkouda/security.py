import platform, secrets, json
from os.path import expanduser
from arkouda import io_util

username_tokenizer = {
  'Windows' : lambda x : x.split('\\'),
  'Linux' : lambda x : x.split('/')
}

def generate_token(length : int=32) -> str:
    """
    Uses the secrets.token_hex() method to generate a
    a hexidecimal token

    :param int length: desired length of token
    :return: hexidecimal string
    :rtype: str
    """
    return secrets.token_hex(length)

def get_home_directory() -> str:
    """
    A platform-independent means of finding path to
    the current user's home directory    

    :return: string corresponding to home directory path
    :rtype: str
    """
    return expanduser("~")

def get_arkouda_directory() -> str:
    """
    A platform-independent means of finding path to
    the current user's .arkouda directory where artifacts
    such as server access tokens are stored.

    :return: string corresponding to home directory path
    :rtype: str
    """
    return '{}/.arkouda'.format(get_home_directory())

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

def generate_username_token_tuple(tuple_length : int=32) -> str:
    return (get_username(),generate_token(tuple_length))

def generate_username_token_json(token : str=None) -> str:
    if not token:
        token = generate_token()
    return json.dumps({'username' : get_username(),
                       'token' : token})
