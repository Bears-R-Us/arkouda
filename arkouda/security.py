import platform, secrets, json
from os.path import expanduser

username_tokenizer = {
  'Windows' : lambda x : x.split('\\'),
  'Linux' : lambda x : x.split('/')
}

def generate_token(length : int=32) -> str:
    return secrets.token_hex(length)

def get_home_directory() -> str:
    return expanduser("~")

def get_arkouda_directory() -> str:
    return '{}/.arkouda'.format(get_home_directory())

def get_username() -> str:
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
