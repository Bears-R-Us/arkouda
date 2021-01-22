from dataclasses import dataclass
from enum import Enum

class MessageFormat(Enum):
    STRING = 'STRING'
    BINARY = 'BINARY'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

@dataclass(frozen=True)
class Message():

    user: str
    token: str
    cmd: str
    format: MessageFormat
    args: str=''

    def __init__(self, user : str, cmd : str, token : str=None, 
                 format : MessageFormat=MessageFormat.STRING, args : str=None) -> None:
        object.__setattr__(self, 'user',user)
        object.__setattr__(self, 'token',token)
        object.__setattr__(self, 'cmd',cmd)
        object.__setattr__(self, 'format',format)
        object.__setattr__(self, 'args',args)

    def asdict(self):
        args = self.args if self.args else ''
        token = self.token if self.token else ''

        return {'user': self.user,
                'token': token,
                'cmd': self.cmd,
                'format': str(self.format),
                'args' : args}
