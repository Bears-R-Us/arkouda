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

    def asdict(self):
        return {'user': self.user,
                'token': self.token,
                'cmd': self.cmd,
                'format': str(self.format),
                'args' : ''}
