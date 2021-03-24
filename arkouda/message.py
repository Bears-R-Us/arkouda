from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict

"""
The MessageFormat enum provides controlled vocabulary for the message
format which can be either a string or a binary (bytes) object.
"""
class MessageFormat(Enum):
    STRING = 'STRING'
    BINARY = 'BINARY'

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a MessageFormat object to JSON.
        """
        return self.value
    
    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a MessageFormat object to JSON.
        """
        return self.value

"""
The MessageType enum provides controlled vocabulary for the message
type which can be either NORMAL, WARNING, or ERROR.
"""
class MessageType(Enum):
    NORMAL = 'NORMAL'
    WARNING = 'WARNING'
    ERROR = 'ERROR'

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a MessageType object to JSON.
        """
        return self.value
    
    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a MessageType object to JSON.
        """
        return self.value

"""
The Message class encapsulates the attributes required to capture the full
context of an Arkouda server request.
"""
@dataclass(frozen=True)
class RequestMessage():
    
    __slots = ('user', 'token', 'cmd', 'format', 'args')

    user: str
    token: str
    cmd: str
    format: MessageFormat
    args: str

    def __init__(self, user : str, cmd : str, token : str=None, 
                 format : MessageFormat=MessageFormat.STRING, 
                 args : str=None) -> None:
        """
        Overridden __init__ method sets instance attributes to 
        default values if the corresponding init params are missing.
        
        Parameters
        ----------
        
        user : str
            The user the request corresponds to
        cmd : str
            The Arkouda server command name
        token : str, defaults to None
            The authentication token corresponding to the user
        format : MessageFormat
            The request message format 
        args : str
            The delimited string containing the command arguments
            
        Returns
        -------
        None
        """
        object.__setattr__(self, 'user',user)
        object.__setattr__(self, 'token',token)
        object.__setattr__(self, 'cmd',cmd)
        object.__setattr__(self, 'format',format)
        object.__setattr__(self, 'args',args)

    def asdict(self) -> Dict:
        """
        Overridden asdict implementation sets the values of non-required 
        fields to an empty space (for Chapel JSON processing) and invokes 
        str() on the format instance attribute.
        
        Returns
        -------
        Dict
            A dict object encapsulating ReplyMessage state 
        """
        # args and token logic will not be needed once Chapel supports nulls
        args = self.args if self.args else ''
        token = self.token if self.token else ''

        return {'user': self.user,
                'token': token,
                'cmd': self.cmd,
                'format': str(self.format),
                'args' : args}

'''
The ReplyMessage class encapsulates the data and metadata corresponding to
a message returned by the Arkouda server
'''
@dataclass(frozen=True)
class ReplyMessage():

    __slots__ = ('msg', 'msgType', 'user')
    
    msg: str
    msgType: MessageType
    user: str
    
    @staticmethod
    def fromdict(values : Dict) -> ReplyMessage:
        """
        Generates a ReplyMessage from a dict encapsulating the data and
        metadata from a reply returned by the Arkouda server.
        
        Parameters
        ----------
        values : Dict
            The dict object encapsulating the fields required to instantiate
            a ReplyMessage
            
        Returns
        -------
        ReplyMessage
            The ReplyMessage composed of values encapsulated within values dict
        
        Raises
        ------
        ValueError
            Raised if the values Dict is missing fields or contains malformed values
        """
        try: 
            return ReplyMessage(msg=values['msg'], 
                        msgType=MessageType(values['msgType']), user=values['user'])
        except KeyError as ke:
            raise ValueError('values dict missing {} field'.format(ke))
