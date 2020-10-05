from typing import Optional
from logging import Logger, Formatter, Handler, StreamHandler
from enum import Enum

__all__ = ['Logger', 'LogLevel']

"""
The LogLevel enum class defines the log levels for the ArkoudaLogger
"""
class LogLevel(Enum):
    
    DEBUG = 'DEBUG'
    CRITICAL = 'CRITICAL'
    INFO = 'INFO'
    WARN = 'WARN'
    ERROR =  'ERROR'

"""
ArkoudaLogger encapsulates logging configuration and logic to log messages
at varying levels invcluding debug, info, critical, warn, and error

    Attributes
    ----------
    name : str
        The logger name, prepends all logging errors
    logLevel : LogLevel
        The log level for the Arkouda logger, defaults to INFO
    handler : Handler
        The Python Handler object that defines where and how to log messages
"""  
class ArkoudaLogger(Logger):
    
    __slots__ = ('name', 'logLevel', 'handler')
    
    def __init__(self, name : str, level : Optional[LogLevel]=LogLevel.INFO, 
            handler : Optional[Handler]=StreamHandler(), 
            logFormat : Optional[str] \
                  ='[%(name)s] Line %(lineno)d %(levelname)s: %(message)s') -> None:
        
        '''
        Initializes name, level, and handler
        
        Attributes
        ----------
        name : str
            The logger name, prepends all logging errors
        logLevel : LogLevel
            The log level for the Arkouda logger, defaults to INFO
        handler : Handler
            The Python Handler object that defines where and how to log messages, 
            defaults to StreamHandler
        logFormat : str
            Defines the string template used to format all log messages, defaults
            to '[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'
            
        Return
        ------
        None
        '''
        Logger.__init__(self, name=name, level=level.value)
        self.logLevel = level
        if logFormat != '':
            handler.setFormatter(Formatter(logFormat))    
        self.addHandler(handler)