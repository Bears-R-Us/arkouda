import os
from typing import cast, List, Optional
from logging import Logger, Formatter, Handler, StreamHandler, DEBUG, \
     INFO, WARN, ERROR, CRITICAL
from enum import Enum
from typeguard import typechecked

__all__ = ['enableVerbose', 'disableVerbose']

loggers = {}

"""
The LogLevel enum class defines the log levels for the ArkoudaLogger, the
purpose of which is to provide controlled vocabulary for setting the log
levels for ArkoudaLogger.
"""
class LogLevel(Enum):
    
    DEBUG = 'DEBUG'
    CRITICAL = 'CRITICAL'
    INFO = 'INFO'
    WARN = 'WARN'
    ERROR =  'ERROR'


"""
ArkoudaLogger encapsulates logging configuration and logic to log messages
at varying levels including debug, info, critical, warn, and error

    Attributes
    ----------
    name : str
        The logger name, prepends all logging errors
    level : int
        The log level for the Arkouda logger, defaults to DEBUG to enable
        fine-grained logging control within individual handlers
    handlers : List[Handler]
        List of 1..n logging.Handler objects that define where and how to log
        messages, defaults to list containing a single StreamHandler object
"""  
class ArkoudaLogger(Logger):
    
    DEFAULT_LOG_FORMAT = '[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'
    
    CLIENT_LOG_FORMAT = ''
    
    levelMappings = {
        LogLevel.DEBUG: DEBUG,
        LogLevel.INFO: INFO,
        LogLevel.WARN: WARN,
        LogLevel.ERROR: ERROR,
        LogLevel.CRITICAL: CRITICAL
    }
    
    @typechecked
    def __init__(self, name : str, logLevel : LogLevel=LogLevel.INFO, 
                      handlers : Optional[List[Handler]]=None, 
                      logFormat : Optional[str] \
             ='[%(name)s] Line %(lineno)d %(levelname)s: %(message)s') -> None:
        
        '''
        Initializes the ArkoudaLogger with the name, level, logFormat, and
        handlers parameters
        
        Parameters
        ----------
        name : str
            The logger name, prepends all logging errors
        logLevel : LogLevel
            The desired log level in the form of a LogLevel enum value, defaults
            to INFO
        handlers : List[Handler]
            A list of logging.Handler objects, if None, a list consisting of 
            one StreamHandler named 'console-handler' is generated and configured
        logFormat : str
            Defines the string template used to format all log messages,
            defaults to '[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'
            
        Return
        ------
        None
        
        Raises
        ------
        TypeError
            Raised if name or logFormat is not a str, logLevel is not a LogLevel
            enum, or handlers is not a list of str objects
        
        Notes
        -----
        ArkoudaLogger is derived from logging.Logger and adds convenience methods
        and enum-enforced parameter control to prevent runtime errors.
        
        The default list of Handler objects consists of a single a StreamHandler 
        that writes log messages to stdout with a format that outputs a message 
        such as the following: 
            [LoggerTest] Line 24 DEBUG: debug message
        
        Important note: if a list of 1..n logging.Handler objects is passed in, and
        dynamic changes to 1..n handlers is desired, set a name for each Handler
        object as follows: handler.name = <desired name>. Setting the Handler names 
        will enable dynamic changes to specific Handlers be retrieving by name the
        Handler object to be updated.
        
        The Logger-scoped level is set to DEBUG to enable fine-grained control at
        the Handler level as a higher level would disable DEBUG-level logging in 
        the individual handlers.
        '''
        Logger.__init__(self, name=name, level=DEBUG)
        if handlers is None:
            handler = cast(Handler,StreamHandler())
            handler.name = 'console-handler'
            handler.setLevel(logLevel.value)
            handlers = [handler]
        for handler in handlers: 
            if logFormat:
                handler.setFormatter(Formatter(logFormat))  
            self.addHandler(handler)
    
    @typechecked  
    def changeLogLevel(self, level : LogLevel, 
                       handlerNames : List[str]=None) -> None:
        """
        Dynamically changes the logging level for ArkoudaLogger and 1..n
        of configured Handlers
        
        Parameters
        ----------
        level : LogLevel
            The desired log level in the form of a LogLevel enum value
        handlerNames : List[str]
            Names of 1..n Handlers configured for the ArkoudaLogger that
            the log level will be changed for.
            
        Returns
        -------
        None
        
        Raises
        ------
        TypeError
            Raised if level is not a LogLevel enum or if handlerNames is
            not a list of str objects
        
        Notes
        -----
        The default is to change the log level of all configured Handlers. 
        If the handlerNames list is not None, then only the log level of 
        the named Handler object is changed.
        """
        newLevel = ArkoudaLogger.levelMappings[level]
        if handlerNames is None:
            # No handler names supplied, so setLevel for all handlers
            for handler in self.handlers:
                handler.setLevel(newLevel)
        else:
            # setLevel for the named handlers
            for name,handler in zip(handlerNames, self.handlers):
                if name == handler.name:
                    handler.setLevel(newLevel)
                    
    def enableVerbose(self) -> None:
        """
        Enables verbose output by setting the log level for all handlers 
        to DEBUG
        
        Returns
        -------
        None
        """
        self.changeLogLevel(LogLevel.DEBUG)
    
    @typechecked   
    def disableVerbose(self, logLevel : LogLevel=LogLevel.INFO) -> None:
        """
        Disables verbose output by setting the log level for all handlers 
        to a level other than DEBUG, with a default of INFO
        
        Parameters
        ----------
        logLevel : LogLevel, defaults to LogLevel.INFO
            The desired log level that will disable verbose output (logging at 
            the DEBUG level) by resetting the log level for all handlers.
            
        Raises
        ------
        TypeError
            Raised if logLevel is not a LogLevel enum
        
        Returns
        -------
        None
        """        
        self.changeLogLevel(logLevel)
    
    @typechecked
    def getHandler(self, name : str) -> Handler:
        """
        Retrieves the Handler object corresponding to the name.
        
        Parameters
        ----------
        name : str
            The name to be used to retrieve the desired Handler
            
        Returns
        -------
        Handler with matching Handler.name
        
        Raises
        ------
        TypeError
            Raised if the name is not a str
        ValueError
            Raised if the name does not match the name of any
            of the configured handlers
        """
        for handler in self.handlers:
            if name == handler.name:
                return handler
        raise ValueError('The name {} does not match any handler'.format(name))

@typechecked        
def getArkoudaLogger(name : str, handlers : Optional[List[Handler]]=None, 
                     logFormat : Optional[str]= ArkoudaLogger.DEFAULT_LOG_FORMAT,
                     logLevel : LogLevel=None) -> ArkoudaLogger:
    """
    A convenience method for instantiating an ArkoudaLogger that retrieves the 
    logging level from the ARKOUDA_LOG_LEVEL env variable
    
    Parameters
    ----------
    name : str
        The name of the ArkoudaLogger
    handlers : List[Handler]
        A list of logging.Handler objects, if None, a list consisting of 
        one StreamHandler named 'console-handler' is generated and configured
    logFormat : str
        The format for log messages, defaults to the following format:
        '[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'
    
    Returns
    -------
    ArkoudaLogger
    
    Raises
    ------
    TypeError
        Raised if either name or logFormat is not a str object or if handlers
        is not a list of str objects
    
    Notes
    -----
    Important note: if a list of 1..n logging.Handler objects is passed in, and
    dynamic changes to 1..n handlers is desired, set a name for each Handler
    object as follows: handler.name = <desired name>, which will enable retrieval
    and updates for the specified handler.
    """
    if not logLevel:
        logLevel = LogLevel(os.getenv('ARKOUDA_LOG_LEVEL', LogLevel('INFO')))
    
    logger = ArkoudaLogger(name=name, handlers=handlers, logFormat=logFormat, 
                logLevel=logLevel)
    loggers[logger.name] = logger
    return logger

@typechecked   
def getArkoudaClientLogger(name : str) -> ArkoudaLogger:
    """
    A convenience method for instantiating an ArkoudaLogger that retrieves the 
    logging level from the ARKOUDA_LOG_LEVEL env variable and outputs log
    messages without any formatting to stdout. 
    
    Parameters
    ----------
    name : str
        The name of the ArkoudaLogger
    
    Returns
    -------
    ArkoudaLogger
    
    Raises
    ------
    TypeError
        Raised if the name is not a str
    
    Notes
    -----
    The returned ArkoudaLogger is configured to write unformatted log messages to
    stdout, making it suitable for logging messages users will see such as 
    confirmation of successful login or pdarray creation
    """
    return getArkoudaLogger(name=name, logFormat=ArkoudaLogger.CLIENT_LOG_FORMAT)

def enableVerbose() -> None:
    """
    Enables verbose logging (DEBUG log level) for all ArkoudaLoggers
    """
    for logger in loggers.values():
        logger.enableVerbose()

@typechecked        
def disableVerbose(logLevel : LogLevel=LogLevel.INFO) -> None:
    """
    Disables verbose logging (DEBUG log level) for all ArkoudaLoggers, setting
    the log level for each to the logLevel parameter
    
    Parameters
    ----------
    logLevel : LogLevel
        The new log level, defaultts to LogLevel.INFO
        
    Raises
    ------
    TypeError
        Raised if logLevel is not a LogLevel enum
    """
    for logger in loggers.values():
        logger.disableVerbose(logLevel)
