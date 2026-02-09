"""
Logging utilities for Arkouda client operations.

The `arkouda.logger` module provides an extensible, configurable logging system tailored
to Arkouda's Python client. It supports structured logging using the standard `logging`
module with added conveniences, such as type-safe log level enums, named handlers,
and global verbosity toggles.

Main Features
-------------
- `ArkoudaLogger`: A subclass of `logging.Logger` with Arkouda-specific defaults and
  dynamic handler configuration.
- `LogLevel`: Enum of supported logging levels (`DEBUG`, `INFO`, `WARN`, etc.)
- Global registry of loggers for coordinated verbosity control
- Utility methods for enabling/disabling verbose output globally
- Client-side custom log injection into the Arkouda server logs via `write_log`

Classes
-------
LogLevel : Enum
    Enum for defining log levels in a type-safe way (`DEBUG`, `INFO`, `WARN`, etc.).

ArkoudaLogger : Logger
    A wrapper around Python's standard `Logger` that adds Arkouda-specific conventions,
    log formatting, and runtime handler reconfiguration.

Functions
---------
get_arkouda_logger(name, handlers=None, log_format=None, log_level=None)
    Instantiate a logger with customizable format and log level.

get_arkouda_client_logger(name)
    Instantiate a logger for client-facing output (no formatting, INFO level default).

enable_verbose()
    Globally set all ArkoudaLoggers to DEBUG level.

disable_verbose(log_level=LogLevel.INFO)
    Globally disable DEBUG output by setting all loggers to the specified level.

write_log(log_msg, tag="ClientGeneratedLog", log_lvl=LogLevel.INFO)
    Submit a custom log message to the Arkouda server’s logging system.

Usage Example
-------------
>>> from arkouda.logger import get_arkouda_logger, LogLevel
>>> logger = get_arkouda_logger("myLogger")
>>> logger.info("This is an info message.")
>>> logger.enable_verbose()
>>> logger.debug("Now showing debug messages.")

See Also
--------
- logging (Python Standard Library)
- arkouda.client.generic_msg

"""

import os
import warnings

from dataclasses import dataclass
from enum import Enum
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARN,
    Formatter,
    Handler,
    Logger,
    StreamHandler,
)
from typing import Any, Final, List, Optional, Union, cast

from typeguard import typechecked


__all__ = [
    "LogLevel",
    "enable_verbose",
    "disable_verbose",
    "write_log",
    "enableVerbose",
    "disableVerbose",
]

loggers = {}


class LogLevel(Enum):
    """
    Enum for defining valid log levels used by ArkoudaLogger.

    Members
    -------
    INFO : str
        Confirmation that things are working as expected.
    DEBUG : str
        Detailed information, typically of interest only when diagnosing problems.
    WARN : str
        An indication that something unexpected happened, or indicative of some problem.
    ERROR : str
        A more serious problem, the software has not been able to perform some function.
    CRITICAL : str
        An extremely serious error, indicating the program itself may be unable to continue.

    Notes
    -----
    This enum provides a controlled vocabulary for setting log levels on ArkoudaLogger
    instances. These are mapped internally to the standard Python `logging` levels.

    """

    DEBUG = "DEBUG"
    CRITICAL = "CRITICAL"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


"""
ArkoudaLogger encapsulates logging configuration and logic to log messages
at varying levels including debug, info, critical, warn, and error.

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


# sentinel

_UnsetType = object  # just for readability in the union


@dataclass(frozen=True)
class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


class ArkoudaLogger(Logger):
    DEFAULT_LOG_FORMAT = "[%(name)s] Line %(lineno)d %(levelname)s: %(message)s"

    CLIENT_LOG_FORMAT = ""

    levelMappings = {
        LogLevel.DEBUG: DEBUG,
        LogLevel.INFO: INFO,
        LogLevel.WARN: WARN,
        LogLevel.ERROR: ERROR,
        LogLevel.CRITICAL: CRITICAL,
    }

    @typechecked
    def __init__(
        self,
        name: str,
        log_level: Union[LogLevel, _UnsetType] = _UNSET,  # sentinel means "not provided"
        handlers: Optional[List[Handler]] = None,
        log_format: Optional[str] = "[%(name)s] Line %(lineno)d %(levelname)s: %(message)s",
        **kwargs,
    ) -> None:
        """
        Initialize the ArkoudaLogger with the name, level, log_format, and handlers parameters.

        Parameters
        ----------
        name : str
            The logger name, prepends all logging errors
        log_level : LogLevel
            The desired log level in the form of a LogLevel enum value, defaults
            to INFO
        handlers : List[Handler]
            A list of logging.Handler objects, if None, a list consisting of
            one StreamHandler named 'console-handler' is generated and configured
        log_format : str
            Defines the string template used to format all log messages,
            defaults to '[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'

        Return
        ------
        None

        Raises
        ------
        TypeError
            Raised if name or log_format is not a str, log_level is not a LogLevel
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

        """
        # --- log_level alias handling ---
        if "logLevel" in kwargs:
            if log_level is not _UNSET:
                raise TypeError("Pass only one of 'logLevel' or 'log_level'")
            warnings.warn(
                "'logLevel' is deprecated; use 'log_level' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            log_level = kwargs.pop("logLevel")
        elif log_level is _UNSET:
            log_level = LogLevel.INFO

        # --- log_format alias handling ---
        if "logFormat" in kwargs:
            if log_format is not _UNSET:
                raise TypeError("Pass only one of 'logFormat' or 'log_format'")
            warnings.warn(
                "'logFormat' is deprecated; use 'log_format' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            log_format = kwargs.pop("logFormat")
        elif log_format is _UNSET:
            log_format = "[%(name)s] Line %(lineno)d %(levelname)s: %(message)s"

        if not isinstance(log_level, LogLevel):
            raise TypeError("log_level must be a LogLevel")

        if not isinstance(log_format, str):
            raise TypeError("log_format must be a str")

        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

        Logger.__init__(self, name=name, level=DEBUG)
        if handlers is None:
            handler = cast(Handler, StreamHandler())
            handler.name = "console-handler"
            handler.setLevel(log_level.value)
            handlers = [handler]
        for handler in handlers:
            if log_format:
                handler.setFormatter(Formatter(log_format))
            self.addHandler(handler)

    @typechecked
    def change_log_level(self, level: LogLevel, handlerNames: Optional[List[str]] = None) -> None:
        """
        Dynamically changes the logging level for ArkoudaLogger and 1..n of configured Handlers.

        Parameters
        ----------
        level : LogLevel
            The desired log level in the form of a LogLevel enum value
        handlerNames : List[str]
            Names of 1..n Handlers configured for the ArkoudaLogger that
            the log level will be changed for.

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
        new_level = ArkoudaLogger.levelMappings[level]
        if handlerNames is None:
            # No handler names supplied, so setLevel for all handlers
            for handler in self.handlers:
                handler.setLevel(new_level)
        else:
            # setLevel for the named handlers
            for name, handler in zip(handlerNames, self.handlers):
                if name == handler.name:
                    handler.setLevel(new_level)

    def changeLogLevel(self, *args: Any, **kwargs: Any) -> None:
        """
        Deprecated alias for :meth:`change_log_level`.

        This method exists for backward compatibility only. Use
        :meth:`change_log_level` instead.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`change_log_level`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`change_log_level`.

        Returns
        -------
        None

        See Also
        --------
        change_log_level : Preferred replacement.
        """
        warnings.warn(
            "changeLogLevel is deprecated; use change_log_level",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.change_log_level(*args, **kwargs)

    def enable_verbose(self) -> None:
        """Enable verbose output by setting the log level for all handlers to DEBUG."""
        self.change_log_level(LogLevel.DEBUG)

    def enableVerbose(self):
        """
        Deprecated alias for :meth:`enable_verbose`.

        This method exists for backward compatibility only. Use
        :meth:`enable_verbose` instead.

        Returns
        -------
        None

        Warns
        -----
        DeprecationWarning
            Always raised. ``enableVerbose`` is deprecated and will be removed
            in a future release.

        See Also
        --------
        enable_verbose : Preferred replacement.
        """
        warnings.warn(
            "enableVerbose is deprecated; use enable_verbose",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.enable_verbose()

    @typechecked
    def disable_verbose(self, log_level: LogLevel | _Unset = _UNSET, **kwargs) -> None:
        """
        Disables verbose output.

        Disables verbose output by setting the log level for all handlers
        to a level other than DEBUG, with a default of INFO

        Parameters
        ----------
        log_level : LogLevel, defaults to LogLevel.INFO
            The desired log level that will disable verbose output (logging at
            the DEBUG level) by resetting the log level for all handlers.

        Raises
        ------
        TypeError
            Raised if log_level is not a LogLevel enum
        """
        if "logLevel" in kwargs:
            if not isinstance(log_level, _Unset):
                raise TypeError("Pass only one of 'logLevel' or 'log_level'")
            warnings.warn(
                "'logLevel' is deprecated; use 'log_level' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            log_level = kwargs.pop("logLevel")
        elif isinstance(log_level, _Unset):
            log_level = LogLevel.INFO

        if kwargs:
            raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

        if not isinstance(log_level, LogLevel):
            raise TypeError("log_level must be a LogLevel")

        self.change_log_level(log_level)

    def disableVerbose(self, logLevel: LogLevel = LogLevel.INFO):
        """
        Deprecated alias for :meth:`disable_verbose`.

        This method exists for backward compatibility only. Use
        :meth:`disable_verbose` instead.

        Parameters
        ----------
        logLevel : LogLevel, default LogLevel.INFO
            The log level to set for all handlers, disabling verbose
            (DEBUG-level) output.

        Returns
        -------
        None

        Warns
        -----
        DeprecationWarning
            Always raised. ``disableVerbose`` is deprecated and will be removed
            in a future release.

        See Also
        --------
        disable_verbose : Preferred replacement.
        enable_verbose : Enable verbose (DEBUG-level) output.
        """
        warnings.warn(
            "disableVerbose is deprecated; use disable_verbose",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.disable_verbose(logLevel)

    @typechecked
    def get_handler(self, name: str) -> Handler:
        """
        Retrieve the Handler object corresponding to the name.

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
        raise ValueError(f"The name {name} does not match any handler")

    def getHandler(self, name: str):
        """
        Deprecated alias for :meth:`get_handler`.

        This method exists for backward compatibility only. Use
        :meth:`get_handler` instead.

        Parameters
        ----------
        name : str
            The name of the handler to retrieve.

        Returns
        -------
        Handler
            The handler with the matching ``Handler.name``.

        Raises
        ------
        TypeError
            Raised if ``name`` is not a string.
        ValueError
            Raised if no configured handler matches ``name``.

        Warns
        -----
        DeprecationWarning
            Always raised. ``getHandler`` is deprecated and will be removed
            in a future release.

        See Also
        --------
        get_handler : Preferred replacement.
        """
        warnings.warn(
            "getHandler is deprecated; use get_handler",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_handler(name)


@typechecked
def get_arkouda_logger(
    name: str,
    handlers: Optional[List[Handler]] = None,
    log_format: Optional[str] = None,  # <-- key change
    log_level: Optional[LogLevel] = None,  # keep as-is (env fallback)
    **kwargs,
) -> ArkoudaLogger:
    if "logFormat" in kwargs:
        # If caller passed both names, always error (regardless of value)
        if log_format is not None:
            raise TypeError("Pass only one of 'logFormat' or 'log_format'")
        warnings.warn(
            "'logFormat' is deprecated; use 'log_format' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_format = kwargs.pop("logFormat")

    if "logLevel" in kwargs:
        if log_level is not None:
            raise TypeError("Pass only one of 'logLevel' or 'log_level'")
        warnings.warn(
            "'logLevel' is deprecated; use 'log_level' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        log_level = kwargs.pop("logLevel")

    if kwargs:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(kwargs)}")

    # Resolve defaults BEFORE validation
    if log_format is None:
        log_format = ArkoudaLogger.DEFAULT_LOG_FORMAT

    if log_level is None:
        env_val = os.getenv("ARKOUDA_LOG_LEVEL", "INFO")
        try:
            log_level = LogLevel(env_val.upper())
        except Exception as e:
            raise ValueError(f"Invalid ARKOUDA_LOG_LEVEL={env_val!r}") from e

    # Validate after defaults
    if not isinstance(log_format, str):
        raise TypeError("log_format must be a str")
    if not isinstance(log_level, LogLevel):
        raise TypeError("log_level must be a LogLevel")

    logger = ArkoudaLogger(
        name=name,
        handlers=handlers,
        log_format=log_format,
        log_level=log_level,
    )
    loggers[logger.name] = logger
    return logger


def getArkoudaLogger(
    name: str,
    handlers: Optional[List[Handler]] = None,
    logFormat: Optional[str] = ArkoudaLogger.DEFAULT_LOG_FORMAT,
    logLevel: Optional[LogLevel] = None,
):
    """
    Deprecated alias for :func:`get_arkouda_logger`.

    This function exists for backward compatibility only. Use
    :func:`get_arkouda_logger` instead.

    Parameters
    ----------
    name : str
        The name of the ArkoudaLogger.
    handlers : list of logging.Handler, optional
        A list of handlers to attach to the logger.
    logFormat : str, optional
        The log message format string.
    logLevel : LogLevel, optional
        The logging level to use. If not provided, the value is
        read from the ``ARKOUDA_LOG_LEVEL`` environment variable.

    Returns
    -------
    ArkoudaLogger
        The configured ArkoudaLogger instance.

    Warns
    -----
    DeprecationWarning
        Always raised. ``getArkoudaLogger`` is deprecated and will be removed
        in a future release.

    See Also
    --------
    get_arkouda_logger : Preferred replacement.
    """
    warnings.warn(
        "getArkoudaLogger is deprecated; use get_arkouda_logger",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_arkouda_logger(
        name=name,
        handlers=handlers,
        logFormat=logFormat,
        logLevel=logLevel,
    )


@typechecked
def get_arkouda_client_logger(name: str) -> ArkoudaLogger:
    """
    Instantiate an ArkoudaLogger that retrieves the logging level from ARKOUDA_LOG_LEVEL env variable.

    Instantiate an ArkoudaLogger that retrieves the
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
    return get_arkouda_logger(name=name, log_format=ArkoudaLogger.CLIENT_LOG_FORMAT)


def getArkoudaClientLogger(name: str):
    """
    Deprecated alias for :func:`get_arkouda_client_logger`.

    This function exists for backward compatibility only. Use
    :func:`get_arkouda_client_logger` instead.

    Parameters
    ----------
    name : str
        The name of the ArkoudaLogger.

    Returns
    -------
    ArkoudaLogger
        A logger configured to emit unformatted messages to stdout.

    Warns
    -----
    DeprecationWarning
        Always raised. ``getArkoudaClientLogger`` is deprecated and will be
        removed in a future release.

    See Also
    --------
    get_arkouda_client_logger : Preferred replacement.
    get_arkouda_logger : Base logger factory.
    """
    warnings.warn(
        "getArkoudaClientLogger is deprecated; use get_arkouda_client_logger",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_arkouda_client_logger(name=name)


def enable_verbose() -> None:
    """Enable verbose logging (DEBUG log level) for all ArkoudaLoggers."""
    for logger in loggers.values():
        logger.enable_verbose()


def enableVerbose():
    """
    Deprecated alias for :func:`enable_verbose`.

    This function exists for backward compatibility only. Use
    :func:`enable_verbose` instead.

    Returns
    -------
    None

    Warns
    -----
    DeprecationWarning
        Always raised. ``enableVerbose`` is deprecated and will be removed
        in a future release.

    See Also
    --------
    enable_verbose : Enable verbose (DEBUG-level) logging for all loggers.
    """
    warnings.warn(
        "enableVerbose is deprecated; use enable_verbose",
        DeprecationWarning,
        stacklevel=2,
    )
    return enable_verbose()


@typechecked
def disable_verbose(logLevel: LogLevel = LogLevel.INFO) -> None:
    """
    Disables verbose logging.

    Disables verbose logging (DEBUG log level) for all ArkoudaLoggers, setting
    the log level for each to the logLevel parameter.

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
        logger.disable_verbose(logLevel)


def disableVerbose(logLevel: LogLevel = LogLevel.INFO):
    """
    Deprecated alias for :func:`disable_verbose`.

    This function exists for backward compatibility only. Use
    :func:`disable_verbose` instead.

    Parameters
    ----------
    logLevel : LogLevel, default LogLevel.INFO
        The log level to apply to all ArkoudaLoggers, disabling
        verbose (DEBUG-level) output.

    Returns
    -------
    None

    Warns
    -----
    DeprecationWarning
        Always raised. ``disableVerbose`` is deprecated and will be removed
        in a future release.

    See Also
    --------
    disable_verbose : Disable verbose logging for all loggers.
    enable_verbose : Enable verbose logging for all loggers.
    """
    warnings.warn(
        "disableVerbose is deprecated; use disable_verbose",
        DeprecationWarning,
        stacklevel=2,
    )
    return disable_verbose(logLevel)


@typechecked
def write_log(log_msg: str, tag: str = "ClientGeneratedLog", log_lvl: LogLevel = LogLevel.INFO):
    """
    Allow the user to write custom logs.

    Parameters
    ----------
    log_msg: str
        The message to be added to the server log
    tag: str
        The tag to use in the log. This takes the place of the server function name.
        Allows for easy identification of custom logs.
        Defaults to "ClientGeneratedLog"
    log_lvl: LogLevel
        The type of log to be written
        Defaults to LogLevel.INFO

    See Also
    --------
    LogLevel

    """
    from arkouda.client import generic_msg

    generic_msg(cmd="clientlog", args={"log_msg": log_msg, "log_lvl": log_lvl.name, "tag": tag})
