"""
Client-side message protocol classes for Arkouda server communication.

The `arkouda.message` module defines the classes and enums used to format,
serialize, and deserialize messages exchanged between the Arkouda client and server.
These classes encapsulate both command requests and reply messages, and provide
tools for converting Python objects to a format compatible with the Chapel server.

Classes
-------
ParameterObject
    Represents a typed key-value parameter used in Arkouda command messages.
    Provides factory methods for constructing parameters from a wide variety
    of Arkouda types (e.g., `pdarray`, `Strings`, `SegArray`, scalars, lists, etc.).

MessageFormat (Enum)
    Specifies the format of a message being sent (`STRING` or `BINARY`).

MessageType (Enum)
    Classifies the type of message returned by the server (`NORMAL`, `WARNING`, or `ERROR`).

RequestMessage
    Dataclass encapsulating the client-to-server command structure, including user info,
    command name, arguments, format, and parameter count.

ReplyMessage
    Dataclass encapsulating the server's response message, including status, message body,
    and originating user.

Key Features
------------
- Serialization of heterogeneous argument structures (scalars, arrays, nested dicts)
- Explicit typing and metadata for Chapel compatibility
- Structured error handling and deserialization
- Uses `__slots__` and `@dataclass(frozen=True)` for performance and immutability

Examples
--------
>>> from arkouda.message import ParameterObject
>>> import arkouda as ak
>>> arr = ak.array([1, 2, 3])
>>> param = ParameterObject.factory("x", arr)
>>> param.dict  # doctest: +SKIP
{'key': 'x', 'dtype': 'int64', 'val': 'id_gHZzPBV_1'}

>>> from arkouda.message import RequestMessage
>>> msg = RequestMessage(user="user", cmd="add", args="x=1;y=2", format=MessageFormat.STRING)
>>> msg.asdict()
{'user': 'user', 'token': '', 'cmd': 'add', 'format': 'STRING', 'args': 'x=1;y=2', 'size': -1}

Notes
-----
These classes are primarily used internally by Arkouda's `generic_msg()` mechanism and are not
typically used directly by end users.

See Also
--------
- arkouda.client.generic_msg
- arkouda.pdarray

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Dict, Optional

from typeguard import typechecked

from arkouda.numpy.dtypes import isSupportedNumber, resolve_scalar_dtype


__all__ = [
    "MessageFormat",
    "MessageType",
    "ParameterObject",
    "ReplyMessage",
    "RequestMessage",
]


class ParameterObject:
    """
    Represents a typed key-value parameter used in Arkouda command messages.

    This class is responsible for wrapping Python values, such as pdarrays,
    Strings, SegArrays, and scalars, into a common structure that can be
    serialized and passed to the Arkouda server.

    Attributes
    ----------
    key : str
        The name of the parameter.
    dtype : str
        A string representation of the type of the value.
    val : str
        A string-encoded representation of the value.

    See Also
    --------
    ParameterObject.factory : Main method for constructing ParameterObjects

    """

    __slots__ = ("key", "dtype", "val")

    key: str
    dtype: str
    val: str

    def __init__(self, key, dtype, val):
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "val", val)

    @property
    def dict(self):
        """
        Return the dictionary representation of the ParameterObject.

        Returns
        -------
        dict
            A dictionary with keys 'key', 'dtype', and 'val'.

        """
        return {
            "key": self.key,
            "dtype": self.dtype,
            "val": self.val,
        }

    @staticmethod
    @typechecked
    def _build_pdarray_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a pdarray value.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            pdarray object to load from the symbol table

        Returns
        -------
        ParameterObject

        """
        return ParameterObject(key, str(val.dtype), val.name)

    @staticmethod
    @typechecked
    def _build_sparray_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a sparray value.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            sparray object to load from the symbol table

        Returns
        -------
        ParameterObject

        """
        return ParameterObject(key, str(val.dtype), val.name)

    @staticmethod
    @typechecked
    def _build_strings_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a Strings value.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            Strings object to load from the symbol table

        Returns
        -------
        ParameterObject

        """
        # empty string if name of String obj is none
        name = val.name if val.name else ""
        return ParameterObject(key, "str", name)

    @staticmethod
    @typechecked
    def _build_segarray_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a SegArray value.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            SegArray object to load from the symbol table

        Returns
        -------
        ParameterObject

        """
        data = json.dumps({"segments": val.segments.name, "values": val.values.name})
        return ParameterObject(key, str(val.values.dtype), data)

    @staticmethod
    def _is_supported_value(val):
        import builtins

        import numpy as np

        return isinstance(val, (str, np.str_, builtins.bool, np.bool_)) or isSupportedNumber(val)

    @staticmethod
    def _format_param(p):
        from arkouda.numpy.segarray import SegArray

        return (
            json.dumps({"segments": p.segments.name, "values": p.values.name})
            if isinstance(p, SegArray)
            else p.name
        )

    @staticmethod
    @typechecked
    def _build_tuple_param(key: str, val: tuple) -> ParameterObject:
        """
        Create a ParameterObject from a tuple.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val : tuple
            tuple object to format as string

        Returns
        -------
        ParameterObject

        """
        return ParameterObject._build_list_param(key, list(val))

    @staticmethod
    @typechecked
    def _build_list_param(key: str, val: list) -> ParameterObject:
        """
        Create a ParameterObject from a list.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val : list
            list object to format as string

        Returns
        -------
        ParameterObject

        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.numpy.segarray import SegArray
        from arkouda.numpy.strings import Strings

        # want the object type. If pdarray the content dtypes can vary
        dtypes = {
            resolve_scalar_dtype(p) if ParameterObject._is_supported_value(p) else type(p).__name__
            for p in val
        }
        if len(dtypes) == 1:
            t = dtypes.pop()
        else:
            for p in val:
                if not (
                    isinstance(p, (pdarray, Strings, SegArray)) or ParameterObject._is_supported_value(p)
                ):
                    raise TypeError(
                        f"List parameters must be pdarray, Strings, SegArray, str or a type "
                        f"that inherits from the aforementioned. {type(p).__name__} "
                        f"does not meet that criteria."
                    )
            t = "mixed"
        data = [
            str(p) if ParameterObject._is_supported_value(p) else ParameterObject._format_param(p)
            for p in val
        ]
        return ParameterObject(key, t, json.dumps(data))

    @staticmethod
    @typechecked
    def _build_dict_param(key: str, val: Dict) -> ParameterObject:
        j = []
        for k, v in val.items():
            if not isinstance(k, str):
                raise TypeError(f"Argument keys are required to be str. Found {type(k)}")
            param = ParameterObject.factory(k, v)
            j.append(json.dumps(param.dict))
        return ParameterObject(key, str(dict.__name__), json.dumps(j))

    @staticmethod
    @typechecked
    def _build_gen_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a single value.

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            singular value to use. This could be str, int, float, etc

        Returns
        -------
        ParameterObject

        """
        v = val if isinstance(val, str) else str(val)
        return ParameterObject(key, resolve_scalar_dtype(val), v)

    @staticmethod
    def generate_dispatch() -> Dict:
        """
        Build and return the dispatch table used to build parameter object.

        Returns
        -------
        Dictionary - mapping the parameter type to the build function

        """
        from arkouda.numpy.segarray import SegArray
        from arkouda.numpy.strings import Strings

        return {
            Strings.__name__: ParameterObject._build_strings_param,
            SegArray.__name__: ParameterObject._build_segarray_param,
            list.__name__: ParameterObject._build_list_param,
            dict.__name__: ParameterObject._build_dict_param,
            tuple.__name__: ParameterObject._build_tuple_param,
        }

    @classmethod
    def factory(cls, key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a key–value pair for server communication.

        This factory method selects the appropriate builder based on the type of
        `val`. It handles `pdarray` and `sparray` specially to avoid duplicate
        dispatch entries, and falls back to a generic parameter builder for other
        types.

        Parameters
        ----------
        key : str
            The name of the parameter.
        val : Any
            The value of the parameter. Supported types include:
            - `pdarray`: constructs a pdarray parameter
            - `sparray`: constructs a sparse array parameter
            - Other types via the generic parameter builder

        Returns
        -------
        ParameterObject
            A `ParameterObject` formatted for the Chapel server.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.message import ParameterObject
        >>> arr = ak.array([1, 2, 3])
        >>> param = ParameterObject.factory("my_array", arr)
        >>> isinstance(param, ParameterObject)
        True

        """
        from arkouda.numpy.pdarrayclass import pdarray
        from arkouda.scipy.sparrayclass import sparray

        dispatch = ParameterObject.generate_dispatch()
        if isinstance(
            val, pdarray
        ):  # this is done here to avoid multiple dispatch entries for the same type
            return cls._build_pdarray_param(key, val)
        elif isinstance(
            val, sparray
        ):  # this is done here to avoid multiple dispatch entries for the same type
            return cls._build_sparray_param(key, val)
        elif (f := dispatch.get(type(val).__name__)) is not None:
            return f(key, val)
        else:
            return ParameterObject._build_gen_param(key, val)


class MessageFormat(Enum):
    """
    Enum representing the format of an Arkouda message.

    Used to distinguish between STRING and BINARY format messages
    exchanged between client and server.
    """

    STRING = "STRING"
    BINARY = "BINARY"

    def __str__(self) -> str:
        """
        Return value.

        Overridden method returns value, which is useful in outputting a MessageFormat object to JSON.

        """
        return self.value

    def __repr__(self) -> str:
        """
        Return value.

        Overridden method returns value, which is useful in outputting a MessageFormat object to JSON.

        """
        return self.value


class MessageType(Enum):
    """
    Enum representing the type of server response message.

    Values indicate whether a server message is NORMAL, a WARNING, or an ERROR.

    """

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ERROR = "ERROR"

    def __str__(self) -> str:
        """Return value, which is useful in outputting a MessageType object to JSON (override)."""
        return self.value

    def __repr__(self) -> str:
        """Return value, which is useful in outputting a MessageType object to JSON (override)."""
        return self.value


@dataclass(frozen=True)
class RequestMessage:
    """
    Encapsulates client-to-server command messages.

    Represents a structured message used to communicate commands from
    the Arkouda Python client to the Chapel server.

    Attributes
    ----------
    user : str
        The name of the requesting user.
    token : str
        The user's session token.
    cmd : str
        The command to execute.
    format : MessageFormat
        The format of the message (STRING or BINARY).
    args : str
        The argument string passed to the command.
    size : str
        Size of the parameter payload, -1 if unknown.

    """

    __slots__ = ("user", "token", "cmd", "format", "args", "size")

    user: str
    token: str
    cmd: str
    format: MessageFormat
    args: str
    size: str

    def __init__(
        self,
        user: str,
        cmd: str,
        token: Optional[str] = None,
        format: MessageFormat = MessageFormat.STRING,
        args: Optional[str] = None,
        size: int = -1,
    ) -> None:
        """
        Initiate request message.

        Overridden __init__ method sets instance attributes to default values
        if the corresponding init params are missing.

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
        size : int
            Value indicating the number of parameters in args
            -1 if args is not json

        """
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "cmd", cmd)
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "size", size)

    def asdict(self) -> Dict:
        """
        Return a dictionary encapsulating ReplyMessage state.

        Overridden asdict implementation sets the values of non-required
        fields to an empty space (for Chapel JSON processing) and invokes
        str() on the format instance attribute.

        Returns
        -------
        Dict
            A dict object encapsulating ReplyMessage state

        """
        # args and token logic will not be needed once Chapel supports nulls
        args = self.args if self.args else ""
        token = self.token if self.token else ""

        return {
            "user": self.user,
            "token": token,
            "cmd": self.cmd,
            "format": str(self.format),
            "args": args,
            "size": self.size,
        }


@dataclass(frozen=True)
class ReplyMessage:
    """
    Encapsulates server-to-client reply messages.

    Represents the result of a command sent to the server, including
    message type and body.

    Attributes
    ----------
    msg : str
        The message body returned from the server.
    msgType : MessageType
        Type of the message (e.g., NORMAL, WARNING, ERROR).
    user : str
        The user to whom the message is addressed.

    """

    __slots__ = ("msg", "msgType", "user")

    msg: str
    msgType: MessageType
    user: str

    @staticmethod
    def fromdict(values: Dict) -> ReplyMessage:
        """
        Generate a ReplyMessage from a dictionary.

        Generate a ReplyMessage from a dict encapsulating the data
        and metadata from a reply from the Arkouda server.

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
            return ReplyMessage(
                msg=values["msg"], msgType=MessageType(values["msgType"]), user=values["user"]
            )
        except KeyError as ke:
            raise ValueError(f"values dict missing {ke} field")
