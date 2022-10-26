from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from typeguard import typechecked


class ObjectType(Enum):
    """
    Class used for assigning object types in the JSON string
    sent to the server for processing
    """

    PDARRAY = "PDARRAY"
    STRINGS = "SEGSTRING"
    LIST = "LIST"
    DICT = "DICT"
    VALUE = "VALUE"
    DATETIME = "DATETIME"
    TIMEDELTA = "TIMEDELTA"

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


class ParameterObject:

    __slots = ("key", "objType", "dtype", "val")

    key: str
    objType: MessageFormat
    dtype: str
    val: str

    def __init__(self, key, objType, dtype, val):
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "objType", objType)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "val", val)

    @property
    def dict(self):
        return {
            "key": self.key,
            "objType": str(self.objType),
            "dtype": self.dtype,
            "val": self.val,
        }

    @staticmethod
    @typechecked
    def _build_pdarray_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a pdarray value

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            pdarray object ot load from the symbol table

        Returns
        -------
        ParameterObject
        """
        return ParameterObject(key, ObjectType.PDARRAY, str(val.dtype), val.name)

    @staticmethod
    @typechecked
    def _build_strings_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a Strings value

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            Strings object ot load from the symbol table

        Returns
        -------
        ParameterObject
        """
        # empty string if name of String obj is none
        name = val.name if val.name else ""
        return ParameterObject(key, ObjectType.STRINGS, "str", name)

    @staticmethod
    @typechecked
    def _build_list_param(key: str, val: list) -> ParameterObject:
        """
        Create a ParameterObject from a list

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
        from arkouda.pdarrayclass import pdarray
        from arkouda.strings import Strings

        # want the object type. If pdarray the content dtypes can vary
        dtypes = {type(p).__name__ for p in val}
        if len(dtypes) > 1:
            t_str = ", ".join(dtypes)
            raise TypeError(f"List values must be of the same type. Found {t_str}")
        t = dtypes.pop()
        if any(x == t for x in [pdarray.__name__, Strings.__name__]):
            return ParameterObject(key, ObjectType.LIST, t, json.dumps([x.name for x in val]))
        else:
            # need all values to be str for chapel to read list properly
            v = val if t == str.__name__ else [str(x) for x in val]
            return ParameterObject(key, ObjectType.LIST, t, json.dumps(v))

    @staticmethod
    @typechecked
    def _build_dict_param(key: str, val: Dict) -> ParameterObject:
        j = []
        for k, v in val.items():
            if not isinstance(k, str):
                raise TypeError(f"Argument keys are required to be str. Found {type(k)}")
            param = ParameterObject.factory(k, v)
            j.append(json.dumps(param.dict))
        return ParameterObject(key, ObjectType.DICT, str(dict.__name__), json.dumps(j))

    @staticmethod
    @typechecked
    def _build_gen_param(key: str, val) -> ParameterObject:
        """
        Create a ParameterObject from a single value

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
        return ParameterObject(key, ObjectType.VALUE, type(val).__name__, v)

    @staticmethod
    def generate_dispatch() -> Dict:
        """
        Builds and returns the dispatch table used to build parameter object.

        Returns
        -------
        Dictionary - mapping the parameter type to the build function
        """
        from arkouda.strings import Strings

        return {
            Strings.__name__: ParameterObject._build_strings_param,
            list.__name__: ParameterObject._build_list_param,
            dict.__name__: ParameterObject._build_dict_param,
        }

    @classmethod
    def factory(cls, key: str, val) -> ParameterObject:
        """
        Factory method used to build ParameterObject given a key value pair

        Parameters
        ----------
        key : str
            key from the dictionary object
        val
            the value corresponding to the provided key from the dictionary

        Returns
        --------
        ParameterObject - The parameter object formatted to be parsed by the chapel server
        """
        from arkouda.pdarrayclass import pdarray

        dispatch = ParameterObject.generate_dispatch()
        if isinstance(
            val, pdarray
        ):  # this is done here to avoid multiple dispatch entries for the same type
            return cls._build_pdarray_param(key, val)
        elif (f := dispatch.get(type(val).__name__)) is not None:
            return f(key, val)
        else:
            return ParameterObject._build_gen_param(key, val)


"""
The MessageFormat enum provides controlled vocabulary for the message
format which can be either a string or a binary (bytes) object.
"""


class MessageFormat(Enum):
    STRING = "STRING"
    BINARY = "BINARY"

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
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ERROR = "ERROR"

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
class RequestMessage:

    __slots = ("user", "token", "cmd", "format", "args", "size")

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
        token: str = None,
        format: MessageFormat = MessageFormat.STRING,
        args: str = None,
        size: int = -1,
    ) -> None:
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
        size : int
            Value indicating the number of parameters in args
            -1 if args is not json

        Returns
        -------
        None
        """
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "cmd", cmd)
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "size", size)

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


"""
The ReplyMessage class encapsulates the data and metadata corresponding to
a message returned by the Arkouda server
"""


@dataclass(frozen=True)
class ReplyMessage:

    __slots__ = ("msg", "msgType", "user")

    msg: str
    msgType: MessageType
    user: str

    @staticmethod
    def fromdict(values: Dict) -> ReplyMessage:
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
            return ReplyMessage(
                msg=values["msg"], msgType=MessageType(values["msgType"]), user=values["user"]
            )
        except KeyError as ke:
            raise ValueError(f"values dict missing {ke} field")
