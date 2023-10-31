import json
from json import JSONEncoder
from typing import List, Union, cast

from typeguard import typechecked

from arkouda.client import generic_msg

__all__ = [
    "AllSymbols",
    "RegisteredSymbols",
    "information",
    "list_registry",
    "list_symbol_table",
    "pretty_print_information",
]
AllSymbols = "__AllSymbols__"
RegisteredSymbols = "__RegisteredSymbols__"


def auto_str(cls):
    def __str__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%s" % item for item in vars(self).items()))

    cls.__str__ = __str__
    return cls


class EntryDecoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


@auto_str
class InfoEntry:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs["name"]
        self.dtype = kwargs["dtype"]
        self.size = kwargs["size"]
        self.ndim = kwargs["ndim"]
        self.shape = kwargs["shape"]
        self.itemsize = kwargs["itemsize"]
        self.registered = kwargs["registered"]


@typechecked
def information(names: Union[List[str], str] = RegisteredSymbols) -> str:
    """
    Returns JSON formatted string containing information about the objects in names

    Parameters
    ----------
    names : Union[List[str], str]
       names is either the name of an object or list of names of objects to retrieve info
       if names is ak.AllSymbols, retrieves info for all symbols in the symbol table
       if names is ak.RegisteredSymbols, retrieves info for all symbols in the registry

    Returns
    ------
    str
        JSON formatted string containing a list of information for each object in names

    Raises
    ------
    RuntimeError
        Raised if a server-side error is thrown in the process of
        retrieving information about the objects in names
    """
    if isinstance(names, str):
        if names in [AllSymbols, RegisteredSymbols]:
            return cast(str, generic_msg(cmd="info", args={"names": names}))
        else:
            names = [names]  # allows user to call ak.information(pda.name)
    return cast(str, generic_msg(cmd="info", args={"names": json.dumps(names)}))


def list_registry(detailed: bool = False):
    """
    Return a list containing the names of all registered objects

    Parameters
    ----------
    detailed: bool
        Default = False
        Return details of registry objects. Currently includes object type for any objects

    Returns
    -------
    dict
        Dict containing keys "Components" and "Objects".

    Raises
    ------
    RuntimeError
        Raised if there's a server-side error thrown
    """
    data = json.loads(cast(str, generic_msg(cmd="list_registry")))
    objs = json.loads(data["Objects"]) if data["Objects"] != "" else []
    obj_types = json.loads(data["Object_Types"]) if data["Object_Types"] != "" else []
    return {
        "Objects": list(zip(objs, obj_types)) if detailed else objs,
        "Components": json.loads(data["Components"]),
    }


def list_symbol_table() -> List[str]:
    """
    Return a list containing the names of all objects in the symbol table

    Parameters
    ----------
    None

    Returns
    -------
    list
        List of all object names in the symbol table

    Raises
    ------
    RuntimeError
        Raised if there's a server-side error thrown
    """
    return [i.name for i in _parse_json(AllSymbols)]


def _parse_json(names: Union[List[str], str]) -> List[InfoEntry]:
    """
    Internal method that converts the JSON output of information into a List of InfoEntry objects

    Parameters
    ----------
    names : Union[List[str], str]
    Names to pass to information

    Returns
    -------
    List[InfoEntry]
        List of InfoEntry python objects for each name in names

    Raises
    ------
    RuntimeError
        Raised if a server-side error is thrown
    """
    return json.loads(information(names), object_hook=lambda d: InfoEntry(**d))


def pretty_print_information(names: Union[List[str], str] = RegisteredSymbols) -> None:
    """
    Prints verbose information for each object in names in a human readable format

    Parameters
    ----------
    names : Union[List[str], str]
       names is either the name of an object or list of names of objects to retrieve info
       if names is ak.AllSymbols, retrieves info for all symbols in the symbol table
       if names is ak.RegisteredSymbols, retrieves info for all symbols in the registry

    Returns
    ------
    None

    Raises
    ------
    RuntimeError
        Raised if a server-side error is thrown in the process of
        retrieving information about the objects in names
    """
    for i in _parse_json(names):
        print(i)
