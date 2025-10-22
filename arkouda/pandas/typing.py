from typing import TYPE_CHECKING, TypeVar, Union

from arkouda.numpy.pdarrayclass import pdarray


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
    from arkouda.pandas.categorical import Categorical
else:
    Strings = TypeVar("Strings")
    Categorical = TypeVar("Categorical")


ArkoudaArrayLike = Union[pdarray, Strings, Categorical]
