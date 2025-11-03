from typing import TYPE_CHECKING, TypeVar, Union

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.strings import Strings


if TYPE_CHECKING:
    from arkouda.pandas.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


ArkoudaArrayLike = Union[pdarray, Strings, Categorical]
