from numpy import bool_, character, int_, integer, object_, str_
from arkouda.numpy.pdarrayclass import pdarray

__all__ = ["bool_", "character", "int_", "integer", "object_", "str_", "isnumeric"]


def isnumeric(pda) -> pdarray:
    """
    Return a boolean pdarray where index i indicates whether string i of the
    Strings has all numeric characters. There are 1922 unicode characters that
    qualify as numeric, including the digits 0 through 9, superscripts and
    subscripted digits, special characters with the digits encircled or
    enclosed in parens, "vulgar fractions," and more.

    Returns
    -------
    pdarray
        True for elements that are numerics, False otherwise

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error thrown
    TypeError
        Raised if the input is not a Strings type

    See Also
    --------
    Strings.isdecimal

    Examples
    --------
    >>> import arkouda as ak
    >>> not_numeric = ak.array([f'Strings {i}' for i in range(3)])
    >>> numeric = ak.array([f'12{i}' for i in range(3)])
    >>> strings = ak.concatenate([not_numeric, numeric])
    >>> strings
    array(['Strings 0', 'Strings 1', 'Strings 2', '120', '121', '122'])
    >>> ak.isnumeric(strings)
    array([False False False True True True])

    Special Character Examples

    >>> special_strings = ak.array(["3.14", "\u0030", "\u00b2", "2³₇", "2³x₇"])
    >>> special_strings
    array(['3.14', '0', '²', '2³₇', '2³x₇'])
    >>> ak.isnumeric(special_strings)
    array([False True True True False])
    """

    from arkouda.numpy.strings import Strings

    if type(pda) is not Strings:
        raise TypeError(f"input to isnumeric must be Strings.  Got {type(pda)}")

    return pda.isnumeric()
