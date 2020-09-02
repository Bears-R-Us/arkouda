import numpy as np

from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray

global verbose

__all__ = ["register_pda","attach_pda","unregister_pda"]

def register_pda(pda, user_defined_name):
    """
    Return a pdarray with a user defined name in the arkouda server so it can be attached to later using arkouda.attach_pda()

    Parameters
    ----------
    pda : str or pdarray
    user_defined_name : string of user defined name

    Returns
    -------
    pdarray : points to original input pda but has user defined name in the arkouda server

    See also
    --------
    attach_pda, unregister_pda

    Examples
    --------
    >>> a = zeros(100)
    >>> r_pda = ak.register_pda(a, "my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.attach_pda("my_zeros")
    >>> # ...other work...
    >>> ak.unregister_pda(b)
    """
    if not isinstance(user_defined_name, str):
        raise ValueError("user_defined_name must be a str")

    ret = None
    if isinstance(pda, pdarray):
        repMsg = generic_msg("register {} {}".format(pda.name, user_defined_name))
        ret = create_pdarray(repMsg)
    elif isinstance(pda, str):
        repMsg = generic_msg("register {} {}".format(pda, user_defined_name))        
        ret = create_pdarray(repMsg)
    else:
        raise ValueError("pda must be pdarray or str")
    
    return ret


def attach_pda(name):
    """
    Return a pdarray attached to the a registered name in the arkouda server which was registered using arkouda.register_pda()
    
    Parameters
    ----------
    name : string of user defined name

    Returns
    -------
    pdarray : points to pdarray with user defined name in the arkouda server

    See also
    --------
    register_pda, unregister_pda

    Examples
    --------
    >>> a = zeros(100)
    >>> r_pda = ak.register_pda(a, "my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.attach_pda("my_zeros")
    >>> # ...other work...
    >>> ak.unregister_pda(b)
    """
    if not isinstance(name, str):
        raise ValueError("user_defined_name must be a str")

    repMsg = generic_msg("attach {}".format(name))
    return create_pdarray(repMsg)


def unregister_pda(pda):
    """
    Unregister a pdarray in the arkouda server which was previously regisgersted using arkouda.register_pda and/or attahced to using arkouda.attach_pda()
    Parameters
    ----------
    pda : str or pdarray which has previously been registered with a user define name using register_pda

    Returns
    -------
    None

    See also
    --------
    register_pda, unregister_pda

    Examples
    --------
    >>> a = zeros(100)
    >>> r_pda = ak.register_pda(a, "my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.attach_pda("my_zeros")
    >>> # ...other work...
    >>> ak.unregister_pda(b)
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("unregister {}".format(pda.name))
    elif isinstance(pda, str):
        repMsg = generic_msg("unregister {}".format(pda))
    else:
        raise ValueError("pda must be pdarray or str")
