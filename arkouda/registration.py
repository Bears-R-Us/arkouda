from typing import Union
from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray

global verbose

__all__ = ["register_pda","attach_pda","unregister_pda"]

def register_pda(pda : pdarray, user_defined_name : str) -> pdarray:
    """
    Return a pdarray with a user defined name in the arkouda server 
    so it can be attached to later using attach_pda()
    
    Parameters
    ----------
    pda : str or pdarray
        the array to register
    user_defined_name : str
        user defined name array is to be registered under

    Returns
    -------
    pdarray
        pdarray which points to original input pdarray but is also 
        registered with user defined name in the arkouda server


    Raises
    ------
    TypeError
        Raised if pda is neither a pdarray nor a str or if
        user_defined_name is not a str

    See also
    --------
    attach_pda, unregister_pda

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion 
    until they are unregistered.

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
        raise TypeError("user_defined_name must be a str")

    ret = None
    if isinstance(pda, pdarray):
        repMsg = generic_msg("register {} {}".\
                             format(pda.name, user_defined_name))
        ret = create_pdarray(repMsg)
    elif isinstance(pda, str):
        repMsg = generic_msg("register {} {}".\
                             format(pda, user_defined_name))        
        ret = create_pdarray(repMsg)
    else:
        raise TypeError("pda must be pdarray or str")
    
    return ret

def attach_pda(user_defined_name : str) -> pdarray:
    """
    Return a pdarray attached to the a registered name in the arkouda 
    server which was registered using register_pda()
    
    Parameters
    ----------
    user_defined_name : str
        user defined name which array was registered under

    Returns
    -------
    pdarray
        pdarray which points to pdarray registered with user defined
        name in the arkouda server
        
    Raises
    ------
    TypeError
        Raised if user_defined_name is not a str

    See also
    --------
    register_pda, unregister_pda

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion 
    until they are unregistered.

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
        raise TypeError("user_defined_name must be a str")

    repMsg = generic_msg("attach {}".format(user_defined_name))
    return create_pdarray(repMsg)


def unregister_pda(pda : Union[str,pdarray]) -> None:
    """
    Unregister a pdarray in the arkouda server which was previously 
    registered using register_pda() and/or attahced to using attach_pda()
    
    Parameters
    ----------
    pda : str or pdarray
        user define name which array was registered under

    Returns
    -------
    None

    Raises 
    ------
    TypeError
        Raised if pda is neither a pdarray nor a str

    See also
    --------
    register_pda, unregister_pda

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion until 
    they are unregistered.

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
        raise TypeError("pda must be pdarray or str")
