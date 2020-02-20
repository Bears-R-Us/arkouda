import numpy as np

from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray

global verbose

__all__ = ["register_pda","attach_pda","unregister_pda"]

def register_pda(pda, user_defined_name):
    if not isinstance(user_defined_name, str):
        raise ValueError("user_defined_name must be a string")

    ret = None
    if isinstance(pda, pdarray):
        repMsg = generic_msg("register {} {}".format(pda.name, user_defined_name))
        ret = create_pdarray(repMsg)
    elif isinstance(pda, str):
        repMsg = generic_msg("register {} {}".format(pda, user_defined_name))        
        ret = create_pdarray(repMsg)
    else:
        raise ValueError("pda must be pdarray or string")
    
    return ret


def attach_pda(name):
    if not isinstance(name, str):
        raise ValueError("user_defined_name must be a string")

    repMsg = generic_msg("attach {}".format(name))
    return create_pdarray(repMsg)


def unregister_pda(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("unregister {}".format(pda.name))
    elif isinstance(pda, str):
        repMsg = generic_msg("unregister {}".format(pda))
    else:
        raise ValueError("pda must be pdarray or string")
