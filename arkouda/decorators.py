from typing import Union
from arkouda import pdarray
from arkouda.strings import Strings

def checkforpdarray(func):
    def check(pda : pdarray, **args):
        if not isinstance(pda, pdarray):
            raise TypeError('must be a pdarray, not a {}'.\
                        format(pda.__class__.__name__))
        return func(pda)
    return check
            
def checkforpdarrayorstrings(func):
    def check(pda : Union[pdarray,Strings], return_counts : bool=False):
        if not isinstance(pda, pdarray) \
                               and not isinstance(pda, Strings):
            raise TypeError('parameter must be a pdarray, not {}'.\
                        format(pda.__class__.__name__))