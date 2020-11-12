from typing import Any, Callable

"""
The rangechecker decorator class retrieves the lower and upper bounds 
for the specified lowerField and upperField parameters and confirms
the following:

1. The lowerField value is greater than or equal to the lowerBound
2. The upperField value is less than or equal to the upperBound
3. The lowerField value is less than the upperField value
"""
class rangechecker():
    
    __slots__ = ('lowerField', 'lowerBound', 'upperField', 'upperBound')

    def __init__(self, lowerField, lowerBound, upperField, upperBound):
        self.lowerField = lowerField
        self.lowerBound = lowerBound
        self.upperField = upperField
        self.upperBound = upperBound

    def __call__(self, method : Callable) -> Any:
        """
        The rangechecker __call__ implementation retrieves the lowerBound
        and upperBound fields from the method arguments and confirms the
        following:
        
        1. The lowerField value is greater than or equal to the lowerBound
        2. The upperField value is less than or equal to the upperBound
        3. The lowerField value is less than the upperField value
        
        Parameters
        ----------
        method : Callable
            The decorated method
            
        Returns
        -------
        Any : the value returned by the decorated method
        
        Raises
        ------
        ValueError
            Raised if any of the 3 conditions are not met
        """
        def check(**args):
            lower = args[self.lowerField]
            upper = args[self.upperField]
            
            if lower < self.lowerBound:
                raise ValueError('the {} value must be >= {}'.\
                                 format(self.lowerField,self.lowerBound))
            elif upper > self.upperBound:
                raise ValueError('the {} value must be <= {}'.\
                                 format(self.upperField,self.upperBound))                
            elif lower >= upper:
                raise ValueError(('the {} value {} must be less than the ' +
                                 '{} value {}').format(self.lowerField,
                                 lower, self.upperField, upper))
            return method(**args)
        return check    
