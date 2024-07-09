#   This script finds all the classes and functions who's doc string contains "array".  These functions are most likely not natively compatible with arkouda and should probalby be excluded from any automatic imports.

import inspect

def exclude(name, obj)->bool:
    if hasattr(obj, "__doc__") and not name.startswith("__") and obj.__doc__ is not None:
        if "array" in obj.__doc__:
            return True
    return False


def main():

    import numpy as np

    exclude_set = set()
    for name, obj in inspect.getmembers(np):
        if exclude(name, obj):
            exclude_set.add(name)
    if inspect.isclass(obj):
        for func_name, func in inspect.getmembers(obj):
            if exclude(name, obj):
                exclude_set.add(name)

    exclude_list = list(exclude_set)    
    keep_list = list(set(dir(np)).difference(exclude_list))
    keep_list = [item for item in keep_list if not item.startswith("_")]
    keep_list = sorted(keep_list)
    
    print("EXCLUDE:")
    print(exclude_list)
    print("KEEP:")
    print(keep_list)


if __name__ == "__main__":
    main()
