# flake8: noqa
import inspect


def insert_spaces_after_newlines(input_string, spaces):
    if input_string is not None:
        lines = input_string.split("\n")
        result = []

        for line in lines:
            if line is not None:
                line = line.rstrip()
                if len(line) > 0:
                    line = spaces + line.rstrip()
                result.append(line)

        if len(result) > 0:
            return "\n".join(result)

    return None


def get_parent_class_str(obj):
    if hasattr(obj, "__class__"):
        parents = inspect.getmro(obj.__class__)
        if len(parents) > 0:
            # Extract the parent class which appears in quotes.
            parent = str(parents[1]).split("'")[1]
            parent_parts = parent.split(".")
            parent = parent_parts.pop()
            parent_module = ".".join(parent_parts)
            return parent_module, parent
    return None


def write_stub(module, filename):
    with open(filename, "w") as f:
        f.write("# flake8: noqa\n")
        f.write("# mypy: ignore-errors\n")
        f.write("from _typeshed import Incomplete\n")

        for name, obj in inspect.getmembers(module):
            #   Skip non-imported objects.
            if hasattr(obj, "__module__") and "arkouda" in obj.__module__:
                continue
            elif name.startswith("__") or inspect.ismodule(obj) or inspect.isbuiltin(obj):
                continue

            f.write("\n\n")
            if obj is None:
                f.write(f"{name} : None")

            elif isinstance(obj, float):
                f.write(f"{name} : float")

            elif inspect.isfunction(obj):
                if not name.startswith("__"):
                    try:
                        f.write(f"def {name}{inspect.signature(obj)}:\n")
                    except:
                        f.write(f"def {name}(self, *args, **kwargs):\n")

                    if len(obj.__doc__) > 0:
                        doc_string = insert_spaces_after_newlines(obj.__doc__, "    ")
                        if doc_string is not None:
                            f.write("    r'''\n")
                            f.write(f"{doc_string}\n")
                            f.write("    '''")
                            f.write("\n    ...")

            else:
                # Assume the object is a class

                parent_module, parent = get_parent_class_str(obj)
                if (
                    parent_module is not None
                    and parent is not None
                    and len(parent_module) > 0
                    and len(parent) > 0
                ):
                    f.write(f"from {parent_module} import {parent}\n\n\n")
                    f.write(f"class {name}")
                    f.write(f"({parent})")
                else:
                    f.write(f"class {name}")
                f.write(f":\n")
                doc_string = insert_spaces_after_newlines(obj.__doc__, "    ")
                if doc_string is not None:
                    f.write("    r'''\n")
                    f.write(f"{doc_string}\n")
                    f.write("    '''")

                for func_name, func in inspect.getmembers(obj):
                    if not func_name.startswith("__"):
                        f.write("\n\n")
                        try:
                            signature = str(inspect.signature(func))
                            if "self" not in signature:
                                signature = signature.replace("(", "(self, ")
                            f.write(f"    def {func_name}{signature}:\n")
                        except:
                            f.write(f"    def {func_name}(self, *args, **kwargs):\n")
                        doc_string = insert_spaces_after_newlines(obj.__doc__, "    ")
                        if doc_string is not None:
                            f.write("        r'''\n")
                            f.write(f"    {doc_string}\n")
                            f.write("        '''")
                            f.write("\n        ...")

            f.write("\n")


def main():
    import arkouda.numpy as aknp
    import arkouda.scipy as akscipy
    import arkouda.scipy.stats as akscipyStats

    write_stub(aknp, "arkouda/numpy/imports.pyi")
    write_stub(akscipy, "arkouda/scipy/imports.pyi")
    write_stub(akscipyStats, "arkouda/scipy/stats/imports.pyi")


if __name__ == "__main__":
    main()
