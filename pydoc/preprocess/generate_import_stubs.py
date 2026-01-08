# flake8: noqa
import inspect
import re


def insert_spaces_after_newlines(input_string, spaces):
    if input_string is not None:
        pattern = r"^\n(\s+)"
        starting_indents = re.findall(pattern, input_string)
        if len(starting_indents) > 0:
            old_indent_pattern = "^" + starting_indents[0]
        else:
            old_indent_pattern = "^"

        lines = input_string.split("\n")
        result = []

        for line in lines:
            if line is not None:
                line = line.rstrip()
                if len(line) > 0:
                    line = re.sub(old_indent_pattern, spaces, line, count=1)
                result.append(line)

        if len(result) > 0:
            return "\n".join(result)

    return None


def reformat_signature(signature_string: str):
    signature_string = re.sub(r"\<([\w.]+):[ \'\w\>]+", "\\1", signature_string)
    signature_string = re.sub(r"\<class \'([\w.]+)\'\>", "\\1", signature_string)
    return signature_string


def get_parent_class_str(obj):
    if hasattr(obj, "__class__"):
        if inspect.isclass(obj):
            parents = inspect.getmro(obj)
        else:
            parents = inspect.getmro(obj.__class__)
        if len(parents) > 0:
            # Extract the parent class which appears in quotes.
            parent = str(parents[1]).split("'")[1]
            parent_parts = parent.split(".")
            parent = parent_parts.pop()
            parent_module = ".".join(parent_parts)
            return parent_module, parent
    return None


def write_formatted_docstring(f, doc_string, spaces):
    doc_string = insert_spaces_after_newlines(doc_string, spaces)
    if doc_string is not None and len(doc_string) > 0:
        #   AutoApi cannot parse "def" inside a docstring, so replace:
        doc_string = doc_string.replace("def ", "def\\ ")

        f.write(spaces + 'r"""\n')
        f.write(f"{doc_string}\n")
        f.write(spaces + '"""')
        f.write("\n" + spaces + "...")
    else:
        f.write("\n" + spaces + "...")


def write_stub(module, filename, all_only=False, allow_arkouda=False):
    with open(filename, "w") as f:
        f.write("# flake8: noqa\n")
        f.write("# mypy: ignore-errors\n")
        f.write("from _typeshed import Incomplete\n")

        for name, obj in inspect.getmembers(module):
            #   Skip non-imported objects.
            if all_only is True and name not in module.__all__:
                continue
            if (allow_arkouda is False) and (hasattr(obj, "__module__") and "arkouda" in obj.__module__):
                continue
            elif name.startswith("__") or inspect.ismodule(obj) or inspect.isbuiltin(obj):
                continue

            f.write("\n\n")
            if obj is None:
                f.write(f"{name} : None")
            elif isinstance(obj, float):
                f.write(f"{name} : float")
            elif isinstance(obj, int):
                f.write(f"{name} : int")

            elif inspect.isfunction(obj):
                if not name.startswith("__"):
                    try:
                        signature_string = reformat_signature(str(inspect.signature(obj)))
                        f.write(f"def {name}{signature_string}:\n")
                    except:
                        f.write(f"def {name}(self, *args, **kwargs):\n")

                    write_formatted_docstring(f, obj.__doc__, "    ")

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
                write_formatted_docstring(f, obj.__doc__, "    ")

                for func_name, func in inspect.getmembers(obj):
                    # Don't document functions that are inherited from the parent class.
                    if inspect.isclass(obj) and func_name in set(dir(inspect.getmro(obj)[1])):
                        continue
                    if not inspect.isclass(obj) and func_name in set(
                        dir(inspect.getmro(obj.__class__)[1])
                    ):
                        continue

                    if not func_name.startswith("__"):
                        # "and" and "or" need to be magic functions b/c naming collision with python.
                        if func_name == "and" or func_name == "or":
                            func_name = "__" + func_name + "__"

                        f.write("\n\n")
                        if isinstance(func, property):
                            f.write(f"    @property\n")
                            f.write(f"    def {func_name}{signature}:\n")
                        else:
                            try:
                                signature = reformat_signature(str(inspect.signature(func)))
                                if "self" not in signature:
                                    signature = signature.replace("(", "(self, ")
                                f.write(f"    def {func_name}{signature}:\n")
                            except:
                                f.write(f"    def {func_name}(self, *args, **kwargs):\n")

                        write_formatted_docstring(f, func.__doc__, "        ")

            f.write("\n")


def main():
    import arkouda as ak
    import arkouda.numpy as aknp
    import arkouda.pandas.dataframe as akDataframe
    import arkouda.pandas.groupbyclass as akGroupbyclass
    import arkouda.scipy as akscipy
    import arkouda.scipy.sparrayclass as akscipySparrayclass
    import arkouda.scipy.sparsematrix as akscipySparsematrix
    import arkouda.scipy.special as akscipySpecial
    import arkouda.scipy.stats as akscipyStats

    write_stub(
        aknp.dtypes,
        "arkouda/numpy/dtypes.pyi",
        all_only=False,
        allow_arkouda=True,
    )
    write_stub(akscipy, "arkouda/scipy.pyi", all_only=True, allow_arkouda=True)
    write_stub(akscipyStats, "arkouda/scipy/stats.pyi", all_only=True, allow_arkouda=True)
    write_stub(akscipySpecial, "arkouda/scipy/special.pyi", all_only=True, allow_arkouda=True)
    write_stub(akscipySparrayclass, "arkouda/scipy/sparrayclass.pyi", all_only=True, allow_arkouda=True)
    write_stub(akscipySparsematrix, "arkouda/scipy/sparsematrix.pyi", all_only=True, allow_arkouda=True)
    write_stub(akDataframe, "arkouda/pandas/dataframe.pyi", all_only=True, allow_arkouda=True)
    write_stub(akGroupbyclass, "arkouda/pandas/groupbyclass.pyi", all_only=True, allow_arkouda=True)
    write_stub(
        aknp.pdarrayclass,
        "arkouda/numpy/pdarrayclass.pyi",
        all_only=True,
        allow_arkouda=True,
    )
    write_stub(aknp.imports, "arkouda/numpy/imports.pyi", all_only=True, allow_arkouda=False)


if __name__ == "__main__":
    main()
