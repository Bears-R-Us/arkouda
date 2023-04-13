from arkouda.strings import Strings
from arkouda.pdarrayclass import create_pdarray
from arkouda.client import generic_msg
from typing import Union, cast


# This will expand as more algorithms become supported
ALGORITHMS = frozenset(["levenshtein"])


def string_match(search_param: str, dataset: Union[str, Strings], algo="levenshtein"):
    if algo.lower() not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algo}. Supported algorithms are: {ALGORITHMS}")

    search_mode = "multi" if isinstance(dataset, Strings) else "single"

    repMsg = cast(str, generic_msg(
        cmd="stringMatching",
        args={
            "query": search_param,
            "dataset": dataset,
            "algorithm": algo.lower(),
            "mode": search_mode
        }
    ))

    if search_mode == "single":
        distances = int(repMsg)
    else:
        distances = create_pdarray(repMsg)
    return distances
