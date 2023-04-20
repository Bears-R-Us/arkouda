from arkouda.strings import Strings
from arkouda.pdarrayclass import create_pdarray
from arkouda.client import generic_msg
from typing import Union, cast


# This will expand as more algorithms become supported
ALGORITHMS = frozenset(["levenshtein"])


def string_match(search_param: Union[str, Strings], dataset: Union[str, Strings], algo="levenshtein"):
    if algo.lower() not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {algo}. Supported algorithms are: {ALGORITHMS}")

    if isinstance(dataset, Strings) and isinstance(search_param, Strings):
        if search_param.size != dataset.size:
            raise ValueError("Strings search_param and dataset must be the same length")
        search_mode = "many"
    elif isinstance(dataset, Strings) and isinstance(search_param, str):
        search_mode = "multi"
    elif isinstance(dataset, str) and isinstance(search_param, str):
        search_mode = "single"
    else:
        raise TypeError(
            f"Combination search_param type {type(search_param)}, dataset type {type(dataset)} not supported"
        )

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
