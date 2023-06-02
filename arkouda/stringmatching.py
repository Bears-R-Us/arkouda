from arkouda.strings import Strings
from arkouda.pdarrayclass import create_pdarray
from arkouda.client import generic_msg
from typing import Union, cast
from math import floor


# This will expand as more algorithms become supported
ALGORITHMS = frozenset(["levenshtein", "jaro", "jaccard"])


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
        distances = repMsg
    else:
        distances = create_pdarray(repMsg)
    return distances


def jaro_distance(s1, s2):

    # If the s are equal
    if s1 == s2:
        return 1.0

    # Length of two s
    len1 = len(s1)
    len2 = len(s2)

    # Maximum distance upto which matching
    # is allowed
    max_dist = floor(max(len1, len2) / 2) - 1

    # Count of matches
    match = 0

    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    # Traverse through the first
    for i in range(len1):

        # Check if there is any matches
        for j in range(max(0, i - max_dist),
                       min(len2, i + max_dist + 1)):

            # If there is a match
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    # If there is no match
    if match == 0:
        return 0.0

    # Number of transpositions
    t = 0
    point = 0

    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if hash_s1[i]:

            # Find the next matched character
            # in second
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1

    t = t // 2

    # Return the Jaro Similarity
    return 1 - (match / len1 + match / len2 + (match - t) / match) / 3.0


def jaccard(str1, str2):
    l1 = set([str1[i] + str1[i+1] for i in range(len(str1) - 1)])
    l2 = set([str2[i] + str2[i + 1] for i in range(len(str2) - 1)])

    intersection = len(list(l1.intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union
