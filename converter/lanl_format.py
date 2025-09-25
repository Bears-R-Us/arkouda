import re

import numpy as np


splitter = re.compile(r"([a-zA-Z]*)(\d*)")


def numerify(s):
    name, num = splitter.match(s).groups()
    prefix = hash(name) & 0xFFFF
    if num == "":
        postfix = 0
    else:
        postfix = int(num)
    return np.int64((prefix << 20) ^ postfix)


def deport(s):
    return np.int64(s.lstrip("Port"))


lanl_options = {
    "delimiter": ",",
    "header": None,
    "names": [
        "start",
        "duration",
        "srcIP",
        "dstIP",
        "proto",
        "srcPort",
        "dstPort",
        "srcPkts",
        "dstPkts",
        "srcBytes",
        "dstBytes",
    ],
    "converters": {  # 'srcIP': numerify, 'dstIP': numerify,
        "srcPort": deport,
        "dstPort": deport,
    },
    "dtype": {
        "srcIP": np.str_,
        "dstIP": np.str_,
        "start": np.int64,
        "duration": np.int64,
        "proto": np.int64,
        "srcPkts": np.int64,
        "dstPkts": np.int64,
        "srcBytes": np.float64,
        "dstBytes": np.float64,
    },
}


OPTIONS = {"lanl": lanl_options}
