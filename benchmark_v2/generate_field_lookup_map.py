#!/usr/bin/env python3
"""
Generate a field-to-regex lookup map for benchmark performance keys.

This script parses `.perfkeys` files located in the benchmark graph infrastructure
directory, infers regular expressions to identify benchmark output keys,
and maps them to internal data paths used for analysis (e.g., `stats.mean`,
`extra_info.transfer_rate`). The result is written to a JSON file for use in
benchmark plotting and evaluation tools.

Main Components:
----------------
- `infer_regex`:
    Infers a benchmark regex pattern based on benchmark and field names.
- `get_header_fields_from_directory`:
    Loads header fields from `.perfkeys` files in a given directory.
- `build_field_lookup_map`:
    Builds the core mapping from benchmark fields to regex expressions and paths.
- `add_default_mappings`:
    Adds standard time/rate mappings for simple benchmarks.
- `add_aggregate_ops`:
    Adds entries for GroupBy and reduction-based aggregate operations.

Output:
-------
Writes the resulting mapping to `benchmark_v2/datdir/configs/field_lookup_map.json`.

"""

import json
import logging
import os
import re

# Aggregate operations explicitly defined
import arkouda as ak
from arkouda.logger import getArkoudaLogger

GRAPH_INFRA_DIR = "benchmark_v2/graph_infra"
OUTPUT_JSON = "benchmark_v2/datdir/configs/field_lookup_map.json"

# Benchmarks that just need default Average rate/time keys
DEFAULT_BENCHMARKS = [
    "stream",
    "argsort",
    "str-argsort",
    "gather",
    "str-gather",
    "scatter",
    "bigint_stream",
    "flatten",
    "noop",
    "split",
]


AGGREGATE_OPS = ak.GroupBy.Reductions


def infer_regex(benchmark_name: str, field: str) -> str:
    """Infer a regex for JSON benchmark names based on perfkey field names."""
    base_bench = re.sub(r"^(str|bigint)(?:_|-)", "", benchmark_name)

    if "array_transfer" in base_bench:
        if "to_ndarray" in field:
            base_bench = base_bench + "_tondarray"
        elif "ak.array" in field:
            base_bench = base_bench + "_akarray"

    # Groupby & Coargsort (with array counts)
    if "array" in field and base_bench in ["groupby", "coargsort"]:
        m = re.search(r"(\d+)-array", field)
        if m:
            num = m.group(1)
            if benchmark_name.startswith("str-"):
                dtype = "str"
            elif benchmark_name.startswith("bigint-"):
                dtype = "bigint"
            else:
                dtype = "(?:int64|float64|bool|uint64)"
            return f"bench_{base_bench}\\[{dtype}-{num}\\]"

    # IO
    if "IO" == benchmark_name:
        m = re.search(r"((?:write|read)) Average", field)
        if m:
            op = m.group(1)
            dtype = "(?:int64|float64|bool|uint64|str)"
            return f"bench_{op}_hdf\\[{dtype}\\]"

    # CSV Read/Write
    if "csvIO" == benchmark_name:
        m1 = re.search(r"(write|read)", field)
        if m1:
            op = m1.group(1)
            dtype = "(?:int64|float64|bool|uint64|str)"
            return f"bench_csv_io\\[{op}-{dtype}\\]"

    # multiIO Read/Write
    if "multiIO" == benchmark_name:
        m1 = re.search(r"(write|read)", field)
        if m1:
            op = m1.group(1)
            dtype = "(?:int64|float64|bool|uint64|str)"
            return f"bench_{op}_hdf_multi\\[{dtype}\\]"

    # parquet IO
    if benchmark_name in ["parquetIO", "parquetMultiIO"]:
        qualifier = "_multi" if "Multi" in benchmark_name else ""
        m1 = re.search(r"((?:write|read)) Average", field)
        m2 = re.search(r"(\w+) =", field)
        if m1 and m2:
            op = m1.group(1)
            compression = m2.group(1)
            compression = "None" if compression == "none" else compression
            dtype = "(?:int64|float64|bool|uint64|str)"
            return f"bench_{op}_parquet{qualifier}\\[{compression}-{dtype}\\]"

    # encode
    if "encode" in benchmark_name:
        m1 = re.search(r"((?:ascii|idna))", field)
        m2 = re.search(r"((?:encode|decode))", field)
        if m1 and m2:
            encoding = m1.group(1)
            mode = m2.group(1)
            return f"bench_{mode}\\[{encoding}\\]"

    # small-str-groupby
    if "small-str-groupby" in benchmark_name:
        m = re.search(r"((?:small|medium|big)) str array Average", field)
        if m:
            op = m.group(1)
            return f"bench_groupby_small_str\\[{op}\\]"

    # dataframe
    if base_bench == "dataframe":
        m = re.search(r"([_\-\w]+) Average", field)
        if m:
            op = m.group(1)
            return f"bench_{base_bench}\\[{op}\\]"

    #  reduce
    if benchmark_name in ["reduce", "scan", "setops", "array_create"]:
        m = re.search(r"(\w+) Average", field)
        if m:
            op = m.group(1)
            dtype = "(?:int64|float64|bool|uint64)"
            return f"bench_{base_bench}\\[{dtype}-{op}\\]"

    # bigint_conversion
    if "bigint_conversion" in benchmark_name:
        m = re.search(r"(\w+) Average", field)
        if m:
            op = m.group(1)
            return f"bench_bigint_conversion\\[{op}\\]"

    # in1d & str-in1d
    if "in1d" in benchmark_name:
        m = re.search(r"((?:Medium|Large)) average", field)
        if m:
            size = m.group(1).upper()
            if benchmark_name.startswith("str-"):
                dtype = "str"
            elif benchmark_name.startswith("bigint-"):
                dtype = "bigint"
            else:
                dtype = "(?:int64|float64|bool|uint64)"
            return f"bench_in1d\\[{size}-{dtype}\\]"

    # str_locality
    if "str-locality" in benchmark_name:
        m1 = re.search(r"((?:Hashing|Regex|Casting|Comparing))", field)
        m2 = re.search(r"((?:good|poor))", field)
        if m1 and m2:
            op = m1.group(1)
            locality = m2.group(1)
            return f"bench_str_locality\\[{locality}-{op}\\]"

    # bigint_bitwise_binops
    if "bigint_bitwise_binops" in benchmark_name:
        m1 = re.search(r"((?:AND|OR|SHIFT))", field)
        if m1:
            op = m1.group(1).lower()
            return f"bench_bitwise_binops\\[{op}\\]"

    # sort-cases
    if benchmark_name == "sort-cases":
        m1 = re.search(r"((?:RadixSortLSD|TwoArrayRadixSort))", field)
        if m1:
            sort = "SortingAlgorithm." + m1.group(1)
            dtype = "(?:int64|float64|bool|uint64|str)"
            if "RMAT" in field:
                return f"bench_rmat\\[{sort}\\]"
            elif "block-sorted" in field:
                m2 = re.search(r"(concat|interleaved)", field)
                if m2:
                    mode = m2.group(1)
                    return f"bench_block_sorted\\[{mode}-{sort}\\]"
            elif "refinement" in field:
                return f"bench_refinement\\[{sort}\\]"
            elif "datetime" in field:
                return f"bench_time_like\\[{sort}\\]"
            elif "IP" in field:
                return f"bench_ip_like\\[{sort}\\]"
            elif "uniform" in field:
                m2 = re.search(r"(\d+)-bit", field)
                m3 = re.search(r"float64", field, re.IGNORECASE)
                if m2:
                    mode = m2.group(1)
                    return f"bench_random_uniform\\[{mode}-bit-{dtype}-{sort}\\]"
                elif m3:
                    return f"bench_random_uniform\\[64-bit-float64-{sort}\\]"
            elif "power" in field:
                return f"bench_power_law\\[{dtype}-{sort}\\]"

    # substring-search
    if benchmark_name == "substring_search":
        m1 = re.search(r"(^(?:non-regex|regex))", field)
        if m1:
            regex_type = m1.group(1)
            if regex_type == "regex":
                m2 = re.search(r"(pattern)", field)
                if m2:
                    return "bench_strings_contains\\[Regex_Pattern\\]"
                else:
                    return "bench_strings_contains\\[Regex_Literal\\]"
            elif regex_type == "non-regex":
                return "bench_strings_contains\\[Non_Regex\\]"

    # Reduce/Scan/Aggregate
    if benchmark_name in {"reduce", "scan", "aggregate"}:
        op = field.split()[0]
        return f"bench_{benchmark_name}\\[{op}\\]"

    # Bigint Stream (special case)
    if benchmark_name == "bigint_stream":
        return r"bench_bigint_stream\[bigint\]"


def get_header_fields_from_directory(directory_path):
    """Load perfkeys headers into a dict."""
    file_contents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".perfkeys"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                key = re.search(r"([\w\-_]+)\.perfkeys", filename)[1]
                file_contents[key] = lines
    return file_contents


def build_field_lookup_map():
    headers = get_header_fields_from_directory(GRAPH_INFRA_DIR)
    field_lookup_map = {}

    for benchmark_name, fields in headers.items():
        field_lookup_map[benchmark_name] = {}
        for field in fields:
            if field == "# Date":
                continue
            regex = infer_regex(benchmark_name, field)
            lookup_path = ["extra_info", "transfer_rate"] if "rate" in field else ["stats", "mean"]

            field_lookup_map[benchmark_name][field] = {
                "name": "",
                "benchmark_name": benchmark_name.replace("str-", "")
                .replace("bigint-", "")
                .replace("-", "_"),
                "lookup_path": lookup_path,
                "lookup_regex": regex,
            }

    return field_lookup_map


def add_default_mappings(field_lookup_map):
    for b in DEFAULT_BENCHMARKS:
        if b not in field_lookup_map:
            base_bench = b.replace("str-", "").replace("bigint-", "").replace("-", "_")
            if b.startswith("str-"):
                dtype = "str"
            elif b.startswith("bigint-"):
                dtype = "bigint"
            else:
                dtype = "(?:int64|float64|bool|uint64)"

            if base_bench == "noop":
                regex = rf"^bench_{base_bench}.*$"
            elif base_bench == "bigint_stream":
                regex = r"bench_bigint_stream\[bigint\]"
            else:
                regex = f"bench_{base_bench}\\[{dtype}\\]"

            field_lookup_map[b] = {
                "Average rate =": {
                    "name": f"bench_{base_bench}",
                    "benchmark_name": base_bench,
                    "lookup_path": ["extra_info", "transfer_rate"],
                    "lookup_regex": regex,
                },
                "Average time =": {
                    "name": f"bench_{base_bench}",
                    "benchmark_name": base_bench,
                    "lookup_path": ["stats", "mean"],
                    "lookup_regex": regex,
                },
            }
    return field_lookup_map


def add_aggregate_ops(field_lookup_map):
    if "aggregate" not in field_lookup_map:
        field_lookup_map["aggregate"] = {}
    if "reduce" not in field_lookup_map:
        field_lookup_map["reduce"] = {}

    for op in AGGREGATE_OPS:  # should include all GroupBy.Reductions ops
        for t in ["time", "rate"]:
            lookup_path = ["extra_info", "transfer_rate"] if t == "rate" else ["stats", "mean"]

            # Correct mapping for GroupBy.aggregate
            field_lookup_map["aggregate"][f"Aggregate {op} Average {t} ="] = {
                "name": f"bench_aggregate[{op}]",
                "benchmark_name": "aggregate",
                "lookup_path": lookup_path,
                "lookup_regex": f"^bench_aggregate\\[{op}\\]$",
            }

    # Keep reduce ops separate (only numeric ops)
    for op in ["sum", "prod", "min", "max", "argmin", "argmax"]:
        for t in ["time", "rate"]:
            lookup_path = ["extra_info", "transfer_rate"] if t == "rate" else ["stats", "mean"]
            field_lookup_map["reduce"][f"Reduce {op} Average {t} ="] = {
                "name": f"bench_reduce[{op}]",
                "benchmark_name": "reduce",
                "lookup_path": lookup_path,
                "lookup_regex": f"bench_reduce\\[[\\w\\d]+-{op}\\]",
            }
    return field_lookup_map


def main():
    logger = getArkoudaLogger(name="generate field lookup map")

    field_lookup_map = build_field_lookup_map()
    field_lookup_map = add_default_mappings(field_lookup_map)
    field_lookup_map = add_aggregate_ops(field_lookup_map)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(field_lookup_map, f, indent=2, sort_keys=True)

    logger.debug(f"Updated {OUTPUT_JSON} with {len(field_lookup_map)} benchmarks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
