# PyTest Benchmarks

**Purpose**: Outline the general steps and options available for running the new PyTest benchmarks

The PyTest benchmark structure allows the flexibility of running the entire suite, a single file, 
or a single benchmark.

## Running The Full Suite
In most cases, running the full benchmark suite is desired. The simplest way to do this is to navigate into the 
root-level of arkouda and run `make benchmark`

This will run the entire benchmark suite with the following command:
```commandline
python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks
```

## Running Single Files or Tests

In instances where a single test or set of tests needs to be ran, use the `-k <name_of_file>.py` argument. For these 
examples, `encoding_benchmark.py` and the specific benchmark from within the file `bench_encode` is used.

```commandline
python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks -k encoding_benchmark.py
```

Running this command, you can expect to see an output table similar to this
```commandline
benchmark_v2/encoding_benchmark.py ....                                                                                                                                       [100%]
Saved benchmark data in: <Arkouda_root>/benchmark_v2/.benchmarks/Linux-CPython-3.9-64bit/0014_31de39be8b19c76d073a8999def6673a305c250d_20230405_145759_uncommited-changes.json

-------------------------------------------------------------------- benchmark 'Strings_EncodeDecode': 4 tests ---------------------------------------------------------------------
Name (time in ms)          Min               Max              Mean            StdDev            Median               IQR            Outliers       OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bench_encode[idna]      3.3304 (1.0)      9.2561 (2.10)     4.7544 (1.27)     2.5306 (6.18)     3.8075 (1.11)     1.9012 (3.62)          1;1  210.3306 (0.79)          5           1
bench_encode[ascii]     3.3805 (1.02)     4.8800 (1.10)     3.7336 (1.0)      0.6465 (1.58)     3.4231 (1.0)      0.5246 (1.0)           1;1  267.8380 (1.0)           5           1
bench_decode[idna]      3.4444 (1.03)     4.4177 (1.0)      3.7852 (1.01)     0.4097 (1.0)      3.5622 (1.04)     0.5837 (1.11)          1;0  264.1882 (0.99)          5           1
bench_decode[ascii]     3.4621 (1.04)     4.9177 (1.11)     4.2250 (1.13)     0.6125 (1.50)     4.0197 (1.17)     0.9991 (1.90)          2;0  236.6864 (0.88)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
Similarly, to only run a single test within a file, specify the test name with the `-k` flag instead of a filename
```commandline
python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks -k bench_encode
```
Results:
```commandline
benchmark_v2/encoding_benchmark.py ..                                                                                                                                         [100%]
Saved benchmark data in: <Arkouda_root>/benchmark_v2/.benchmarks/Linux-CPython-3.9-64bit/0015_31de39be8b19c76d073a8999def6673a305c250d_20230405_145947_uncommited-changes.json

-------------------------------------------------------------------- benchmark 'Strings_EncodeDecode': 2 tests ---------------------------------------------------------------------
Name (time in ms)          Min               Max              Mean            StdDev            Median               IQR            Outliers       OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bench_encode[ascii]     3.4298 (1.0)      3.6450 (1.0)      3.5541 (1.0)      0.0889 (1.0)      3.5801 (1.00)     0.1436 (1.0)           2;0  281.3620 (1.0)           5           1
bench_encode[idna]      3.4875 (1.02)     4.5255 (1.24)     3.7912 (1.07)     0.4328 (4.87)     3.5652 (1.0)      0.4869 (3.39)          1;0  263.7659 (0.94)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Benchmark Arguments
There are a large number of commandline arguments available for configuring the benchmarks to run in a way fitting 
to any use case.

<table>
<tr><th>Argument</th><th>Default Value</th></tr>
<tr><th colspan="2">Description</th></tr>
<tr>
    <td>--benchmark-autosave</td>
    <td>True when using "<b>make benchmark"</b></td>
</tr>
<tr><td colspan="2">Save the benchmark JSON results to the provided storage location</td></tr>
<tr>
    <td>--benchmark-storage</td>
    <td>"file://benchmark_v2/.benchmarks" when using "<b>make benchmark"</b></td>
</tr>
<tr><td colspan="2">Storage location to save benchmark output JSON</td></tr>
<tr>
    <td>--benchmark-save</td>
    <td>"COMMIT-ID_DATE_TIME_IS-DIRTY" example: 
31de39be8b19c76d073a8999def6673a305c250d_20230405_145947_uncommited-changes.json</td>
</tr>
<tr><td colspan="2">Name to save the output JSON as. Will be saved as "counter_NAME.json"</td></tr>
<tr>
    <td>--size</td>
    <td>10**8</td>
</tr>
<tr><td colspan="2">Problem size: length of array to use for benchmarks.</td></tr>
<tr>
    <td>--trials</td>
    <td>5</td>
</tr>
<tr><td colspan="2">Number of times to run each test before averaging results. For tests that run as many trials as possible in 
a given time, will be treated as number of seconds to run for.</td></tr>
<tr>
    <td>--seed</td>
    <td></td>
</tr>
<tr><td colspan="2">Value to initialize random number generator.</td></tr>
<tr>
    <td>--dtype</td>
    <td></td>
</tr>
<tr><td colspan="2">Dtypes to run benchmarks against. Comma separated list (NO SPACES) allowing for multiple. Accepted values: 
int64, uint64, bigint, float64, bool, str and mixed. Mixed is used to generate sets of multiple types.</td></tr>
<tr>
    <td>--numpy</td>
    <td>False</td>
</tr>
<tr><td colspan="2">When set, runs numpy comparison benchmarks.</td></tr>
<tr>
    <td>--maxbits</td>
    <td>-1</td>
</tr>
<tr><td colspan="2">Only applies to bigint testing. Maximum number of bits, so values > 2**max_bits will wraparound. -1 is 
interpreted as no maximum.</td></tr>
<tr>
    <td>--alpha</td>
    <td>1.0</td>
</tr>
<tr><td colspan="2">Scalar multiple</td></tr>
<tr>
    <td>--randomize</td>
    <td>False</td>
</tr>
<tr><td colspan="2">Fill arrays with random values instead of ones</td></tr>
<tr>
    <td>--index_size</td>
    <td></td>
</tr>
<tr><td colspan="2">Only used for Gather and Scatter Benchmarks. Length of index array (number of gathers to perform)</td></tr>
<tr>
    <td>--value_size</td>
    <td></td>
</tr>
<tr><td colspan="2">Only used for Gather and Scatter Benchmarks. Length of array from which values are gathered</td></tr>
<tr>
    <td>--encoding</td>
    <td></td>
</tr>
<tr><td colspan="2">Only applies to encoding benchmarks. Comma separated list (NO SPACES) allowing for multiple encoding to be 
used. Accepted values: idna, ascii</td></tr>
<tr>
    <td>--io_only_write</td>
    <td>False</td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Only write the files; files will not be removed</td></tr>
<tr>
    <td>--io_only_read</td>
    <td>False</td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Only read the files; files will not be removed</td></tr>
<tr>
    <td>--io_only_delete</td>
    <td>False</td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Only delete files created from writing with this benchmark</td></tr>
<tr>
    <td>--io_files_per_loc</td>
    <td>1</td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Number of files to create per locale</td></tr>
<tr>
    <td>--io_compression</td>
    <td></td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Compression types to run Parquet IO benchmarks against. Comma delimited list 
(NO SPACES) allowing for multiple. Accepted values: none, snappy, gzip, brotli, zstd, and lz4</td></tr>
<tr>
    <td>--io_path</td>
    <td>arkouda_root/benchmark_v2/ak_io_benchmark</td>
</tr>
<tr><td colspan="2">Only applies to IO benchmarks. Target path for measuring read/write rates</td></tr>
</table>

## Reading the JSON Output

The output JSON contains a lot of information. Not all of this information is beneficial for our purposes. 
The main area we care about is the `benchmarks.stats` section and associated information. To a lesser extent, 
machine information can be beneficial to see how different CPU architectures perform differently.

The `benchmarks` section contains one entry for every benchmark that gets ran. For a full build, this will result
in 350 entries. Each entry contains the name of the benchmark and a group name that allows for easy association
between related benchmarks.

The results for the benchmarks can be found within the provided benchmark storage path, which by default is within a
directory found in `//benchmark_v2/.benchmarks`.

The below JSON is the output from the above example 
`python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks -k bench_encode --size=100`

<details>
    <summary>Full example output JSON</summary>
    <pre>
        {
        "machine_info": {
            "node": "MSI",
            "processor": "x86_64",
            "machine": "x86_64",
            "python_compiler": "GCC 9.3.0",
            "python_implementation": "CPython",
            "python_implementation_version": "3.9.0",
            "python_version": "3.9.0",
            "python_build": [
                "default",
                "Nov 26 2020 07:57:39"
            ],
            "release": "5.10.16.3-microsoft-standard-WSL2",
            "system": "Linux",
            "cpu": {
                "python_version": "3.9.0.final.0 (64 bit)",
                "cpuinfo_version": [
                    9,
                    0,
                    0
                ],
                "cpuinfo_version_string": "9.0.0",
                "arch": "X86_64",
                "bits": 64,
                "count": 12,
                "arch_string_raw": "x86_64",
                "vendor_id_raw": "GenuineIntel",
                "brand_raw": "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz",
                "hz_advertised_friendly": "2.2000 GHz",
                "hz_actual_friendly": "2.2080 GHz",
                "hz_advertised": [
                    2200000000,
                    0
                ],
                "hz_actual": [
                    2207999000,
                    0
                ],
                "stepping": 10,
                "model": 158,
                "family": 6,
                "flags": [
                    "3dnowprefetch",
                    "abm",
                    "adx",
                    "aes",
                    "apic",
                    "arch_capabilities",
                    "avx",
                    "avx2",
                    "bmi1",
                    "bmi2",
                    "clflush",
                    "clflushopt",
                    "cmov",
                    "constant_tsc",
                    "cpuid",
                    "cx16",
                    "cx8",
                    "de",
                    "erms",
                    "f16c",
                    "flush_l1d",
                    "fma",
                    "fpu",
                    "fsgsbase",
                    "fxsr",
                    "ht",
                    "hypervisor",
                    "ibpb",
                    "ibrs",
                    "invpcid",
                    "invpcid_single",
                    "lahf_lm",
                    "lm",
                    "mca",
                    "mce",
                    "mmx",
                    "movbe",
                    "msr",
                    "mtrr",
                    "nopl",
                    "nx",
                    "osxsave",
                    "pae",
                    "pat",
                    "pcid",
                    "pclmulqdq",
                    "pdpe1gb",
                    "pge",
                    "pni",
                    "popcnt",
                    "pse",
                    "pse36",
                    "pti",
                    "rdrand",
                    "rdrnd",
                    "rdseed",
                    "rdtscp",
                    "rep_good",
                    "sep",
                    "smap",
                    "smep",
                    "ss",
                    "ssbd",
                    "sse",
                    "sse2",
                    "sse4_1",
                    "sse4_2",
                    "ssse3",
                    "stibp",
                    "syscall",
                    "tsc",
                    "vme",
                    "xgetbv1",
                    "xsave",
                    "xsavec",
                    "xsaveopt",
                    "xsaves",
                    "xtopology"
                ],
                "l3_cache_size": 9437184,
                "l2_cache_size": "1.5 MiB",
                "l1_data_cache_size": 196608,
                "l1_instruction_cache_size": 196608,
                "l2_cache_line_size": 256,
                "l2_cache_associativity": 6
            }
        },
        "commit_info": {
            "id": "31de39be8b19c76d073a8999def6673a305c250d",
            "time": "2023-04-04T16:26:14+00:00",
            "author_time": "2023-04-04T12:26:14-04:00",
            "dirty": true,
            "project": "arkouda",
            "branch": "2324_pytest_benchmark_docs"
        },
        "benchmarks": [
            {
                "group": "Strings_EncodeDecode",
                "name": "bench_encode[idna]",
                "fullname": "benchmark_v2/encoding_benchmark.py::bench_encode[idna]",
                "params": {
                    "encoding": "idna"
                },
                "param": "idna",
                "extra_info": {
                    "description": "Measures the performance of Strings.encode",
                    "problem_size": 100,
                    "transfer_rate": "0.0002 GiB/sec"
                },
                "options": {
                    "disable_gc": false,
                    "timer": "perf_counter",
                    "min_rounds": 5,
                    "max_time": 1.0,
                    "min_time": 5e-06,
                    "warmup": false
                },
                "stats": {
                    "min": 0.004066600000442122,
                    "max": 0.007168699999965611,
                    "mean": 0.0048064200000226265,
                    "stddev": 0.001326192548940973,
                    "rounds": 5,
                    "median": 0.004246700000294368,
                    "iqr": 0.0009575499998391024,
                    "q1": 0.004131924999910552,
                    "q3": 0.005089474999749655,
                    "iqr_outliers": 1,
                    "stddev_outliers": 1,
                    "outliers": "1;1",
                    "ld15iqr": 0.004066600000442122,
                    "hd15iqr": 0.007168699999965611,
                    "ops": 208.0550596900172,
                    "total": 0.024032100000113132,
                    "iterations": 1
                }
            },
            {
                "group": "Strings_EncodeDecode",
                "name": "bench_encode[ascii]",
                "fullname": "benchmark_v2/encoding_benchmark.py::bench_encode[ascii]",
                "params": {
                    "encoding": "ascii"
                },
                "param": "ascii",
                "extra_info": {
                    "description": "Measures the performance of Strings.encode",
                    "problem_size": 100,
                    "transfer_rate": "0.0002 GiB/sec"
                },
                "options": {
                    "disable_gc": false,
                    "timer": "perf_counter",
                    "min_rounds": 5,
                    "max_time": 1.0,
                    "min_time": 5e-06,
                    "warmup": false
                },
                "stats": {
                    "min": 0.00383609999971668,
                    "max": 0.0043372999998609885,
                    "mean": 0.004057779999857303,
                    "stddev": 0.00018361238254747651,
                    "rounds": 5,
                    "median": 0.0040258999997604406,
                    "iqr": 0.0002090000002681336,
                    "q1": 0.0039507749997937935,
                    "q3": 0.004159775000061927,
                    "iqr_outliers": 0,
                    "stddev_outliers": 2,
                    "outliers": "2;0",
                    "ld15iqr": 0.00383609999971668,
                    "hd15iqr": 0.0043372999998609885,
                    "ops": 246.44017172817806,
                    "total": 0.020288899999286514,
                    "iterations": 1
                }
            }
        ],
        "datetime": "2023-04-05T15:32:09.097392",
        "version": "4.0.0"
    }
</pre>
</details>

Simplified version of the JSON with only sections that we care about:
```json
{
    "machine_info": {
        "python_version": "3.9.0",
        "release": "5.10.16.3-microsoft-standard-WSL2",
        "system": "Linux",
        "cpu": {
            "arch": "X86_64",
            "count": 12,
            "brand_raw": "Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz"
        }
    },
    "benchmarks": [
        {
            "group": "Strings_EncodeDecode",
            "name": "bench_encode[ascii]",
            "params": {
                "encoding": "ascii"
            },
            "extra_info": {
                "description": "Measures the performance of Strings.encode",
                "problem_size": 100,
                "transfer_rate": "0.0002 GiB/sec"
            },
            "stats": {
                "min": 0.00383609999971668,
                "max": 0.0043372999998609885,
                "mean": 0.004057779999857303,
                "stddev": 0.00018361238254747651,
                "rounds": 5,
                "median": 0.0040258999997604406,
                "iqr": 0.0002090000002681336,
                "q1": 0.0039507749997937935,
                "q3": 0.004159775000061927,
                "iqr_outliers": 0,
                "stddev_outliers": 2,
                "outliers": "2;0",
                "ld15iqr": 0.00383609999971668,
                "hd15iqr": 0.0043372999998609885,
                "ops": 246.44017172817806,
                "total": 0.020288899999286514
            }
        }
    ],
    "datetime": "2023-04-05T15:32:09.097392"
}
```

The main important components are `benchmarks.extra_info`, which contains the details on the problem size and 
data transfer rate, and `benchmarks.stats` which contains all the timing statistic information calculated from
the number of trials we ran, represented in `benchmarks.stats.rounds`