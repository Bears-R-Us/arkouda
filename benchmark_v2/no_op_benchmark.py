import arkouda as ak
import numpy as np
import pytest


@pytest.mark.benchmark(group="Arkouda_No_Op")
def bench_ak_noop(benchmark):
    benchmark.pedantic(ak.client._no_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = \
        "Measures the performance of ak.client._no_op for a basic round-trip time"
    benchmark.extra_info["problem_size"] = "N/A"
    benchmark.extra_info["transfer_rate"] = "N/A"


@pytest.mark.benchmark(group="Arkouda_No_Op")
def bench_np_noop(benchmark):
    if pytest.numpy:
        benchmark.pedantic(np.get_include, rounds=pytest.trials)

        benchmark.extra_info["description"] = \
            "Measures the performance of np.get_include for a basic round-trip time comparison" \
            "with ak.client._no_op"
        benchmark.extra_info["problem_size"] = "N/A"
        benchmark.extra_info["transfer_rate"] = "N/A"
