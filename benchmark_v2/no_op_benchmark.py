import arkouda as ak
import numpy as np
import pytest

SECONDS = pytest.trials

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_No_Op",
                       max_time=SECONDS
                       )
def bench_ak_noop(benchmark):
    benchmark(ak.client._no_op)

    benchmark.extra_info["description"] = \
        "Measures the performance of ak.client._no_op for a basic round-trip time"
    benchmark.extra_info["problem_size"] = "N/A"
    benchmark.extra_info["transfer_rate"] = f"{benchmark.stats['rounds'] / benchmark.stats['total']:.4f} " \
                                            f"operations per second"

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_No_Op",
                       max_time=SECONDS
                       )
def bench_np_noop(benchmark):
    if pytest.numpy:
        benchmark(np.get_include)

        benchmark.extra_info["description"] = \
            "Measures the performance of np.get_include for a basic round-trip time comparison" \
            "with ak.client._no_op"
        benchmark.extra_info["problem_size"] = "N/A"
        benchmark.extra_info["transfer_rate"] = f"{benchmark.stats['rounds'] / benchmark.stats['total']:.4f} " \
                                                f"operations per second"
