import numpy as np
import pytest

import arkouda as ak


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_No_Op")
def bench_noop(benchmark):
    if pytest.numpy:

        def noop():
            np.get_include()

        backend = "NumPy"
        description = "Measures NumPy no-op (np.get_include)"
    else:

        def noop():
            ak.client._no_op()

        backend = "Arkouda"
        description = "Measures Arkouda no-op (ak.client._no_op)"

    benchmark.pedantic(noop, rounds=pytest.trials)

    benchmark.extra_info["description"] = description
    benchmark.extra_info["problem_size"] = "N/A"
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["transfer_rate"] = (
        f"{benchmark.stats['rounds'] / benchmark.stats['total']:.4f} operations per second"
    )
