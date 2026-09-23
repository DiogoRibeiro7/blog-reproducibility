"""Use a non-interactive rendering backend on every platform.

Under pytest-xdist each worker is its own process, so the workers already use
every core. Letting each of them also start a BLAS, OpenMP, and joblib pool the
size of the machine oversubscribes it many times over, which is slower than a
serial run. Workers are therefore limited to one thread each, before NumPy or
scikit-learn load their thread pools.
"""

import os

if "PYTEST_XDIST_WORKER" in os.environ:
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    # joblib sizes n_jobs=-1 from this count.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
