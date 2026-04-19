"""
Bootstrap significance test.

Core idea: randomise the *return values* themselves (not their timing) to
simulate what chance alone could produce.

Steps
-----
1. Zero-centre the rule's daily returns by subtracting the observed mean.
   This enforces H0: the rule has no edge (expected return = 0).
2. Resample the zero-centred returns with replacement N times and compute
   the mean of each resample → the bootstrap sampling distribution.
3. p-value = fraction of simulated means ≥ the observed mean.
"""

import numpy as np
import ray


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

@ray.remote
def _ray_bootstrap_batch(
    centered: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """
    Run one batch of bootstrap simulations and return an array of simulated means.

    Parameters
    ----------
    centered   : zero-centred rule returns (enforces H0)
    batch_size : number of simulations in this batch
    seed       : per-batch random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(centered)
    idx = rng.integers(0, n, size=(batch_size, n))
    return centered[idx].mean(axis=1)


def run_bootstrap_test(
    rule_returns: np.ndarray,
    observed_mean: float,
    n_simulations: int,
    cpu_cores: int,
    random_seed: int,
    pbar=None,
    progress_callback=None,
) -> np.ndarray:
    """
    Distribute bootstrap simulations across Ray workers and return the full
    array of simulated means.

    Parameters
    ----------
    progress_callback : callable(batch_index, total_batches) or None
        Called each time a batch completes. batch_index is 1-based.
    """
    centered = rule_returns - observed_mean
    batch_sizes = _split_into_batches(n_simulations, cpu_cores)
    total_batches = len(batch_sizes)

    refs = [
        _ray_bootstrap_batch.remote(centered, batch_size, random_seed + i)
        for i, batch_size in enumerate(batch_sizes)
    ]

    return _collect_results(refs, pbar, progress_callback, total_batches)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_into_batches(n: int, k: int) -> list:
    """Divide n simulations into k roughly equal integer batches."""
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _collect_results(refs: list, pbar, progress_callback, total_batches: int) -> np.ndarray:
    parts = []
    remaining = list(refs)
    completed = 0
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1, timeout=0.5)
        for ref in done:
            parts.append(ray.get(ref))
            completed += 1
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None:
                try:
                    progress_callback(completed, total_batches)
                except Exception:
                    pass
    return np.concatenate(parts)
