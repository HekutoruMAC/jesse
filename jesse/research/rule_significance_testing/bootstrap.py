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
    # Draw all resample indices in a single vectorised call: shape (batch_size, n)
    idx = rng.integers(0, n, size=(batch_size, n))
    return centered[idx].mean(axis=1)   # shape (batch_size,)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def run_bootstrap_test(
    rule_returns: np.ndarray,
    observed_mean: float,
    n_simulations: int,
    cpu_cores: int,
    random_seed: int,
    pbar,
) -> np.ndarray:
    """
    Distribute the bootstrap simulations across Ray workers and return the
    full array of simulated means.

    Parameters
    ----------
    rule_returns   : per-bar returns of the rule (signal × detrended log-return)
    observed_mean  : mean(rule_returns), pre-computed by the caller
    n_simulations  : total number of bootstrap resamples
    cpu_cores      : number of Ray workers to use
    random_seed    : base seed; each batch gets seed + batch_index
    pbar           : tqdm progress bar or None
    """
    # Zero-centre the returns to enforce H0: E[return] = 0.
    # Without this step the resampling distribution would be centred on the
    # observed mean rather than on zero, which would bias the p-value.
    centered = rule_returns - observed_mean

    # Split n_simulations into cpu_cores equal-ish batches
    batch_sizes = _split_into_batches(n_simulations, cpu_cores)

    refs = [
        _ray_bootstrap_batch.remote(centered, batch_size, random_seed + i)
        for i, batch_size in enumerate(batch_sizes)
    ]

    return _collect_results(refs, pbar)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_into_batches(n: int, k: int) -> list:
    """Divide n simulations into k roughly equal integer batches."""
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _collect_results(refs: list, pbar) -> np.ndarray:
    """Collect Ray futures one-by-one (preserves progress-bar accuracy)."""
    parts = []
    remaining = list(refs)
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1, timeout=0.5)
        for ref in done:
            parts.append(ray.get(ref))
            if pbar is not None:
                pbar.update(1)
    return np.concatenate(parts)
