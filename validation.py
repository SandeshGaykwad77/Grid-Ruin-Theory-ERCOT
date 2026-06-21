"""
validation — Monte Carlo simulation engine for Cramér–Lundberg verification.
=============================================================================

Empirically validates the analytical ruin-probability bounds by directly
simulating the classical surplus process over thousands of independent
trajectories.

Two theoretical references are computed for each storage capacity *S*:

1. **Lundberg upper bound**: :math:`e^{-R_g S}` — a strict inequality
   that holds for all *S* under the i.i.d. assumption.
2. **Cramér asymptotic**: :math:`C \\, e^{-R_g S}` — the exact
   large-*S* limit, where :math:`C` is the Cramér constant.

The module supports two resampling modes:

* **i.i.d. resampling** (``block_length=1``) — shuffles the net-surplus
  vector uniformly at random, matching the independence assumption of
  the Cramér–Lundberg theory.  Used to validate that the analytical
  bound is correct.

* **Circular Block Bootstrap** (``block_length > 1``, Politis & Romano
  1992) — resamples contiguous blocks to preserve the Markovian weather
  autocorrelation.  Used to demonstrate that real-world temporal
  dependence causes the i.i.d.-based bound to *underestimate* ruin.

Surplus dynamics
----------------
Each trajectory starts at initial surplus :math:`U_0 = S` and evolves
as an unrestricted random walk (no upper capacity clamp):

.. math::

    U_{t+1} = U_t + X_t

Ruin occurs the first time :math:`U_t \\le 0`.  This matches the
classical actuarial surplus process for which the Cramér–Lundberg bound
is derived.

Public API
----------
optimal_block_length           Cube-root heuristic for block size.
circular_block_bootstrap       Generate one resampled trajectory.
simulate_trajectories          Vectorised multi-path surplus simulation.
empirical_ruin_probability     Ruin fraction for a single capacity.
ruin_probability_curve         Ruin fractions across a capacity vector.
compare_with_theory            Side-by-side empirical vs. analytical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ruin_theory


# ===================================================================== #
#  Data structures                                                       #
# ===================================================================== #

@dataclass(frozen=True)
class ValidationResult:
    """Comparison of empirical and theoretical ruin probabilities.

    Attributes
    ----------
    storage_capacities : np.ndarray
        Vector of storage sizes S (MWh).
    psi_empirical : np.ndarray
        Monte Carlo ruin probabilities.
    psi_lundberg_bound : np.ndarray
        Lundberg upper bound exp(-R_g S).
    psi_cramer_asymptotic : np.ndarray
        Cramér asymptotic C exp(-R_g S).
    R_g : float
        Grid Lundberg coefficient.
    C : float
        Cramér asymptotic constant.
    n_trajectories : int
        Number of Monte Carlo paths per capacity.
    block_length : int
        Bootstrap block length used (1 = i.i.d.).
    trajectory_length : int
        Number of hourly steps per trajectory.
    """
    storage_capacities: np.ndarray
    psi_empirical: np.ndarray
    psi_lundberg_bound: np.ndarray
    psi_cramer_asymptotic: np.ndarray
    R_g: float
    C: float
    n_trajectories: int
    block_length: int
    trajectory_length: int


# ===================================================================== #
#  Block bootstrap                                                       #
# ===================================================================== #

def optimal_block_length(n: int) -> int:
    """Compute the optimal block length via the cube-root heuristic.

    For a stationary time series of length *n*, the asymptotically
    optimal block length for the circular block bootstrap scales as
    :math:`n^{1/3}` (Hall, Horowitz & Jing, 1995).

    Parameters
    ----------
    n : int   Length of the original time series.

    Returns
    -------
    int
        Block length :math:`\\lceil n^{1/3} \\rceil`, minimum 1.
    """
    return max(1, int(np.ceil(n ** (1 / 3))))


def circular_block_bootstrap(
    x: np.ndarray,
    trajectory_length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single resampled trajectory via circular block bootstrap.

    The original series *x* is treated as periodic (circular): blocks
    that start near the end wrap around to the beginning.  Blocks of
    length *block_length* are drawn at uniformly random starting
    positions and concatenated until the desired *trajectory_length* is
    reached, then truncated to exactly that length.

    Parameters
    ----------
    x : np.ndarray
        Original net-surplus vector (1-D).
    trajectory_length : int
        Desired length of the resampled trajectory.
    block_length : int
        Number of consecutive observations per block.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray
        Resampled trajectory of length *trajectory_length*.
    """
    n = len(x)
    n_blocks = int(np.ceil(trajectory_length / block_length))

    starts = rng.integers(0, n, size=n_blocks)

    indices = np.concatenate([
        np.arange(s, s + block_length) % n for s in starts
    ])

    return x[indices[:trajectory_length]]


def _batch_circular_block_bootstrap(
    x: np.ndarray,
    trajectory_length: int,
    block_length: int,
    n_trajectories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate all resampled trajectories in a single vectorised pass.

    Produces an (n_trajectories, trajectory_length) matrix of resampled
    indices, then fancy-indexes into *x* once.  This avoids Python-level
    loops over trajectories.

    Parameters
    ----------
    x : np.ndarray          Original net-surplus vector.
    trajectory_length : int Length of each resampled path.
    block_length : int      Block size for the circular bootstrap.
    n_trajectories : int    Number of independent paths.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Shape (n_trajectories, trajectory_length).
    """
    n = len(x)
    n_blocks = int(np.ceil(trajectory_length / block_length))

    starts = rng.integers(0, n, size=(n_trajectories, n_blocks))

    offsets = np.arange(block_length)
    all_indices = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % n
    all_indices = all_indices.reshape(n_trajectories, -1)[:, :trajectory_length]

    return x[all_indices]


# ===================================================================== #
#  Surplus-process simulation                                            #
# ===================================================================== #

def simulate_trajectories(
    x: np.ndarray,
    storage_capacity: float,
    n_trajectories: int = 5000,
    trajectory_length: int | None = None,
    block_length: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Simulate surplus-process trajectories via block bootstrap.

    The surplus process is **unrestricted above** (no upper capacity
    clamp) to match the classical Cramér–Lundberg model.  The initial
    surplus is :math:`U_0 = S` and evolves as

    .. math::

        U_{t+1} = U_t + X_t

    Ruin occurs when :math:`U_t \\le 0`.

    Parameters
    ----------
    x : np.ndarray
        Historical net-surplus vector used for resampling.
    storage_capacity : float
        Initial surplus level S (MWh).
    n_trajectories : int
        Number of independent Monte Carlo paths.
    trajectory_length : int, optional
        Steps per trajectory.  Defaults to ``len(x)``.
    block_length : int, optional
        Circular block bootstrap block size.  Defaults to the cube-root
        heuristic on ``len(x)``.  Use ``1`` for i.i.d. resampling.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Surplus matrix of shape (n_trajectories, trajectory_length + 1),
        including the initial state at index 0.
    """
    rng = np.random.default_rng(seed)

    if trajectory_length is None:
        trajectory_length = len(x)
    if block_length is None:
        block_length = optimal_block_length(len(x))

    increments = _batch_circular_block_bootstrap(
        x, trajectory_length, block_length, n_trajectories, rng
    )

    cumulative = np.cumsum(increments, axis=1)

    surplus = np.empty((n_trajectories, trajectory_length + 1), dtype=np.float64)
    surplus[:, 0] = storage_capacity
    surplus[:, 1:] = storage_capacity + cumulative

    return surplus


# ===================================================================== #
#  Empirical ruin probability                                            #
# ===================================================================== #

def empirical_ruin_probability(
    x: np.ndarray,
    storage_capacity: float,
    n_trajectories: int = 5000,
    trajectory_length: int | None = None,
    block_length: int | None = None,
    seed: int = 0,
) -> float:
    """Estimate the ruin probability for a single storage capacity.

    Ruin is defined as the surplus reaching zero or below at any point
    during the trajectory.

    Parameters
    ----------
    x : np.ndarray           Historical net-surplus vector.
    storage_capacity : float Initial surplus S (MWh).
    n_trajectories : int     Number of Monte Carlo paths.
    trajectory_length : int, optional  Steps per path.
    block_length : int, optional       Bootstrap block size (1 = i.i.d.).
    seed : int               Random seed.

    Returns
    -------
    float
        Fraction of trajectories in which ruin occurred.
    """
    soc = simulate_trajectories(
        x, storage_capacity,
        n_trajectories=n_trajectories,
        trajectory_length=trajectory_length,
        block_length=block_length,
        seed=seed,
    )

    ruined = np.any(soc[:, 1:] <= 0, axis=1)
    return float(np.mean(ruined))


def ruin_probability_curve(
    x: np.ndarray,
    storage_capacities: np.ndarray,
    n_trajectories: int = 5000,
    trajectory_length: int | None = None,
    block_length: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Compute empirical ruin probabilities across a range of capacities.

    Parameters
    ----------
    x : np.ndarray
        Historical net-surplus vector.
    storage_capacities : array_like
        Vector of battery capacities S (MWh).
    n_trajectories : int
        Number of Monte Carlo paths per capacity.
    trajectory_length : int, optional
        Steps per path.
    block_length : int, optional
        Bootstrap block size (1 = i.i.d.).
    seed : int
        Base random seed (incremented per capacity for independence).

    Returns
    -------
    np.ndarray
        Empirical ruin probabilities, same length as *storage_capacities*.
    """
    storage_capacities = np.asarray(storage_capacities, dtype=np.float64)
    psi = np.empty(len(storage_capacities))

    for i, S in enumerate(storage_capacities):
        psi[i] = empirical_ruin_probability(
            x, S,
            n_trajectories=n_trajectories,
            trajectory_length=trajectory_length,
            block_length=block_length,
            seed=seed + i,
        )

    return psi


# ===================================================================== #
#  Theory comparison                                                     #
# ===================================================================== #

def compare_with_theory(
    x: np.ndarray,
    storage_capacities: np.ndarray,
    n_trajectories: int = 5000,
    trajectory_length: int | None = None,
    block_length: int | None = None,
    seed: int = 0,
) -> ValidationResult:
    """Run Monte Carlo validation and compare against Cramér–Lundberg.

    Performs the full pipeline:
    1. Compute the analytical Lundberg coefficient and Cramér constant.
    2. For each storage capacity, run bootstrap Monte Carlo.
    3. Package empirical results alongside both theoretical curves.

    Parameters
    ----------
    x : np.ndarray
        Historical net-surplus vector.
    storage_capacities : array_like
        Vector of battery capacities S (MWh).
    n_trajectories : int
        Number of Monte Carlo paths per capacity.
    trajectory_length : int, optional
        Steps per path (defaults to ``len(x)``).
    block_length : int, optional
        Bootstrap block size.  Use ``1`` for i.i.d. resampling (to
        validate the Lundberg bound) or ``None`` for the cube-root
        heuristic (to test the autocorrelation effect).
    seed : int
        Base random seed.

    Returns
    -------
    ValidationResult
        Frozen dataclass with empirical and both theoretical curves.
    """
    storage_capacities = np.asarray(storage_capacities, dtype=np.float64)

    analysis = ruin_theory.lundberg_analysis(x)

    if trajectory_length is None:
        trajectory_length = len(x)
    if block_length is None:
        block_length = optimal_block_length(len(x))

    psi_emp = ruin_probability_curve(
        x, storage_capacities,
        n_trajectories=n_trajectories,
        trajectory_length=trajectory_length,
        block_length=block_length,
        seed=seed,
    )

    psi_bound = np.clip(np.exp(-analysis.R_g * storage_capacities), 0.0, 1.0)
    psi_asymp = ruin_theory.ruin_probability(
        storage_capacities, analysis.R_g, analysis.C
    )

    return ValidationResult(
        storage_capacities=storage_capacities,
        psi_empirical=psi_emp,
        psi_lundberg_bound=psi_bound,
        psi_cramer_asymptotic=psi_asymp,
        R_g=analysis.R_g,
        C=analysis.C,
        n_trajectories=n_trajectories,
        block_length=block_length,
        trajectory_length=trajectory_length,
    )


def summary_table(result: ValidationResult) -> pd.DataFrame:
    """Format a validation result as a readable comparison table.

    Parameters
    ----------
    result : ValidationResult
        Output of ``compare_with_theory``.

    Returns
    -------
    pd.DataFrame
        Columns: S, psi_empirical, psi_lundberg_bound,
        psi_cramer_asymptotic, bound_holds.
    """
    with np.errstate(divide="ignore"):
        log_emp = np.where(
            result.psi_empirical > 0,
            np.log10(result.psi_empirical),
            np.nan,
        )
        log_bound = np.log10(np.maximum(result.psi_lundberg_bound, 1e-300))

    return pd.DataFrame({
        "S_mwh": result.storage_capacities,
        "psi_empirical": result.psi_empirical,
        "psi_lundberg_bound": result.psi_lundberg_bound,
        "psi_cramer_asymptotic": result.psi_cramer_asymptotic,
        "bound_holds": result.psi_empirical <= result.psi_lundberg_bound,
        "log10_psi_empirical": log_emp,
        "log10_psi_bound": log_bound,
    })
