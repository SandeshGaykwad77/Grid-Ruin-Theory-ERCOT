"""
ruin_theory — Cramér–Lundberg analysis engine for grid storage.
================================================================

Maps classical actuarial ruin theory onto a renewable-energy storage
system.  The battery state of charge plays the role of the insurer's
surplus; generation surpluses are "premiums" and demand deficits are
"claims."

Key quantities
--------------

**Net surplus increment** (per hour):

.. math::

    X_t = \\eta_c \\, G_t  -  \\frac{D_t}{\\eta_d}

where :math:`\\eta_c` is the charging efficiency and :math:`\\eta_d` is
the discharging efficiency.  The asymmetry reflects the physical reality
that energy is lost both when storing and when retrieving.

**Grid Lundberg Coefficient** :math:`R_g`:

The unique positive root of the moment-generating-function (MGF) equation

.. math::

    \\mathbb{E}\\bigl[e^{-R_g \\, X}\\bigr] = 1

When :math:`\\mathbb{E}[X] > 0` (system is net-surplus on average) and
the MGF is finite in a neighborhood of the origin, a unique positive
:math:`R_g` exists by convexity of the MGF.

**Ruin probability (Cramér–Lundberg bound)**:

.. math::

    \\psi(S) \\;\\le\\; C \\, e^{-R_g \\, S}

where :math:`S` is the storage capacity and

.. math::

    C = \\frac{\\mathbb{E}[X]}{\\mathbb{E}[X \\, e^{-R_g X}]}  \\cdot  R_g^{-1}

is the Cramér asymptotic constant (exact prefactor for the compound-
Poisson / renewal limit).

**Tail-index estimation** (Hill estimator):

Applied to drought-event severities (cumulative deficit during
consecutive-deficit periods) to test whether the deficit distribution
is light-tailed (finite MGF → exponential ruin decay) or heavy-tailed
(infinite MGF → Embrechts–Veraverbeke power-law ruin).

Public API
----------
net_surplus               Compute X_t vector with round-trip efficiency.
empirical_mgf             Evaluate E[exp(-r X)] for a given r.
find_lundberg_coefficient Solve for R_g via Brent's method.
cramer_asymptotic_const   Compute the Cramér constant C.
ruin_probability          Full ψ(S) = C exp(-R_g S).
extract_drought_events    Identify consecutive-deficit episodes.
hill_estimator            Estimate tail index α from drought severities.
tail_classification       Classify light vs. heavy tail from Hill estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize


# ===================================================================== #
#  Data structures                                                       #
# ===================================================================== #

@dataclass(frozen=True)
class LundbergResult:
    """Container for the full Cramér–Lundberg analysis output.

    Attributes
    ----------
    R_g : float
        Grid Lundberg coefficient (positive root of the MGF equation).
    C : float
        Cramér asymptotic constant (ruin-probability prefactor).
    mean_surplus : float
        Mean net surplus E[X] (must be > 0 for the bound to apply).
    mgf_at_Rg : float
        Value of E[exp(-R_g X)] at the solution (should be ≈ 1).
    """
    R_g: float
    C: float
    mean_surplus: float
    mgf_at_Rg: float


@dataclass(frozen=True)
class TailResult:
    """Container for Hill-estimator tail analysis.

    Attributes
    ----------
    alpha : float
        Estimated tail index (Pareto exponent).
    alpha_std : float
        Asymptotic standard error of the Hill estimator.
    k : int
        Number of upper-order statistics used.
    is_heavy : bool
        True if the estimated tail is heavy (α finite and small enough
        that the MGF diverges), suggesting power-law ruin scaling.
    classification : str
        Human-readable label: 'light' or 'heavy (α ≈ …)'.
    """
    alpha: float
    alpha_std: float
    k: int
    is_heavy: bool
    classification: str


# ===================================================================== #
#  Net surplus                                                           #
# ===================================================================== #

def net_surplus(
    generation: np.ndarray,
    demand: np.ndarray,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
) -> np.ndarray:
    """Compute the hourly net surplus vector X_t with round-trip losses.

    The asymmetric efficiency model penalises both injection into storage
    and withdrawal from storage:

    .. math::

        X_t = \\eta_c \\, G_t  \\;-\\;  \\frac{D_t}{\\eta_d}

    When the system is in surplus (:math:`X_t > 0`), only the fraction
    :math:`\\eta_c` of generation is actually captured.  When the system
    is in deficit (:math:`X_t < 0`), the required withdrawal is amplified
    by :math:`1/\\eta_d` to deliver the demanded power.

    Parameters
    ----------
    generation : array_like   Hourly generation G_t (MW).
    demand : array_like       Hourly demand D_t (MW).
    eta_charge : float        Charging (injection) efficiency, 0 < η_c ≤ 1.
    eta_discharge : float     Discharging (withdrawal) efficiency, 0 < η_d ≤ 1.

    Returns
    -------
    np.ndarray
        Net surplus vector X_t (MW·h per hour, i.e. MWh).

    Raises
    ------
    ValueError
        If efficiencies are outside (0, 1].
    """
    if not (0 < eta_charge <= 1):
        raise ValueError(f"eta_charge must be in (0, 1], got {eta_charge}")
    if not (0 < eta_discharge <= 1):
        raise ValueError(f"eta_discharge must be in (0, 1], got {eta_discharge}")

    generation = np.asarray(generation, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)

    return eta_charge * generation - demand / eta_discharge


# ===================================================================== #
#  Moment-generating function and Lundberg coefficient                   #
# ===================================================================== #

def empirical_mgf(x: np.ndarray, r: float) -> float:
    """Evaluate the empirical moment-generating function E[exp(-r X)].

    Uses the log-sum-exp trick for numerical stability when *r* is large
    or the surplus vector contains extreme values.

    Parameters
    ----------
    x : np.ndarray   Net surplus samples.
    r : float        Evaluation point (typically > 0).

    Returns
    -------
    float
        Sample mean of exp(-r X_i).  Returns ``np.inf`` when the result
        overflows (indicating the MGF diverges at this *r*).
    """
    exponents = -r * x
    max_exp = np.max(exponents)
    shifted = np.exp(exponents - max_exp)
    log_result = max_exp + np.log(np.mean(shifted))
    if log_result > 700:
        return np.inf
    return np.exp(log_result)


def _mgf_equation(r: float, x: np.ndarray) -> float:
    """Objective for root-finding: E[exp(-r X)] - 1 = 0."""
    return empirical_mgf(x, r) - 1.0


def find_lundberg_coefficient(
    x: np.ndarray,
    bracket_max: float = 1.0,
    tol: float = 1e-12,
    max_bracket_expansions: int = 20,
) -> float:
    """Find the Grid Lundberg Coefficient R_g via Brent's method.

    R_g is the unique positive root of

    .. math::

        \\mathbb{E}[e^{-R_g X}] = 1

    Existence requires :math:`\\mathbb{E}[X] > 0` (net positive drift) and
    a finite MGF in a neighborhood of the origin.  The function
    :math:`r \\mapsto \\mathbb{E}[e^{-rX}]` is convex with value 1 at
    :math:`r=0` and derivative :math:`-\\mathbb{E}[X] < 0` at the origin,
    so it initially decreases below 1 before rising back through 1 at
    :math:`r = R_g`.

    The algorithm adaptively expands the search bracket until a sign change
    is found, then applies Brent's method for guaranteed convergence.

    Parameters
    ----------
    x : np.ndarray
        Net surplus samples (must have positive mean).
    bracket_max : float
        Initial upper bound of the search interval [ε, bracket_max].
    tol : float
        Root-finding tolerance.
    max_bracket_expansions : int
        Maximum number of doublings to find a valid bracket.

    Returns
    -------
    float
        The Grid Lundberg coefficient R_g > 0.

    Raises
    ------
    ValueError
        If E[X] ≤ 0 (no positive root exists) or the bracket expansion
        fails to locate a sign change.
    """
    mean_x = np.mean(x)
    if mean_x <= 0:
        raise ValueError(
            f"Mean net surplus E[X] = {mean_x:.4f} ≤ 0.  The Lundberg "
            f"coefficient requires a net-positive drift (generation must "
            f"exceed demand on average after efficiency losses)."
        )

    r_lo = 1e-10

    f_lo = _mgf_equation(r_lo, x)
    if f_lo >= 0:
        raise ValueError(
            "MGF equation is non-negative at r ≈ 0.  This should not happen "
            "when E[X] > 0; check data for degenerate distributions."
        )

    r_hi = bracket_max
    for _ in range(max_bracket_expansions):
        f_hi = _mgf_equation(r_hi, x)
        if f_hi > 0:
            break
        r_hi *= 2.0
    else:
        raise ValueError(
            f"Could not bracket a positive root of the MGF equation after "
            f"expanding to r = {r_hi:.2e}.  The deficit distribution may be "
            f"too heavy-tailed for the MGF to exceed 1."
        )

    result = optimize.brentq(_mgf_equation, r_lo, r_hi, args=(x,), xtol=tol)
    return float(result)


# ===================================================================== #
#  Cramér asymptotic constant                                            #
# ===================================================================== #

def cramer_asymptotic_const(x: np.ndarray, R_g: float) -> float:
    """Compute the Cramér asymptotic constant C.

    In the classical Cramér–Lundberg model the exact ruin-probability
    asymptotics are

    .. math::

        \\psi(S) \\;\\sim\\; C \\, e^{-R_g S}, \\qquad S \\to \\infty

    where

    .. math::

        C = \\frac{R_g \\, \\mathbb{E}[X]}
             {\\mathbb{E}\\bigl[-X \\, e^{-R_g X}\\bigr]}

    The denominator equals :math:`M'(R_g)` (derivative of the MGF
    evaluated at the adjustment coefficient) and is always positive
    at the root.

    Parameters
    ----------
    x : np.ndarray   Net surplus samples.
    R_g : float      Grid Lundberg coefficient.

    Returns
    -------
    float
        Cramér constant C > 0.
    """
    mean_x = np.mean(x)

    exponents = -R_g * x
    max_exp = np.max(exponents)
    stable_weights = np.exp(exponents - max_exp)

    denominator = np.exp(max_exp) * np.mean(-x * stable_weights)

    return (R_g * mean_x) / denominator


# ===================================================================== #
#  Full Cramér–Lundberg analysis                                         #
# ===================================================================== #

def lundberg_analysis(
    x: np.ndarray,
    bracket_max: float = 1.0,
    tol: float = 1e-12,
) -> LundbergResult:
    """Run the complete Cramér–Lundberg analysis pipeline.

    Parameters
    ----------
    x : np.ndarray      Net surplus samples.
    bracket_max : float  Initial bracket upper bound for root search.
    tol : float          Root-finding tolerance.

    Returns
    -------
    LundbergResult
        Frozen dataclass with R_g, C, mean_surplus, and a verification
        MGF value.
    """
    R_g = find_lundberg_coefficient(x, bracket_max=bracket_max, tol=tol)
    C = cramer_asymptotic_const(x, R_g)
    mean_x = float(np.mean(x))
    mgf_check = empirical_mgf(x, R_g)

    return LundbergResult(R_g=R_g, C=C, mean_surplus=mean_x, mgf_at_Rg=mgf_check)


def ruin_probability(S: np.ndarray, R_g: float, C: float) -> np.ndarray:
    """Evaluate the Cramér–Lundberg ruin probability ψ(S) = C exp(-R_g S).

    Parameters
    ----------
    S : array_like   Storage capacities (MWh) at which to evaluate.
    R_g : float      Grid Lundberg coefficient.
    C : float        Cramér asymptotic constant.

    Returns
    -------
    np.ndarray
        Ruin (blackout) probabilities, clipped to [0, 1].
    """
    S = np.asarray(S, dtype=np.float64)
    psi = C * np.exp(-R_g * S)
    return np.clip(psi, 0.0, 1.0)


# ===================================================================== #
#  Drought-event extraction                                              #
# ===================================================================== #

def extract_drought_events(x: np.ndarray) -> list[dict]:
    """Identify consecutive-deficit episodes in the surplus process.

    A "drought" begins when X_t < 0 and ends when X_t ≥ 0.  For each
    episode the function records the start index, duration (hours), and
    cumulative severity (total energy deficit, always positive).

    Parameters
    ----------
    x : np.ndarray   Net surplus vector.

    Returns
    -------
    list of dict
        Each dict has keys ``start`` (int), ``duration`` (int), and
        ``severity`` (float, sum of |X_t| during the deficit episode).
    """
    deficit_mask = x < 0
    events: list[dict] = []

    in_drought = False
    start = 0
    severity = 0.0

    for t, is_deficit in enumerate(deficit_mask):
        if is_deficit:
            if not in_drought:
                in_drought = True
                start = t
                severity = 0.0
            severity += abs(x[t])
        else:
            if in_drought:
                events.append({
                    "start": start,
                    "duration": t - start,
                    "severity": severity,
                })
                in_drought = False

    if in_drought:
        events.append({
            "start": start,
            "duration": len(x) - start,
            "severity": severity,
        })

    return events


# ===================================================================== #
#  Hill estimator for tail-index α                                       #
# ===================================================================== #

def hill_estimator(
    severities: np.ndarray,
    k: Optional[int] = None,
) -> TailResult:
    """Estimate the Pareto tail index α via the Hill estimator.

    For order statistics :math:`X_{(1)} \\ge X_{(2)} \\ge \\cdots \\ge X_{(n)}`
    the Hill estimator is

    .. math::

        \\hat{\\alpha}_k^{-1} = \\frac{1}{k} \\sum_{i=1}^{k}
            \\ln X_{(i)} - \\ln X_{(k+1)}

    A finite :math:`\\alpha < 2` indicates a heavy-tailed (sub-exponential)
    deficit distribution; the Embrechts–Veraverbeke theorem then implies
    ruin probability decays as a *power law* rather than exponentially:

    .. math::

        \\psi(S) \\sim S^{1-\\alpha} \\bar{F}(S)

    Parameters
    ----------
    severities : array_like
        Drought-event severities (must be strictly positive).
    k : int, optional
        Number of upper order statistics to use.  If ``None``, defaults
        to ``int(√n)`` (a standard heuristic balancing bias and variance).

    Returns
    -------
    TailResult
        Frozen dataclass with α, its standard error, k, and
        classification.

    Raises
    ------
    ValueError
        If fewer than 10 severity values are provided (too few for
        reliable estimation) or if any severity ≤ 0.
    """
    severities = np.asarray(severities, dtype=np.float64)
    severities = severities[severities > 0]

    n = len(severities)
    if n < 10:
        raise ValueError(
            f"Need at least 10 positive drought severities for Hill "
            f"estimation, got {n}."
        )

    if k is None:
        k = int(np.sqrt(n))
    k = min(k, n - 1)
    k = max(k, 1)

    sorted_desc = np.sort(severities)[::-1]

    log_ratios = np.log(sorted_desc[:k]) - np.log(sorted_desc[k])
    gamma_hat = np.mean(log_ratios)          # 1 / α
    alpha_hat = 1.0 / gamma_hat if gamma_hat > 0 else np.inf

    alpha_std = alpha_hat / np.sqrt(k) if np.isfinite(alpha_hat) else np.inf

    is_heavy = np.isfinite(alpha_hat) and alpha_hat < 4.0

    if is_heavy:
        classification = f"heavy (α ≈ {alpha_hat:.2f})"
    else:
        classification = "light"

    return TailResult(
        alpha=float(alpha_hat),
        alpha_std=float(alpha_std),
        k=k,
        is_heavy=is_heavy,
        classification=classification,
    )


def hill_plot_data(
    severities: np.ndarray,
    k_range: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute Hill estimates across a range of k for a Hill plot.

    A Hill plot shows :math:`\\hat{\\alpha}(k)` versus *k*.  Stability
    of the estimate across a plateau of *k* values supports the
    Pareto-tail hypothesis.

    Parameters
    ----------
    severities : array_like
        Drought-event severities (positive).
    k_range : np.ndarray, optional
        Array of *k* values.  Defaults to 10 … n/2.

    Returns
    -------
    pd.DataFrame
        Columns ``k``, ``alpha``, ``alpha_std``.
    """
    severities = np.asarray(severities, dtype=np.float64)
    severities = severities[severities > 0]
    n = len(severities)

    if k_range is None:
        k_range = np.arange(10, max(11, n // 2))

    records = []
    for k_val in k_range:
        try:
            result = hill_estimator(severities, k=int(k_val))
            records.append({
                "k": result.k,
                "alpha": result.alpha,
                "alpha_std": result.alpha_std,
            })
        except ValueError:
            continue

    return pd.DataFrame(records)


# ===================================================================== #
#  Tail classification helper                                            #
# ===================================================================== #

def tail_classification(
    severities: np.ndarray,
    k: Optional[int] = None,
    heavy_threshold: float = 4.0,
) -> str:
    """Classify the deficit-severity tail and state implications for storage.

    Parameters
    ----------
    severities : array_like   Drought severities.
    k : int, optional         Hill estimator order-statistics count.
    heavy_threshold : float   α below this is classified as heavy-tailed.

    Returns
    -------
    str
        Multi-line summary describing the tail behaviour and its
        implications for the storage-scaling law.
    """
    result = hill_estimator(severities, k=k)

    if result.is_heavy:
        return (
            f"HEAVY TAIL DETECTED (α ≈ {result.alpha:.2f} ± "
            f"{result.alpha_std:.2f}, k = {result.k}).\n"
            f"The moment-generating function diverges, invalidating the "
            f"exponential Cramér–Lundberg bound.\n"
            f"By the Embrechts–Veraverbeke theorem, ruin probability decays "
            f"as a POWER LAW: ψ(S) ~ S^(1−α) F̄(S).\n"
            f"Implication: exponential storage scaling COLLAPSES; "
            f"dramatically more storage is needed than the Lundberg "
            f"coefficient suggests."
        )

    return (
        f"LIGHT TAIL (α ≈ {result.alpha:.2f} ± {result.alpha_std:.2f}, "
        f"k = {result.k}).\n"
        f"The moment-generating function is finite in a neighborhood of "
        f"the origin.\n"
        f"The Cramér–Lundberg exponential bound ψ(S) ≤ C exp(−R_g S) is "
        f"VALID.\n"
        f"Implication: storage capacity scales logarithmically with the "
        f"desired reliability level."
    )
