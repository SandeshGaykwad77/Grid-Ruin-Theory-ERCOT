"""
figures — Publication-quality visualisations for Nature Energy.
===============================================================

Generates three core manuscript figures:

1. **Ruin Curve** (semi-log) — Empirical Monte Carlo ruin probability
   (scatter) versus the Lundberg upper bound :math:`e^{-R_g S}` and
   the Cramér asymptotic :math:`C e^{-R_g S}`.  When using i.i.d.
   resampling, the empirical points should fall *below* the Lundberg
   bound (validating the theory); when using block bootstrap (preserving
   autocorrelation), the points rise *above*, demonstrating the
   breakdown of the independence assumption.

2. **Hill Plot** — Tail-index estimate :math:`\\hat{\\alpha}(k)` versus
   the number of upper-order statistics *k*, with a regime-separation
   line at :math:`\\alpha = 4`.

3. **Overbuild–Storage Frontier** — Required storage capacity
   :math:`S^*` to meet a target blackout probability as a function of
   the renewable overbuild multiplier.

Style conventions
-----------------
* Minimalist scientific aesthetic (``seaborn-v0_8-paper`` base).
* No top/right spines.
* LaTeX rendering for axis labels where available.
* 300 DPI minimum; vector PDF as primary output.
* Tight layout with explicit padding.

All figure functions accept an optional *save_dir* argument and write
both ``.pdf`` and ``.png`` to ``<save_dir>/``.

Public API
----------
configure_style           Apply the global matplotlib style.
plot_ruin_curve           Figure 1 — semi-log ruin comparison.
plot_hill                 Figure 2 — Hill-estimator stability plot.
plot_overbuild_frontier   Figure 3 — overbuild vs. storage tradeoff.
generate_all_figures      Convenience wrapper for the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import ruin_theory
from .validation import ValidationResult


# ===================================================================== #
#  Global style                                                          #
# ===================================================================== #

_DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def configure_style() -> None:
    """Apply a minimalist scientific figure style globally.

    Attempts ``seaborn-v0_8-paper`` first, falls back to ``seaborn-paper``
    for older matplotlib versions, then applies custom overrides for a
    clean, Nature-compatible aesthetic.
    """
    for candidate in ("seaborn-v0_8-paper", "seaborn-paper"):
        if candidate in plt.style.available:
            plt.style.use(candidate)
            break

    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    try:
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    except Exception:
        pass


def _ensure_dir(path: Path) -> Path:
    """Create the directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, name: str, save_dir: Path) -> list[Path]:
    """Save a figure as both PDF and PNG, return the saved paths."""
    save_dir = _ensure_dir(save_dir)
    paths = []
    for ext in ("pdf", "png"):
        p = save_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    return paths


# ===================================================================== #
#  Colour palette                                                        #
# ===================================================================== #

_C_BOUND = "#2166AC"
_C_ASYMPTOTIC = "#4393C3"
_C_EMPIRICAL = "#B2182B"
_C_THRESHOLD = "#878787"
_C_FRONTIER = "#1B7837"
_C_HILL_LINE = "#2166AC"
_C_HILL_CI = "#92C5DE"


# ===================================================================== #
#  Figure 1: Ruin Curve (semi-log)                                       #
# ===================================================================== #

def plot_ruin_curve(
    result: ValidationResult,
    save_dir: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot empirical vs. theoretical ruin probability on a semi-log scale.

    Draws two theoretical references:
    * **Lundberg bound** :math:`e^{-R_g S}` (solid) — the strict upper
      bound under the i.i.d. assumption.
    * **Cramér asymptotic** :math:`C e^{-R_g S}` (dashed) — the exact
      large-*S* limit.

    Empirical Monte Carlo ruin probabilities are shown as scatter points.

    Parameters
    ----------
    result : ValidationResult
        Output of ``validation.compare_with_theory``.
    save_dir : Path, optional
        Directory to write PDF/PNG.  Defaults to ``../figures/``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_style()
    save_dir = Path(save_dir) if save_dir else _DEFAULT_FIG_DIR

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    S = result.storage_capacities

    S_fine = np.linspace(S.min(), S.max(), 500)
    bound_fine = np.clip(np.exp(-result.R_g * S_fine), 0, 1)
    asymp_fine = np.clip(result.C * np.exp(-result.R_g * S_fine), 0, 1)

    ax.semilogy(
        S_fine, bound_fine,
        color=_C_BOUND, linestyle="-", linewidth=1.6,
        label=r"Lundberg bound $e^{-R_g S}$",
    )

    ax.semilogy(
        S_fine, asymp_fine,
        color=_C_ASYMPTOTIC, linestyle="--", linewidth=1.3,
        label=r"Cram$\acute{\mathrm{e}}$r asymptotic $C\,e^{-R_g S}$",
    )

    mask = result.psi_empirical > 0
    resample_label = "i.i.d." if result.block_length == 1 else f"block({result.block_length})"
    ax.semilogy(
        S[mask], result.psi_empirical[mask],
        "o", color=_C_EMPIRICAL, markersize=5, markeredgewidth=0.5,
        markeredgecolor="white", zorder=5,
        label=f"Monte Carlo ({resample_label}, $N$={result.n_trajectories})",
    )

    ax.set_xlabel(r"Storage Capacity $S$ [MWh]")
    ax.set_ylabel(r"Ruin Probability $\psi(S)$")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    _save(fig, "fig1_ruin_curve", save_dir)
    return fig


# ===================================================================== #
#  Figure 2: Hill Plot                                                   #
# ===================================================================== #

def plot_hill(
    severities: np.ndarray,
    heavy_threshold: float = 4.0,
    save_dir: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the Hill estimator stability curve for tail-index diagnosis.

    The Hill plot shows :math:`\\hat{\\alpha}(k)` versus the number of
    upper-order statistics *k*.  A horizontal plateau indicates robust
    tail-index estimation.  A dashed threshold line separates the
    light-tailed regime (above) from the heavy-tailed regime (below).

    Parameters
    ----------
    severities : np.ndarray
        Drought-event severity values (positive).
    heavy_threshold : float
        Horizontal line separating light / heavy regimes.
    save_dir : Path, optional
        Output directory.  Defaults to ``../figures/``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_style()
    save_dir = Path(save_dir) if save_dir else _DEFAULT_FIG_DIR

    hill_df = ruin_theory.hill_plot_data(severities)
    if hill_df.empty:
        raise ValueError("Not enough drought events to construct a Hill plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    k = hill_df["k"].values
    alpha = hill_df["alpha"].values
    alpha_std = hill_df["alpha_std"].values

    ax.plot(k, alpha, color=_C_HILL_LINE, linewidth=1.4)
    ax.fill_between(
        k,
        alpha - 1.96 * alpha_std,
        alpha + 1.96 * alpha_std,
        color=_C_HILL_CI, alpha=0.3,
        label=r"95% CI",
    )

    ax.axhline(
        heavy_threshold, color=_C_THRESHOLD, linestyle="--", linewidth=1.0,
        label=rf"$\alpha = {heavy_threshold:.0f}$ threshold",
    )

    ax.set_xlabel(r"Number of Upper Order Statistics $k$")
    ax.set_ylabel(r"Tail Index $\hat{\alpha}(k)$")
    ax.legend(frameon=False)

    y_lo = max(0, np.nanmin(alpha - 2 * alpha_std) - 0.5)
    y_hi = min(np.nanmax(alpha + 2 * alpha_std) + 1.0, heavy_threshold * 3)
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    _save(fig, "fig2_hill_plot", save_dir)
    return fig


# ===================================================================== #
#  Figure 3: Overbuild–Storage Frontier                                  #
# ===================================================================== #

def _required_storage(
    R_g: float,
    C: float,
    target_psi: float,
) -> float:
    """Invert the Lundberg bound to find S* for a target ruin probability.

    Uses the Lundberg upper bound :math:`\\psi(S) \\le e^{-R_g S}` to
    compute the *minimum guaranteed* storage:

    .. math::

        S^* = -\\frac{\\ln \\psi}{R_g}

    Parameters
    ----------
    R_g : float        Grid Lundberg coefficient.
    C : float          Cramér constant (unused — bound-based inversion).
    target_psi : float Target ruin probability.

    Returns
    -------
    float
        Required storage capacity S* (MWh), or ``np.nan`` if infeasible.
    """
    if R_g <= 0 or target_psi <= 0 or target_psi >= 1:
        return np.nan
    return -np.log(target_psi) / R_g


def compute_overbuild_frontier(
    generation: np.ndarray,
    demand: np.ndarray,
    overbuild_range: np.ndarray,
    target_psi: float = 0.01,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
) -> pd.DataFrame:
    """Compute the overbuild–storage frontier for a fixed blackout target.

    For each overbuild multiplier *m*, scales the generation vector by
    *m*, recomputes the Lundberg coefficient, and inverts the Lundberg
    bound to find the required storage :math:`S^*` such that
    :math:`\\psi(S^*) \\le \\epsilon`.

    Parameters
    ----------
    generation : np.ndarray
        Base hourly generation vector G_t (MW).
    demand : np.ndarray
        Hourly demand vector D_t (MW).
    overbuild_range : array_like
        Vector of overbuild multipliers (e.g. 1.0 … 3.0).
    target_psi : float
        Target ruin (blackout) probability.
    eta_charge : float
        Charging efficiency.
    eta_discharge : float
        Discharging efficiency.

    Returns
    -------
    pd.DataFrame
        Columns ``overbuild``, ``R_g``, ``C``, ``S_star_mwh``,
        ``mean_surplus``.  Rows with infeasible configurations have
        ``NaN`` in the storage column.
    """
    overbuild_range = np.asarray(overbuild_range, dtype=np.float64)
    records = []

    for m in overbuild_range:
        x = ruin_theory.net_surplus(
            m * generation, demand,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
        )

        mean_x = float(np.mean(x))

        if mean_x <= 0:
            records.append({
                "overbuild": m,
                "R_g": np.nan,
                "C": np.nan,
                "S_star_mwh": np.nan,
                "mean_surplus": mean_x,
            })
            continue

        try:
            analysis = ruin_theory.lundberg_analysis(x)
            S_star = _required_storage(analysis.R_g, analysis.C, target_psi)
        except ValueError:
            S_star = np.nan
            analysis = None

        records.append({
            "overbuild": m,
            "R_g": analysis.R_g if analysis else np.nan,
            "C": analysis.C if analysis else np.nan,
            "S_star_mwh": S_star,
            "mean_surplus": mean_x,
        })

    return pd.DataFrame(records)


def plot_overbuild_frontier(
    frontier_df: pd.DataFrame,
    target_psi: float = 0.01,
    save_dir: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the overbuild-vs-storage tradeoff frontier.

    Shows the monotonically decreasing relationship between renewable
    overbuild and the storage required to achieve a fixed blackout
    probability target.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of ``compute_overbuild_frontier``.
    target_psi : float
        The blackout-probability target (for annotation).
    save_dir : Path, optional
        Output directory.  Defaults to ``../figures/``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    configure_style()
    save_dir = Path(save_dir) if save_dir else _DEFAULT_FIG_DIR

    valid = frontier_df.dropna(subset=["S_star_mwh"])
    if valid.empty:
        raise ValueError(
            "No feasible overbuild–storage points.  Check that the "
            "overbuild range produces positive mean surplus."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    ax.plot(
        valid["overbuild"], valid["S_star_mwh"],
        "o-", color=_C_FRONTIER, linewidth=1.6,
        markersize=4, markeredgecolor="white", markeredgewidth=0.4,
    )

    psi_exp = int(np.log10(target_psi))
    psi_str = rf"$\psi \leq 10^{{{psi_exp}}}$"
    ax.annotate(
        psi_str,
        xy=(valid["overbuild"].iloc[-1], valid["S_star_mwh"].iloc[-1]),
        xytext=(12, 8), textcoords="offset points",
        fontsize=8, color=_C_FRONTIER,
    )

    ax.set_xlabel("Renewable Overbuild Multiplier")
    ax.set_ylabel(r"Required Storage $S^*$ [MWh]")

    fig.tight_layout()
    _save(fig, "fig3_overbuild_frontier", save_dir)
    return fig


# ===================================================================== #
#  Convenience: generate all figures from raw data                       #
# ===================================================================== #

def generate_all_figures(
    generation: np.ndarray,
    demand: np.ndarray,
    storage_capacities: np.ndarray | None = None,
    overbuild_range: np.ndarray | None = None,
    target_psi: float = 0.01,
    n_trajectories: int = 5000,
    eta_charge: float = 0.95,
    eta_discharge: float = 0.95,
    seed: int = 0,
    save_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """Generate all three manuscript figures from raw generation/demand data.

    This is a convenience wrapper that chains the full pipeline:
    data -> ruin analysis -> Monte Carlo validation -> plots.

    The Monte Carlo validation uses **i.i.d. resampling**
    (``block_length=1``) so that the empirical points validate the
    Lundberg bound directly.

    Parameters
    ----------
    generation : np.ndarray   Hourly generation G_t (MW).
    demand : np.ndarray       Hourly demand D_t (MW).
    storage_capacities : np.ndarray, optional
        Storage sizes for the ruin curve.  Defaults to a range from
        500 to 8000 MWh.
    overbuild_range : np.ndarray, optional
        Overbuild multipliers for the frontier plot.  Defaults to
        1.1 … 3.0.
    target_psi : float        Blackout-probability target for frontier.
    n_trajectories : int      Monte Carlo paths.
    eta_charge : float        Charging efficiency.
    eta_discharge : float     Discharging efficiency.
    seed : int                Random seed.
    save_dir : Path, optional Output directory.

    Returns
    -------
    dict[str, Figure]
        Keys: ``ruin_curve``, ``hill_plot``, ``overbuild_frontier``.
    """
    from . import validation

    save_dir = Path(save_dir) if save_dir else _DEFAULT_FIG_DIR

    x = ruin_theory.net_surplus(
        generation, demand,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
    )

    if storage_capacities is None:
        storage_capacities = np.linspace(500, 8000, 16)
    if overbuild_range is None:
        overbuild_range = np.arange(1.1, 3.05, 0.1)

    val_result = validation.compare_with_theory(
        x, storage_capacities,
        n_trajectories=n_trajectories,
        block_length=1,
        seed=seed,
    )

    events = ruin_theory.extract_drought_events(x)
    severities = np.array([e["severity"] for e in events])

    frontier_df = compute_overbuild_frontier(
        generation, demand, overbuild_range,
        target_psi=target_psi,
        eta_charge=eta_charge,
        eta_discharge=eta_discharge,
    )

    figs = {}
    figs["ruin_curve"] = plot_ruin_curve(val_result, save_dir=save_dir)
    figs["hill_plot"] = plot_hill(severities, save_dir=save_dir)
    figs["overbuild_frontier"] = plot_overbuild_frontier(
        frontier_df, target_psi=target_psi, save_dir=save_dir,
    )

    return figs
