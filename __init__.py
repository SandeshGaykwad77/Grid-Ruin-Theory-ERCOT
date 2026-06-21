"""
Grid Ruin Theory — Cramér-Lundberg Analysis for Renewable Energy Storage
=========================================================================

Transplants classical actuarial ruin theory onto renewable-energy grid
storage to derive closed-form analytic bounds on blackout probability as
a function of storage capacity S:

    ψ(S) ≈ C · exp(−R_g · S)

where R_g is the Grid Lundberg Coefficient and C is the Cramér asymptotic
constant.  The package further tests whether generation-deficit durations
are light-tailed (exponential storage scaling) or heavy-tailed / sub-
exponential (power-law regime governed by the Embrechts–Veraverbeke
theorem).

Modules
-------
data_pipeline   Ingest and align hourly generation/demand data.
ruin_theory     Core mathematical engine (Lundberg coefficient, Hill
                estimator, Cramér constant).
validation      Monte Carlo simulation engine for empirical verification.
figures         Publication-quality matplotlib visualisations.
"""

__version__ = "0.1.0"
