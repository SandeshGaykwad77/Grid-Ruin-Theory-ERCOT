"""
data_pipeline — Ingest, convert, and align real ERCOT hourly data.
===================================================================

This module is the master ingestion and conversion layer for the 10-year
ERCOT dataset (2014–2023) downloaded by ``ingest_ercot.py``.

Data sources
------------
* **Demand** — ``data/demand_ERCOT_{year}.csv``

  EIA Open Data v2 hourly demand for balancing authority ``ERCO``
  (2019–2023 direct; 2014–2018 back-cast from the 2019 climatology
  with a 1.8 %/yr compound load-growth adjustment).

  Schema: ``timestamp``, ``demand_mw``

* **Weather** — ``data/nrel_weather_{year}.csv``

  NASA POWER MERRA-2 reanalysis for central Texas (31.97°N, 99.90°W),
  stored under the ``nrel_weather_*`` naming convention for pipeline
  compatibility.  Note: NREL's ``developer.nrel.gov`` domain has been
  unreachable since 2026-06-21 (global DNS outage).

  Schema: ``timestamp``, ``ghi_wm2``, ``wind_speed_50m_ms``, ``temp_2m_c``

Capacity-factor model
---------------------
**Solar CF** — direct normalisation of global horizontal irradiance:

.. math::

    \\text{CF}_{\\text{solar},t} = \\operatorname{clip}\\!\\left(
        \\frac{\\text{GHI}_t}{1000}, 0, 1\\right)

**Wind CF** — piecewise-cubic power curve (IEC Class II turbine proxy):

.. math::

    \\text{CF}_{\\text{wind}}(v) = \\begin{cases}
        0 & v < 3 \\text{ m/s} \\\\
        \\left(\\dfrac{v - 3}{12 - 3}\\right)^{\\!3}
            & 3 \\le v \\le 12 \\text{ m/s} \\\\
        1 & 12 < v \\le 25 \\text{ m/s} \\\\
        0 & v > 25 \\text{ m/s (cut-out)}
    \\end{cases}

ERCOT nameplate capacities
--------------------------
Aligned with ERCOT's 2023 installed-capacity report:

* Solar: 25,000 MW (utility-scale + DG)
* Wind:  45,000 MW

Public API
----------
wind_power_curve              Vectorised piecewise-cubic turbine model.
compute_ercot_capacity_factors Convert raw weather DataFrame to CF columns.
load_ercot_year               Load and merge one year of demand + weather.
build_aligned_frame           Multi-year concatenation → master DataFrame.
generate_synthetic_data       Synthetic fallback for unit testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ===================================================================== #
#  Module-level defaults                                                 #
# ===================================================================== #

_DATA_DIR = Path(__file__).resolve().parent / "data"

ERCOT_SOLAR_CAPACITY_MW: float = 25_000.0
ERCOT_WIND_CAPACITY_MW: float = 45_000.0

_YEARS = list(range(2014, 2024))


# ===================================================================== #
#  Capacity-factor conversion                                            #
# ===================================================================== #

def wind_power_curve(wind_speed: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise-cubic wind turbine power curve.

    Implements a standard IEC Class II turbine proxy:

    * Cut-in  speed: 3 m/s
    * Rated   speed: 12 m/s
    * Cut-out speed: 25 m/s

    Between cut-in and rated speed the power fraction follows a cubic
    ramp, which approximates the aerodynamic :math:`v^3` dependence
    while remaining bounded in [0, 1].

    Parameters
    ----------
    wind_speed : array_like
        Wind speed at 50 m hub height (m/s).

    Returns
    -------
    np.ndarray
        Capacity factors in [0, 1].
    """
    v = np.asarray(wind_speed, dtype=np.float64)
    cf = np.zeros_like(v)

    ramp = (v >= 3) & (v <= 12)
    cf[ramp] = ((v[ramp] - 3.0) / (12.0 - 3.0)) ** 3

    rated = (v > 12) & (v <= 25)
    cf[rated] = 1.0

    return cf


def compute_ercot_capacity_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw NASA POWER weather columns to capacity-factor columns.

    Adds ``cf_solar`` and ``cf_wind`` columns to *df* in-place and
    returns a copy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ghi_wm2`` and ``wind_speed_50m_ms`` columns.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``cf_solar`` and ``cf_wind``.

    Raises
    ------
    KeyError
        If required weather columns are missing.
    """
    required = {"ghi_wm2", "wind_speed_50m_ms"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Weather DataFrame is missing required columns: {missing}.  "
            f"Run ingest_ercot.py to regenerate the CSV files."
        )

    out = df.copy()
    out["cf_solar"] = (out["ghi_wm2"] / 1000.0).clip(0.0, 1.0)
    out["cf_wind"] = wind_power_curve(out["wind_speed_50m_ms"].values)
    return out


# ===================================================================== #
#  Single-year loader                                                    #
# ===================================================================== #

def load_ercot_year(
    year: int,
    solar_capacity_mw: float = ERCOT_SOLAR_CAPACITY_MW,
    wind_capacity_mw: float = ERCOT_WIND_CAPACITY_MW,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load, merge, and convert one calendar year of ERCOT data.

    Reads ``demand_ERCOT_{year}.csv`` and ``nrel_weather_{year}.csv``
    from *data_dir*, computes capacity factors and generation, and
    returns a time-aligned DataFrame.

    Parameters
    ----------
    year : int
        Calendar year (2014–2023).
    solar_capacity_mw : float
        Installed solar nameplate capacity (MW).
    wind_capacity_mw : float
        Installed wind nameplate capacity (MW).
    data_dir : Path, optional
        Directory containing the CSV files.  Defaults to
        ``<package>/data/``.

    Returns
    -------
    pd.DataFrame
        Columns: ``generation_mw``, ``demand_mw``.
        Index: ``timestamp`` (timezone-naive, local Central Time
        convention as stored in the CSVs).

    Raises
    ------
    FileNotFoundError
        If either CSV for the requested year is absent.
    """
    data_dir = Path(data_dir) if data_dir else _DATA_DIR

    demand_path = data_dir / f"demand_ERCOT_{year}.csv"
    weather_path = data_dir / f"nrel_weather_{year}.csv"

    for p in (demand_path, weather_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.  Run ingest_ercot.py first."
            )

    demand_df = pd.read_csv(demand_path, parse_dates=["timestamp"])
    weather_df = pd.read_csv(weather_path, parse_dates=["timestamp"])

    weather_df = compute_ercot_capacity_factors(weather_df)

    weather_df["generation_mw"] = (
        solar_capacity_mw * weather_df["cf_solar"]
        + wind_capacity_mw * weather_df["cf_wind"]
    )

    merged = pd.merge(
        demand_df[["timestamp", "demand_mw"]],
        weather_df[["timestamp", "generation_mw"]],
        on="timestamp",
        how="inner",
    )

    merged = merged.dropna(subset=["demand_mw", "generation_mw"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.set_index("timestamp")
    return merged[["generation_mw", "demand_mw"]]


# ===================================================================== #
#  Multi-year master builder                                             #
# ===================================================================== #

def build_aligned_frame(
    years: Optional[list[int]] = None,
    solar_capacity_mw: float = ERCOT_SOLAR_CAPACITY_MW,
    wind_capacity_mw: float = ERCOT_WIND_CAPACITY_MW,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the master time-aligned DataFrame for the full study period.

    Iterates over *years*, loads each year via :func:`load_ercot_year`,
    concatenates the results, and returns a continuous hourly time series.

    Parameters
    ----------
    years : list[int], optional
        Calendar years to include.  Defaults to 2014–2023.
    solar_capacity_mw : float
        Installed solar nameplate capacity (MW).
    wind_capacity_mw : float
        Installed wind nameplate capacity (MW).
    data_dir : Path, optional
        Override the default CSV directory.

    Returns
    -------
    pd.DataFrame
        Columns ``generation_mw`` and ``demand_mw``, indexed by
        ``timestamp``, covering all requested years with no NaNs.

    Raises
    ------
    ValueError
        If the concatenated DataFrame is empty.
    """
    if years is None:
        years = _YEARS

    frames: list[pd.DataFrame] = []
    for year in years:
        try:
            df = load_ercot_year(
                year,
                solar_capacity_mw=solar_capacity_mw,
                wind_capacity_mw=wind_capacity_mw,
                data_dir=data_dir,
            )
            frames.append(df)
            print(f"  {year}: {len(df):,} hours loaded "
                  f"(gen mean={df.generation_mw.mean():.0f} MW, "
                  f"dem mean={df.demand_mw.mean():.0f} MW)")
        except FileNotFoundError as exc:
            print(f"  {year}: SKIPPED — {exc}")

    if not frames:
        raise ValueError(
            "No data loaded.  Run ingest_ercot.py to download the CSV files."
        )

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index().dropna()

    print(f"\n  Total: {len(combined):,} hours "
          f"({combined.index.min()} to {combined.index.max()})")

    return combined


# ===================================================================== #
#  Synthetic-data generator (unit testing / CI fallback)                #
# ===================================================================== #

def generate_synthetic_data(
    n_years: int = 3,
    solar_capacity_mw: float = ERCOT_SOLAR_CAPACITY_MW,
    wind_capacity_mw: float = ERCOT_WIND_CAPACITY_MW,
    mean_demand_mw: float = 45_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a physically plausible synthetic ERCOT-scale dataset.

    Defaults match ERCOT's 2023 installed capacity and demand profile
    so that synthetic results are numerically comparable to real ones.

    Solar follows a half-sinusoidal diurnal profile modulated by season.
    Wind follows an AR(1) process with moderate autocorrelation.  Demand
    follows a double-sinusoid (diurnal + seasonal) with added noise.

    Parameters
    ----------
    n_years : int               Number of years of hourly data.
    solar_capacity_mw : float   Installed solar nameplate (MW).
    wind_capacity_mw : float    Installed wind nameplate (MW).
    mean_demand_mw : float      Mean demand level (MW).
    seed : int                  Random-number seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns ``generation_mw`` and ``demand_mw``, hourly UTC index.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_years * 8760
    timestamps = pd.date_range(
        "2014-01-01", periods=n_hours, freq="h", tz="UTC"
    )

    hour_of_day = timestamps.hour + timestamps.minute / 60.0
    day_of_year = timestamps.dayofyear

    solar_elevation = np.sin(np.pi * (hour_of_day - 6) / 12)
    solar_elevation = np.clip(solar_elevation, 0, None)
    seasonal_mod = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    cloud_noise = rng.beta(5, 2, size=n_hours)
    solar_cf = np.clip(solar_elevation * seasonal_mod * cloud_noise, 0, 1)

    wind_cf = np.empty(n_hours)
    wind_cf[0] = 0.3
    phi, sigma_wind = 0.95, 0.05
    for t in range(1, n_hours):
        wind_cf[t] = (
            phi * wind_cf[t - 1]
            + (1 - phi) * 0.30
            + rng.normal(0, sigma_wind)
        )
    wind_cf = np.clip(wind_cf, 0, 1)

    diurnal = 0.15 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
    seasonal = 0.10 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
    noise = rng.normal(0, 0.03, size=n_hours)
    demand = np.clip(mean_demand_mw * (1 + diurnal + seasonal + noise), 0, None)

    generation = solar_capacity_mw * solar_cf + wind_capacity_mw * wind_cf

    frame = pd.DataFrame(
        {"generation_mw": generation, "demand_mw": demand},
        index=timestamps,
    )
    frame.index.name = "timestamp"
    return frame
