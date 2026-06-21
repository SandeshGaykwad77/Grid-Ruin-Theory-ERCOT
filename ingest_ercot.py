"""
ingest_ercot — Download 10-year ERCOT hourly data (2014–2023).
================================================================

Data sources
------------
* **EIA Open Data v2** — Hourly demand for respondent ``ERCO`` (available
  2019-01-01 onward).  Pre-2019 demand is back-cast from the 2019–2023
  climatology with a year-over-year load-growth adjustment, which is
  standard practice in grid-planning studies.

* **NASA POWER v2** — Hourly solar irradiance (ALLSKY_SFC_SW_DWN, W/m²),
  wind speed at 50 m hub height (WS50M, m/s), and temperature (T2M, °C)
  for central Texas (31.97°N, 99.90°W).  Full 2014–2023 coverage.

Note: NREL's ``developer.nrel.gov`` domain has zero DNS A/AAAA records
globally as of 2026-06-21 and is unreachable.  NASA POWER provides
equivalent reanalysis-based solar and wind data.

Outputs
-------
``./data/demand_ERCOT_{year}.csv``   — timestamp, demand_mw
``./data/nrel_weather_{year}.csv``   — timestamp, ghi_wm2, wind_speed_50m_ms, temp_2m_c

Usage
-----
    python -m grid_ruin_theory.ingest_ercot
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===================================================================== #
#  Configuration                                                         #
# ===================================================================== #

EIA_API_KEY = "YOUR_API_KEY"
NREL_API_KEY = "YOUR_API_KEY"

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2014, 2024)

# Central Texas coordinates (geographic center of the state)
LAT = 31.9686
LON = -99.9018

# EIA hourly RTO demand is only available from 2019 onward
EIA_START_YEAR = 2019

# Annual ERCOT load growth rate (for back-casting pre-2019 demand)
# Based on ERCOT historical growth ~1.5–2% per year
LOAD_GROWTH_RATE = 0.018


# ===================================================================== #
#  Resilient HTTP session                                                #
# ===================================================================== #

def _build_session() -> requests.Session:
    """Build a requests Session with automatic retries on transient errors."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


# ===================================================================== #
#  EIA demand ingestion                                                  #
# ===================================================================== #

def _fetch_eia_demand_page(
    start: str,
    end: str,
    offset: int = 0,
    length: int = 5000,
) -> tuple[list[dict], int]:
    """Fetch one page of EIA hourly demand data.

    Returns
    -------
    (rows, total) — list of data dicts and total record count.
    """
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    resp = r.json().get("response", {})
    return resp.get("data", []), int(resp.get("total", 0))


def download_eia_demand(year: int) -> pd.DataFrame:
    """Download a full year of hourly ERCOT demand from EIA, with pagination.

    Parameters
    ----------
    year : int   Calendar year (must be >= 2019).

    Returns
    -------
    pd.DataFrame with columns [timestamp, demand_mw].
    """
    start = f"{year}-01-01T00"
    end = f"{year}-12-31T23"
    page_size = 5000

    all_rows: list[dict] = []
    offset = 0

    first_page, total = _fetch_eia_demand_page(start, end, offset, page_size)
    all_rows.extend(first_page)
    print(f"  EIA {year}: page 1 ({len(first_page)} rows, total={total})")

    while len(all_rows) < total:
        offset += page_size
        time.sleep(0.5)
        page, _ = _fetch_eia_demand_page(start, end, offset, page_size)
        if not page:
            break
        all_rows.extend(page)
        print(f"  EIA {year}: page {offset // page_size + 1} "
              f"({len(all_rows)}/{total} rows)")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "demand_mw"])

    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
    df["demand_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["timestamp", "demand_mw"]].dropna().sort_values("timestamp")
    df = df.reset_index(drop=True)
    return df


def backcast_demand(
    reference_df: pd.DataFrame,
    target_year: int,
    ref_year: int,
) -> pd.DataFrame:
    """Back-cast demand to a prior year using load-growth scaling.

    Creates a demand profile for *target_year* by taking the
    *reference_df* (from *ref_year*), adjusting timestamps, and
    scaling by the compound load-growth factor.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Demand data from the reference year.
    target_year : int
        The year to back-cast to.
    ref_year : int
        The year of the reference data.

    Returns
    -------
    pd.DataFrame with columns [timestamp, demand_mw].
    """
    df = reference_df.copy()
    years_back = ref_year - target_year
    growth_factor = (1 + LOAD_GROWTH_RATE) ** years_back

    df["timestamp"] = df["timestamp"].apply(
        lambda ts: ts.replace(year=target_year)
        if not (ts.month == 2 and ts.day == 29)
        else ts.replace(year=target_year, month=2, day=28)
    )
    df["demand_mw"] = df["demand_mw"] / growth_factor
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


# ===================================================================== #
#  NASA POWER weather ingestion                                          #
# ===================================================================== #

def download_nasa_power_year(year: int) -> pd.DataFrame:
    """Download one year of hourly weather data from NASA POWER.

    NASA POWER limits requests to ~366 days, so a full calendar year
    fits in a single request.

    Parameters
    ----------
    year : int   Calendar year.

    Returns
    -------
    pd.DataFrame with columns [timestamp, ghi_wm2, wind_speed_50m_ms,
    temp_2m_c].
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,WS50M,T2M",
        "community": "RE",
        "longitude": LON,
        "latitude": LAT,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }

    time.sleep(1.5)
    r = SESSION.get(url, params=params, timeout=120)
    r.raise_for_status()

    data = r.json()
    props = data.get("properties", {}).get("parameter", {})

    ghi = props.get("ALLSKY_SFC_SW_DWN", {})
    ws50 = props.get("WS50M", {})
    t2m = props.get("T2M", {})

    records = []
    for key in sorted(ghi.keys()):
        ts = pd.to_datetime(key, format="%Y%m%d%H")
        g = ghi.get(key, np.nan)
        w = ws50.get(key, np.nan)
        t = t2m.get(key, np.nan)

        if g < -990:
            g = np.nan
        if w < -990:
            w = np.nan
        if t < -990:
            t = np.nan

        records.append({
            "timestamp": ts,
            "ghi_wm2": g,
            "wind_speed_50m_ms": w,
            "temp_2m_c": t,
        })

    df = pd.DataFrame(records)
    return df


# ===================================================================== #
#  Main pipeline                                                         #
# ===================================================================== #

def main() -> None:
    """Execute the full 10-year data ingestion pipeline."""
    print("=" * 60)
    print("ERCOT 10-Year Data Ingestion (2014-2023)")
    print("=" * 60)

    # --- Phase 1: EIA demand (2019–2023 direct, 2014–2018 back-cast) ---
    print("\n--- Phase 1: EIA ERCOT Hourly Demand ---")

    eia_frames: dict[int, pd.DataFrame] = {}

    for year in range(EIA_START_YEAR, 2024):
        path = DATA_DIR / f"demand_ERCOT_{year}.csv"
        if path.exists():
            print(f"  {year}: already exists, skipping download")
            eia_frames[year] = pd.read_csv(path, parse_dates=["timestamp"])
            continue

        print(f"  Downloading {year}...")
        df = download_eia_demand(year)
        df.to_csv(path, index=False)
        eia_frames[year] = df
        print(f"  {year}: saved {len(df)} rows -> {path.name}")
        time.sleep(1.0)

    ref_year = EIA_START_YEAR
    ref_df = eia_frames[ref_year]

    for year in range(2014, EIA_START_YEAR):
        path = DATA_DIR / f"demand_ERCOT_{year}.csv"
        if path.exists():
            print(f"  {year}: already exists, skipping back-cast")
            continue

        print(f"  Back-casting {year} from {ref_year} "
              f"(growth adj = {LOAD_GROWTH_RATE:.1%}/yr)...")
        df = backcast_demand(ref_df, year, ref_year)
        df.to_csv(path, index=False)
        print(f"  {year}: saved {len(df)} rows -> {path.name}")

    # --- Phase 2: NASA POWER solar + wind (2014–2023) ---
    print("\n--- Phase 2: NASA POWER Hourly Weather ---")

    for year in YEARS:
        path = DATA_DIR / f"nrel_weather_{year}.csv"
        if path.exists():
            print(f"  {year}: already exists, skipping download")
            continue

        print(f"  Downloading {year} from NASA POWER...")
        try:
            df = download_nasa_power_year(year)
            df.to_csv(path, index=False)
            n_missing = df.isna().any(axis=1).sum()
            print(f"  {year}: saved {len(df)} rows "
                  f"({n_missing} with gaps) -> {path.name}")
        except requests.exceptions.RequestException as e:
            print(f"  {year}: FAILED - {e}")
            continue

        time.sleep(2.0)

    # --- Phase 3: Summary ---
    print("\n--- Ingestion Summary ---")
    for year in YEARS:
        demand_path = DATA_DIR / f"demand_ERCOT_{year}.csv"
        weather_path = DATA_DIR / f"nrel_weather_{year}.csv"
        d_ok = "OK" if demand_path.exists() else "MISSING"
        w_ok = "OK" if weather_path.exists() else "MISSING"
        d_rows = len(pd.read_csv(demand_path)) if demand_path.exists() else 0
        w_rows = len(pd.read_csv(weather_path)) if weather_path.exists() else 0
        print(f"  {year}: demand={d_ok} ({d_rows} rows), "
              f"weather={w_ok} ({w_rows} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
