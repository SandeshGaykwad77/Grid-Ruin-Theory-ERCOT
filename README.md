# Heavy-Tailed Renewable Droughts Collapse Exponential Storage Scaling Laws

This repository contains the computational pipeline and mathematical engine for the manuscript **"Heavy-Tailed Renewable Droughts Collapse Exponential Storage Scaling Laws"** (submitted to *Nature Energy*).

This project transplants classical Cramér-Lundberg actuarial ruin theory onto renewable grid energy storage. By applying this mathematical framework to 10 years of ERCOT demand and NASA MERRA-2 meteorological data, we demonstrate that extreme renewable droughts are heavy-tailed ($\alpha \approx 2.67$). Consequently, by the Embrechts-Veraverbeke theorem, the exponential reliability scaling of battery storage collapses to a power law. 

## Repository Structure

The codebase is modularized to ensure reproducibility:

* **`ingest_ercot.py`**: Handles API pagination, connection resilience, and the downloading of historical hourly demand (EIA Open Data v2) and meteorological data (NASA POWER MERRA-2) for the 2014-2023 study period.
* **`data_pipeline.py`**: The ingestion and conversion layer. Converts raw global horizontal irradiance (GHI) and 50m wind speeds into normalized capacity factors using piecewise-cubic turbine power curves and standard PV efficiency models.
* **`ruin_theory.py`**: The core analytical engine. Computes the Grid Lundberg Coefficient ($R_g$) via empirical moment-generating functions, calculates the Cramér asymptotic constant, extracts drought events, and applies the Hill estimator to determine the tail index $\alpha$.
* **`validation.py`**: A vectorized Monte Carlo simulation engine utilizing a Circular Block Bootstrap (Politis & Romano, 1992) to preserve Markovian weather autocorrelation and empirically validate the analytical Cramér-Lundberg bounds.
* **`figures.py`**: Generates the publication-ready, high-DPI figures (Ruin Curve, Hill Plot, and Overbuild-Storage Frontier) used in the manuscript.

## Data Availability

The complete, processed 10-year aligned dataset (87,648 hourly observations) generated and analyzed by this codebase is hosted separately to ensure permanent access. It is available on Zenodo at: **[Insert Zenodo Data DOI Here]**

## Requirements

The analysis is entirely self-contained within the standard Python scientific stack. To run the pipeline, you will need:

```bash
pip install numpy scipy pandas matplotlib requests
