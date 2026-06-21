## Requirements

The analysis is entirely self-contained within the standard Python scientific stack. To run the pipeline, you will need to install the following dependencies:

```bash
pip install numpy scipy pandas matplotlib requests
```

---

## 🚀 Usage

To execute the full pipeline from raw data to the final mathematical figures:

1. **Prepare Data:** Ensure the `data/` directory contains the required ERCOT and NASA CSVs. If starting from scratch, run the ingestion script with valid API keys:
   ```bash
   python ingest_ercot.py
   ```

2. **Generate Analysis & Figures:** Execute the figures module. This will run the ruin theory mathematics, trigger the block-bootstrap Monte Carlo validation, and output the final plots into the `figures/` directory:
   ```bash
   python figures.py
   ```

---

## 📊 Data Availability

The complete, processed 10-year aligned dataset (87,648 hourly observations) generated and analyzed by this codebase is hosted separately to ensure permanent open-access. 

* **Dataset:** Available on Zenodo at **https://doi.org/10.5281/zenodo.20785257**
* **Code Archive:** A static snapshot of this codebase is archived on Zenodo at **https://doi.org/10.5281/zenodo.20785310**

---

## 📝 Citation

If you use this code, methodology, or dataset in your research, please cite our paper:

> **Gaykwad, S. (2026).** Heavy-Tailed Renewable Droughts Collapse Exponential Storage Scaling Laws. *Nature Energy* (Under Review).

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
