# NRCD resource paper & analysis

CIKM resource paper and reproducible analysis scripts for the **National Running Club Database (NRCD)**.

The standalone **`nrcd` Python package** lives in [`nrcd_package/`](nrcd_package/) (for athletes/researchers standardizing their own data). Paper analysis uses `scripts/` and does not require running package tests.

## Data

Download CSV exports from Zenodo: **https://zenodo.org/records/17917357**

Place files in `data/` (see `data/README.md`).

## Paper

- LaTeX: `papers/cikm.tex`
- Build PDF: `paper_build/build_cikm.sh` → `paper_build/cikm.pdf`

## Analysis (reproduce paper statistics)

```bash
pip install -r requirements.txt
python scripts/run_all.py --paper-only   # main paper numbers
```

Key scripts:

| Script | Purpose |
|--------|---------|
| `scripts/run_all.py` | Orchestrates validations and writes `results/dataset_stats.json` |
| `scripts/load_data.py` | Merge approved CSV tables |
| `scripts/standardization.py` | XC standardization pipeline (paper formulas) |
| `scripts/validation_*.py` | Formula validation experiments |

Optional API backfill (`scripts/enrich_api.py`) requires the separate `nrcd` package: `pip install nrcd[apis]`.

## Citation

> **NRCD: An Open Database of Collegiate Running with Unified Performance Standardization**  
> Jonathan A. Karr Jr, Ryan M. Fryer, Ben Darden, Nicholas Pell, Kayla Ambrose, Evan Hall, Ramzi K. Bualuan, and Nitesh V. Chawla.  
> arXiv preprint (forthcoming).

Dataset: [Zenodo 17917357](https://zenodo.org/records/17917357)

---

Repository by [Jonathan A. Karr Jr.](https://orcid.org/0009-0000-1600-6122) ([jkarr@nd.edu](mailto:jkarr@nd.edu)), with the help of [Cursor](https://cursor.com).
