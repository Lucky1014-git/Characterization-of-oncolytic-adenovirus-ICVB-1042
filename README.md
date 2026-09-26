# Characterization of Oncolytic Adenovirus ICVB-1042

Computational modeling of the engineered oncolytic adenovirus **ICVB-1042** compared to **wild-type Ad5 (Wt-Ad5)**, using ordinary differential equation (ODE) models fit to published mouse breast cancer data. This project was conducted at TCU under Dr. Hana Dobrovolny.

## Overview

Oncolytic viruses like ICVB-1042 are engineered to selectively infect and kill cancer cells. This project builds a mathematical model of viral infection dynamics within a tumor and fits it to experimental data (viral titer and tumor volume) in order to compare how ICVB-1042 and Wt-Ad5 behave inside a host, and to identify which biological mechanisms explain ICVB-1042's improved systemic efficacy.

Data source: Kato et al. 2024, human breast cancer (MDA-MB-231) mouse xenografts, virus injected intravenously at multiple time points.

## Model

An initial two-compartment model (uninfected/infected tumor cells, viral production, viral clearance) predicted tumor shrinkage far faster than observed. The model was rebuilt as a four-compartment ODE system to correct this:

- Uninfected tumor cells
- Infected tumor cells
- Interferon-resistant tumor cells (with waning resistance over time)
- Free virus

Components:
- Tumor growth rate, fit from vehicle (untreated) control data
- Infection, viral production, and viral clearance rates, fit by minimizing the sum of squared residuals against viral titer and tumor volume data
- Interferon-induced resistance, allowing part of the tumor population to temporarily escape infection

The model is fit separately to ICVB-1042 and Wt-Ad5 data, then compared.

## Statistical Analysis

Parameter uncertainty and comparisons between viruses are assessed via bootstrapping:
- The fit is repeated 1,000 times over resampled data
- Parameter distributions across bootstrap runs are compared between viruses using Mann-Whitney U tests
- Results show statistically significant differences in **viral production rate** (p = 0.0003) and **viral clearance rate** (p = 0.0004) between the two viruses, with ICVB-1042 exhibiting the higher production rate — a candidate explanation for its stronger systemic performance.

## Repository Structure

| File | Description |
|---|---|
| `adenovirus_ICVB1042_model.py`, `adenovirus_ICVB1042_model_co...py` | Core ODE model definition and fitting for ICVB-1042 |
| `adenovirus_ICB1042_final.py` | Finalized ICVB-1042 model/fit script |
| `adenovirus_ICB1042_bootstrappi...py`, `adenoICB1042-bootstrap_results....` | Bootstrap fitting procedure and saved results for ICVB-1042 |
| `adenovirus_WtAd5_final.py` | Finalized wild-type Ad5 model/fit script |
| `adenovirus_WtAd5_bootstrappin...py` | Bootstrap fitting procedure for Wt-Ad5 |
| `ICVB-1042_histograms.py` | Generates histograms of bootstrapped parameter distributions for comparison |
| `42003_2024_6839_MOESM6_E...` | Supplementary data file from the source publication (Kato et al. 2024) |
| `assignment_*.py` | Coursework scripts, not part of the core research pipeline |

*(Update file names/descriptions above to exact matches once finalized — some names were truncated in the file browser.)*

## Requirements

- Python 3.x
- `numpy`
- `scipy` (`scipy.optimize` for curve fitting)
- `matplotlib` (for histograms/plots)

Install with:
```bash
pip install numpy scipy matplotlib
```

## Usage

1. Place the source data file (e.g. the supplementary Kato et al. data) in the repository root.
2. Run the model/fit script for the virus of interest, e.g.:
   ```bash
   python adenovirus_ICB1042_final.py
   ```
3. Run the corresponding bootstrap script to generate parameter distributions:
   ```bash
   python adenovirus_ICB1042_bootstrapping.py
   ```
4. Generate comparison histograms:
   ```bash
   python ICVB-1042_histograms.py
   ```

## Presentations & Publications

- Poster: *"Characterization of Oncolytic Adenovirus ICVB-1042"* — Vengadeswaran and Dobrovolny, TCU Department of Physics and Astronomy
- Presented at the 2026 TCU Spring Student Research Symposium
- Presented at SIAM TX-LA
- To be presented at the TCU 2026 Fall Student Research Symposium
- Manuscript in preparation (co-authored)

## Acknowledgments

Research conducted under the supervision of Dr. Hana Dobrovolny, TCU Department of Physics and Astronomy.
