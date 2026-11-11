# cnc-power-calibration-coverage

Reproducible framework for evaluating calibration coverage, cross-machine power-model transfer, and target-machine retraining using the MSM CNC milling dataset.

Code, fixed seeds, result tables and regeneration checks behind the NAMRC 55 manuscript
"Calibration Coverage and Evaluation Protocol in Cross-Machine CNC Milling Power Prediction: Transfer Versus Retraining".
Contact: ssavro@miners.utep.edu. Code: MIT (`LICENSE`). Data: not included (`DATA_LICENSE.md`). Cite: `CITATION.cff`.

## Research objective
Compare offset-calibrated transfer of a source-machine model with target-only retraining (physics-structured, Gaussian-process and gradient-boosted models)
at matched calibration budgets, under experiment-disjoint evaluation, and study how calibration coverage (materials, tools, operating conditions) changes the preferred strategy.
One machine pair (Hurco VM20i and VMX30Ui, 2,508 cutting paths), retrospective analysis, exploratory.

## Repository layout
| Folder / file | Content |
|---|---|
| `src/` | data loading and features (`common.py`), models and calibration-design code (`bench.py`, `bench2.py`, `coverage.py`, `e5.py`, `matched.py`), cluster bootstrap (`analyze2.py`) |
| `experiments/` | scripts for evaluation protocols (`leak.py`), selection rule (`ideaA*.py`), robustness (`loo*.py`, `coef*.py`, `matched_*.py`), summaries (`cov_*.py`, `e5_analysis.py`, `analyze3/4.py`), index export (`export_indices.py`), exploratory (`diag.py`, `drift.py`) |
| `results/` | numerical results behind the tables and figures (CSV), `numbers.json`, `calibration_indices.csv`, `test_splits.csv` (calibration cuts and held-out experiments for the Fig. 3 and Fig. 4 experiments), `robustness_summary.txt` |
| `figures/` | Figs. 1–4 and supplementary `figS1.png`; `make_assets.py` (Figs. 2–4, S1 and `numbers.json` from `results/`), `make_fig1.py` (Fig. 1) |
| `configs/` | seeds, model parameters/bounds, evaluation settings (documentation of what the scripts use; the scripts do not read these files) |
| `docs/preprocessing.md` | how the 2,508-path table is defined and what is not provided |
| `data/` | place `master_cutting_paths_CLEAN.csv` here (not tracked) |

The raw MSM dataset is **not** copied here: use the original release (Sci. Data 2026, https://doi.org/10.1038/s41597-026-07255-7; files https://doi.org/10.34740/kaggle/ds/8392825) and check its licence.

## Install and reproduce
```
pip install -r requirements.txt
export MSM_MASTER_CSV=data/master_cutting_paths_CLEAN.csv     # SHA-256 in CHECKSUMS.txt
python figures/make_assets.py          # Figs. 2-4, S1 and results/numbers.json from results/*.csv (~1-2 min)
python figures/make_fig1.py            # Fig. 1
python experiments/export_indices.py   # re-creates calibration_indices.csv / test_splits.csv and checks them against the archived tables
bash run_all.sh                        # full re-run of every experiment (~30-40 min on 2 cores)
```
Run all commands from the repository root.

## Script map
| Manuscript item | Script(s) | Result files |
|---|---|---|
| Fig. 2 (evaluation protocols) | `experiments/leak.py` | `leakage_ladder.csv` |
| Direct transfer MAE | `src/bench.py`, `src/bench2.py` | `bench_all.csv` |
| Fig. 3, Table 2 | `src/bench2.py exp/naive`, `src/analyze2.py`, `experiments/analyze3.py` | `b2_exp.csv`, `b2_naive.csv`, `a2_*.csv` |
| Fig. 4, Table 3, coefficient analysis | `src/coverage.py`, `experiments/cov_analysis.py`, `cov_within.py`, `coef*.py` | `cov_draws*.csv`, `cov_exp.csv`, `coef_diag.csv` |
| Section 4.5, Fig. S1 | `src/e5.py`, `experiments/e5_analysis.py` | `e5_draws.csv`, `e5_exp.csv` |
| Section 4.6, Table S1 | `experiments/ideaA.py`, `ideaA_analysis.py` | `ideaA_draws.csv`, `ideaA_exp.csv` |
| Robustness | `src/matched.py`, `experiments/matched_*.py`, `loo*.py` | `matched_sets.pkl`, `loo_sensitivity.csv`, `robustness_summary.txt` |

## Status of reproduction (read before relying on this)
* **Verified (this layout):** `figures/make_assets.py` runs from `results/*.csv` plus the path table and reproduces `numbers.json` identically; `experiments/export_indices.py` re-creates all 3,503 coverage-experiment draws and matches the archived rows.
* **Not verified from scratch in this package:** a complete `run_all.sh` rerun after the folder reorganisation.
* **Not provided:** preprocessing from the raw MSM release to the path table (`docs/preprocessing.md`); calibration index lists for the commissioning-design, selection-rule and robustness runs (fixed by seeds in `SEEDS.md`/`configs/seeds.yaml`, not exported).
* The physics model leaves a material's coefficient at its starting value (K = 1.5 W per mm^3/s) when that material has no calibration cuts (`src/bench2.py`, `fit_reg`); this is an algorithmic fallback, not a physical estimate.
* Bootstrap intervals resample all target-machine experiments (37 VM20i, 19 VMX30Ui); each experiment is held out in about 30% of repeats.

## Releasing
1. Create the GitHub repository (name above) and push. 2. Make it public and, if you want a DOI, connect a release to Zenodo. 3. Put the real URL in `CITATION.cff` and the URL/DOI in the paper's Appendix A.
