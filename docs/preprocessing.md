# Building the 2,508-path table

The analysis starts from `data/master_cutting_paths_CLEAN.csv` (SHA-256 in `CHECKSUMS.txt`). Raw data: the public MSM milling dataset
(Sci. Data 2026, https://doi.org/10.1038/s41597-026-07255-7; files https://doi.org/10.34740/kaggle/ds/8392825). The raw files are **not** redistributed here.

## What the table contains
One row per programmed cutting path (`global_path_id`): experiment, machine, tool, workpiece, `N`, `ap`, `ae`, `F`, `Z`, `D`, milling `direction`,
`watt_mean` (mean total electrical power over the path, W) and other sensor summaries.

## Filters and derived features applied in code (`src/common.py::load`)
1. keep rows with `watt_mean` not missing and `machine` in {imi_vm20i, imi_vmx30ui} -> 2,508 paths (1,583 VM20i, 925 VMX30Ui; 37 + 19 experiments);
   the Haas VF-10 group has no comparable power and 5 paths lack usable power;
2. `dir = (direction == "UP")`, `MRR = ap*ae*F/1000`, `fz = F*60/(N*Z)`, `vc = pi*D*N/60000`, `ae_D = ae/D`, `ap_D = ap/D`,
   `hm = 2*fz*sqrt(a*(1-a))` with `a = clip(ae_D, 1e-4, 1)`; `trt` = experiment/settings key.

## Not provided
The step from raw MSM files (MTConnect/sensor streams, labels, NC programs) to the master table, i.e. segmenting streams into cutting paths and averaging power
per path, is not part of this repository. Anyone rebuilding the table should reproduce Table 1 of the paper (path/experiment counts, mean power
3,866 +/- 740 W and 2,821 +/- 796 W) and compare against `CHECKSUMS.txt`. Calibration and test experiment identifiers for the coverage and budget-curve experiments are in
`results/calibration_indices.csv` and `results/test_splits.csv`.
