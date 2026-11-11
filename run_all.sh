#!/usr/bin/env bash
# Full re-run of every experiment (run from the repository root).
# Requires data/master_cutting_paths_CLEAN.csv (see data/README or the main README) or MSM_MASTER_CSV.
set -e; export PYTHONWARNINGS=ignore; export MSM_MASTER_CSV=${MSM_MASTER_CSV:-data/master_cutting_paths_CLEAN.csv}
mkdir -p results figures
python experiments/leak.py
python src/bench.py all 30
python src/bench2.py exp 80; python src/bench2.py exp_purged 80; python src/bench2.py naive 40; python src/bench2.py exp_healthy 60; python src/bench2.py expcal 60
python src/coverage.py 80; python src/e5.py 100; python experiments/ideaA.py 80
python src/matched.py 60; python experiments/loo.py 16; python experiments/coef.py 80
python experiments/cov_analysis.py; python experiments/cov_within.py; python experiments/e5_analysis.py; python experiments/ideaA_analysis.py
python src/analyze2.py; python experiments/analyze3.py; python experiments/analyze4.py
python figures/make_assets.py; python figures/make_fig1.py; python experiments/export_indices.py
python experiments/matched_analysis.py; python experiments/matched_stratum.py; python experiments/loo_analysis.py; python experiments/coef_analysis.py
