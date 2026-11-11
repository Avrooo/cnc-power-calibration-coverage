# Random seeds (numpy `default_rng(seed)`, per repeat index `rep`)
| Experiment | Script | Seed | Repeats |
|---|---|---|---|
| Budget curves, protocols exp / exp_purged / exp_healthy / expcal | `bench2.py` | `7919*rep+13` | 80 / 80 / 60 / 60 |
| Naive split | `bench2.py` | `7919*rep+13` | 40 |
| Coverage experiment (Fig. 4) | `coverage.py` | `104729*rep+5` | 80 |
| Commissioning designs (Fig. 5) | `e5.py` | `15485863*(rep+1)+3` | 100 |
| Selection rule (Table 4) | `ideaA.py` | `32452843*(rep+1)+11` | 80 |
| Matched-coverage test | `matched.py` | `7001*rep+17` | 60 |
| Leave-one-experiment-out | `loo.py` | `900001*(excluded_index+2)+137*rep+3` | 16 per exclusion |
| Coefficient diagnostic | `coef.py` | `4242*rep+9` | 80 |
| Cluster bootstrap | `analyze2.py` (and `matched_analysis.py`, `matched_stratum.py`) | module-level seeds (0, 5, 11), 2,000 resamples | - |
