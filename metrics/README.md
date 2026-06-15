# Objective Metric Results

This directory contains the objective evaluation synchronized from:

```text
/mnt/data_disk/ZQL/TIP_TMO_result
```

The full remote experiment directory is approximately 299 GB because it also
contains generated tone-mapped images. Only the compact, GitHub-suitable metric
artifacts are included here.

The traditional TMO images evaluated by these metrics were generated with the
MATLAB implementations. Deep-learning images came from the corresponding
original model implementations. These files are not evaluation results of the
reference Python port under `hdrtmo/`.

## Layout

```text
metrics/
├── metric_results_last400/
│   ├── paper_report.md
│   ├── paper_summary.csv
│   ├── paper_summary_grouped.csv
│   ├── paper_table.tex
│   ├── tradition/<method>/
│   │   ├── metrics.csv
│   │   ├── summary.json
│   │   └── summary.txt
│   └── dnn/<method>/
│       ├── metrics.csv
│       ├── summary.json
│       └── summary.txt
└── scripts/
    ├── compute_tip_metrics.py
    ├── run_tip_metrics.sh
    ├── run_tip_metrics_parallel.sh
    ├── run_tip_metrics_matlab.sh
    └── run_tip_database_from_env.m
```

`run.log` and `runner_logs/` are retained locally for diagnosis but excluded
from the GitHub package by `.gitignore`.

## Evaluation Protocol

- Dataset: 1,000-image HDR survey dataset
- Split: filename-sorted final 400 references
- Resolution: 800 x 600
- Traditional TMO image generation: MATLAB
- Deep TMO image generation: original model repositories
- Metrics: TMQI, TMQI-II, and NLPD
- Expected pairs per method: 400
- G-SemTMO valid pairs: 395

Higher TMQI/TMQI-II values indicate better predicted quality. Lower NLPD
indicates a smaller normalized Laplacian pyramid distance.

The final manuscript table maps several experiment directory names to paper
names:

| Experiment directory | Paper name |
| --- | --- |
| `Laplacianet` | DRLTM |
| `lei21_CAN` | Le21 |
| `TMO_GAN` | TMO-GAN |
| `UnCL` | UnCLTMO |
| `Unpaired` | Unpaired-TMO |
| `BestExposure(Histogram)` | BestExp (Hist) |
| `BestExposure(Mean)` | BestExp (Mean) |
| `ReinhardDevlin` | Reinhard05 |
| `WardHistAdj` | Ward |

The root README reports the 23 methods used in the final manuscript. The
synchronized results additionally contain PUCN and KimKautzConsistent for
future analysis.
