# Introduction to Data Science — Assignment 1: Exploratory Data Analysis

Holon Institute of Technology (HIT), Faculty of Computer Science.

| | |
|---|---|
| **Course** | Introduction to Data Science |
| **Assignment** | 1 — Exploratory Data Analysis (EDA) |
| **Submission** | Individual |
| **Due date** | 02.08.2026 |

**The deliverable is [`assignment1-eda/notebooks/steam_2024_eda.ipynb`](assignment1-eda/notebooks/steam_2024_eda.ipynb).**
It renders directly on GitHub with all outputs and figures included.

## Dataset

`Steam_2024_bestRevenue_1500.csv` — the 1,500 highest-revenue games released on
Steam during 2024, obtained from the Kaggle dataset
[*Top 1500 Games on Steam by Revenue (09-09-2024)*](https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024)
published by the user `alicemtopcu`.

| Property | Value |
|---|---|
| Rows | 1,500 |
| Columns | 11 |
| Size | 176.5 KB |
| Release dates covered | 2024-01-01 to 2024-09-06 |
| Extraction date | 2024-09-09 (stated by the source, corroborated by the file's timestamp) |

The dataset mixes numeric (`copiesSold`, `price`, `revenue`, `avgPlaytime`,
`reviewScore`), temporal (`releaseDate`) and categorical (`publisherClass`,
`publishers`, `developers`) variables.

## Headline findings

- **Revenue is concentrated; authorship is not.** Five games hold half of all
  revenue and *Black Myth: Wukong* alone holds 21.2%, yet it takes 328 different
  publishers to cover half the games and 83% of publishers appear exactly once.
- **The monetary columns are estimates, not measurements.** Steam publishes no
  per-title sales figures, and four pieces of internal evidence show the values
  are model output.
- **Flaws are encoded as valid numbers.** `reviewScore` uses `0` for "no score
  available" in 99 rows, biasing the mean by 5.4 points; `price` uses `0` for
  genuinely free games. The same literal value, opposite meanings.
- **The rows are not ordered as the file name implies.** The file is four
  concatenated sorted blocks, so `head(10)` and the true top 10 by revenue share
  no games at all.
- **Selection erased a real effect.** Time on sale ranges from 5 to 254 days, yet
  median revenue is flat across release months — a collider effect created by
  selecting on revenue itself.

## Repository structure

```
.
├── Data Tables/
│   └── Steam_2024_bestRevenue_1500.csv   Raw dataset, unmodified
├── assignment1-eda/
│   ├── notebooks/
│   │   └── steam_2024_eda.ipynb          Main deliverable, sections 1-9
│   ├── src/                              Reusable helper modules
│   │   ├── data_loading.py               Paths, column roles, date and list parsing
│   │   ├── metadata.py                   File-level and structural profiling
│   │   ├── quality_checks.py             Missingness, duplicates, cardinality, index
│   │   ├── cleaning.py                   Corrections justified by section 4
│   │   ├── univariate.py                 Summary statistics and outlier detectors
│   │   ├── associations.py               Correlations, Cramér's V, binning
│   │   ├── features.py                   Engineered features for section 9
│   │   ├── hypothesis_tests.py           Non-parametric tests with correction
│   │   └── plotting.py                   House chart style and figure builders
│   └── reports/figures/                  Exported figures
├── requirements.txt
└── README.md
```

## Reproducing the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name steam-eda --display-name "Python (steam-eda)"
jupyter lab assignment1-eda/notebooks/steam_2024_eda.ipynb
```

Select the **Python (steam-eda)** kernel, then run *Restart & Run All*. The
notebook executes top to bottom with no manual steps. Developed against Python
3.14.5.

## Notes

Analysis prose, code comments and headings are written in English throughout, as
the assignment requires. The notebook is committed with its outputs so it can be
read without being run.
