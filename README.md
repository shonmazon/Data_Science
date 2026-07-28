# Introduction to Data Science — Assignment 1: Exploratory Data Analysis

Holon Institute of Technology (HIT), Faculty of Computer Science.

| | |
|---|---|
| **Course** | Introduction to Data Science |
| **Assignment** | 1 — Exploratory Data Analysis (EDA) |
| **Submission** | Individual |
| **Due date** | 02.08.2026 |

## Dataset

`Steam_2024_bestRevenue_1500.csv` — the 1,500 highest-revenue games released on
Steam during 2024, sourced from Kaggle.

| Property | Value |
|---|---|
| Rows | 1,500 |
| Columns | 11 |
| Size | ~176 KB |
| Time span | 2024-01-01 to 2024-09-06 |

The dataset mixes numeric (`copiesSold`, `price`, `revenue`, `avgPlaytime`,
`reviewScore`), temporal (`releaseDate`) and categorical (`publisherClass`,
`publishers`, `developers`) variables.

## Repository structure

```
.
├── Data Tables/
│   └── Steam_2024_bestRevenue_1500.csv   Raw dataset (unmodified)
├── assignment1-eda/
│   ├── notebooks/
│   │   └── steam_2024_eda.ipynb          Main deliverable
│   ├── src/                              Reusable helper modules
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
notebook executes top to bottom with no manual steps.

Developed against Python 3.14.5.
