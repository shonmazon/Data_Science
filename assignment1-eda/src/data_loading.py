"""Loading utilities and schema definitions for the Steam 2024 dataset.

This module centralises two things that would otherwise be repeated throughout
the notebook: the location of the raw file, and the semantic role of every
column. Keeping them here means the analysis never hard-codes a file path or a
column list, so a change to the schema is made in exactly one place.
"""

import re
from pathlib import Path

import pandas as pd

# The repository root is two levels above this file: src -> assignment1-eda -> repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "Data Tables" / "Steam_2024_bestRevenue_1500.csv"

# Minimum dataset dimensions required by the assignment specification.
MIN_REQUIRED_ROWS = 1000
MIN_REQUIRED_COLUMNS = 10

# releaseDate is stored as DD-MM-YYYY text. The format is stated explicitly
# because letting pandas infer it would silently misread every date whose day
# and month are both 12 or below, such as "01-02-2024".
RELEASE_DATE_FORMAT = "%d-%m-%Y"

# The publisher and developer fields hold comma-delimited lists of companies,
# but several legal names contain a comma of their own ("CAPCOM Co., Ltd.").
# In this file the two cases are separable: a list separator is never followed
# by a space, while a comma inside a legal name always is.
ENTITY_SEPARATOR_PATTERN = re.compile(r",(?!\s)")

# Semantic roles of the columns. These describe what each column *means*, which
# is deliberately kept separate from how it happens to be stored on disk:
# releaseDate, for instance, is a date that pandas reads as plain text.
NUMERIC_COLUMNS = ["copiesSold", "price", "revenue", "avgPlaytime", "reviewScore"]
TEMPORAL_COLUMNS = ["releaseDate"]
CATEGORICAL_COLUMNS = ["publisherClass", "publishers", "developers"]
IDENTIFIER_COLUMNS = ["name", "steamId"]

COLUMN_ROLES = {
    column: role
    for role, columns in [
        ("numeric", NUMERIC_COLUMNS),
        ("temporal", TEMPORAL_COLUMNS),
        ("categorical", CATEGORICAL_COLUMNS),
        ("identifier", IDENTIFIER_COLUMNS),
    ]
    for column in columns
}


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the dataset exactly as it is stored on disk.

    No dtype coercion, index assignment or cleaning is applied here. The
    meta-analysis in section 3 needs to observe the types pandas infers by
    default, because those inferences are themselves a finding.

    Args:
        path: Location of the raw CSV file.

    Returns:
        The dataset as read by pandas, with default inferred dtypes.
    """
    return pd.read_csv(path)


def verify_selection_requirements(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Check the dataset against the assignment's selection criteria.

    Args:
        dataframe: The raw dataset.

    Returns:
        One row per requirement, reporting the required and observed values
        alongside a pass or fail marker.
    """
    row_count, column_count = dataframe.shape
    present = set(dataframe.columns)

    checks = [
        ("Tabular format", "Yes", "Yes (single flat CSV table)", True),
        (
            "Minimum rows",
            f">= {MIN_REQUIRED_ROWS}",
            f"{row_count}",
            row_count >= MIN_REQUIRED_ROWS,
        ),
        (
            "Minimum columns",
            f">= {MIN_REQUIRED_COLUMNS}",
            f"{column_count}",
            column_count >= MIN_REQUIRED_COLUMNS,
        ),
    ]

    for role_label, role_columns in [
        ("Numeric variables", NUMERIC_COLUMNS),
        ("Temporal variables", TEMPORAL_COLUMNS),
        ("Categorical variables", CATEGORICAL_COLUMNS),
    ]:
        found = [column for column in role_columns if column in present]
        checks.append(
            (role_label, ">= 1", f"{len(found)} ({', '.join(found)})", bool(found))
        )

    return pd.DataFrame(
        [
            {
                "Requirement": name,
                "Required": required,
                "Observed": observed,
                "Status": "PASS" if passed else "FAIL",
            }
            for name, required, observed, passed in checks
        ]
    )


def build_column_inventory(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Summarise every column: assigned role, storage dtype and basic counts.

    Args:
        dataframe: The raw dataset.

    Returns:
        One row per column, ordered as the columns appear in the file.
    """
    return pd.DataFrame(
        [
            {
                "Column": column,
                "Assigned role": COLUMN_ROLES.get(column, "unassigned"),
                "Stored dtype": str(dataframe[column].dtype),
                "Non-null": int(dataframe[column].notna().sum()),
                "Missing": int(dataframe[column].isna().sum()),
                "Unique values": int(dataframe[column].nunique()),
                "Example value": dataframe[column].dropna().iloc[0],
            }
            for column in dataframe.columns
        ]
    )


def parse_release_date(dataframe: pd.DataFrame) -> pd.Series:
    """Convert the releaseDate text column into real timestamps.

    Args:
        dataframe: The raw dataset.

    Returns:
        The release dates as datetime values. Any value that does not match
        the expected format becomes NaT rather than being silently reinterpreted.
    """
    return pd.to_datetime(
        dataframe["releaseDate"], format=RELEASE_DATE_FORMAT, errors="coerce"
    )


def split_entity_list(value: object) -> list[str]:
    """Split a delimited company list into individual company names.

    Commas that belong to a legal name are preserved, because only commas that
    are not followed by whitespace are treated as separators.

    Args:
        value: A single cell from the publishers or developers column.

    Returns:
        The individual company names, or an empty list for a missing value.
    """
    if pd.isna(value):
        return []
    return [part.strip() for part in ENTITY_SEPARATOR_PATTERN.split(str(value)) if part.strip()]


def summarise_entity_columns(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Quantify how far the company columns depart from being atomic categories.

    Args:
        dataframe: The raw dataset.
        columns: Names of the columns holding delimited company lists.

    Returns:
        One row per column comparing the number of distinct stored strings
        against the number of distinct companies those strings actually contain.
    """
    rows = []
    for column in columns:
        entity_lists = dataframe[column].apply(split_entity_list)
        list_lengths = entity_lists.apply(len)
        distinct_entities = {name for names in entity_lists for name in names}
        rows.append(
            {
                "Column": column,
                "Distinct stored strings": int(dataframe[column].nunique()),
                "Distinct companies": len(distinct_entities),
                "Rows holding >1 company": int((list_lengths > 1).sum()),
                "Most companies in one row": int(list_lengths.max()),
            }
        )
    return pd.DataFrame(rows)
