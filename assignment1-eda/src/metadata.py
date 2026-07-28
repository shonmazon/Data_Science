"""File-level and structural profiling utilities.

These functions describe the dataset as an artefact — a file produced at a
point in time by some process — rather than as a table of values. The
distinction matters: several of the strongest findings in this analysis come
from the file's own metadata rather than from its contents.
"""

import re
from pathlib import Path

import pandas as pd

# A column name written in camelCase: a lowercase first word followed by
# capitalised words, e.g. "releaseDate".
CAMEL_CASE_PATTERN = re.compile(r"[a-z]+(?:[A-Z][a-z0-9]*)*$")

# Metadata recorded on the authoring machine at download time. Filesystem
# timestamps are not preserved by git, so the observed value is stored rather
# than re-read at runtime: after a clone, stat() would report the checkout date
# instead of the moment the file was obtained.
DOWNLOAD_TIMESTAMP = pd.Timestamp("2024-09-11 07:12:36", tz="UTC")


def _describe_encoding(raw_bytes: bytes) -> str:
    """Report the narrowest encoding that can decode the file."""
    try:
        raw_bytes.decode("ascii")
        return "ASCII"
    except UnicodeDecodeError:
        pass
    try:
        raw_bytes.decode("utf-8")
        return "UTF-8 (contains non-ASCII characters)"
    except UnicodeDecodeError:
        return "Not valid UTF-8"


def _describe_line_terminator(raw_bytes: bytes) -> str:
    """Identify the newline convention, which hints at the producing platform."""
    carriage_returns = raw_bytes.count(b"\r\n")
    line_feeds = raw_bytes.count(b"\n")
    if carriage_returns and carriage_returns == line_feeds:
        return r"CRLF (\r\n) — Windows convention"
    if carriage_returns:
        return "Mixed CRLF and LF"
    return r"LF (\n) — Unix convention"


def describe_file(path: Path) -> pd.DataFrame:
    """Summarise the physical properties of the data file.

    Args:
        path: Location of the raw CSV file.

    Returns:
        A two-column table of property names and their observed values.
    """
    file_stats = path.stat()
    raw_bytes = path.read_bytes()

    # st_birthtime is the true creation time and exists on macOS; on Linux the
    # attribute is absent, so the modification time is the fallback. The value
    # is read defensively here but is not what gets reported: the timestamps
    # below come from DOWNLOAD_TIMESTAMP, for the reason given at its definition.
    creation_time = getattr(file_stats, "st_birthtime", file_stats.st_mtime)
    line_count = raw_bytes.count(b"\n")
    recorded_timestamp = DOWNLOAD_TIMESTAMP.strftime("%Y-%m-%d %H:%M:%S UTC")

    properties = [
        ("File name", path.name),
        ("Format", "CSV — delimited plain text"),
        ("Size on disk", f"{file_stats.st_size:,} bytes ({file_stats.st_size / 1024:.1f} KB)"),
        ("Encoding", _describe_encoding(raw_bytes)),
        ("Byte order mark", "Absent" if not raw_bytes.startswith(b"\xef\xbb\xbf") else "Present"),
        ("Line terminator", _describe_line_terminator(raw_bytes)),
        ("Field delimiter", "Comma, with quoting for fields containing commas"),
        ("Quote characters", f"{raw_bytes.count(b'\"'):,}"),
        ("Header row", "Present (first line holds column names)"),
        ("Total lines", f"{line_count:,} (1 header + {line_count - 1:,} data rows)"),
        ("Created", recorded_timestamp),
        ("Last modified", recorded_timestamp),
    ]

    return pd.DataFrame(properties, columns=["Property", "Value"])


def analyse_naming_convention(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Describe how the column names are constructed.

    Naming style is evidence about provenance: a perfectly consistent
    convention suggests names generated from a schema, whereas mixed styles
    suggest a table assembled by hand from several sources.

    Args:
        dataframe: The raw dataset.

    Returns:
        One row per column giving its naming style and whether it reads as a
        plural, which signals a field that may hold more than one value.
    """
    rows = []
    for column in dataframe.columns:
        final_word = re.findall(r"[A-Z]?[a-z]+", column)[-1]
        rows.append(
            {
                "Column": column,
                "Naming style": "camelCase" if CAMEL_CASE_PATTERN.fullmatch(column) else "other",
                "Reads as plural": "Yes" if final_word.endswith("s") and not final_word.endswith("ss") else "No",
            }
        )
    return pd.DataFrame(rows)


def build_dtype_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Report the storage type and memory footprint of every column.

    Args:
        dataframe: The raw dataset.

    Returns:
        One row per column with its dtype, the Python type actually stored in
        the first value, and its deep memory usage in kilobytes.
    """
    memory_by_column = dataframe.memory_usage(deep=True)
    return pd.DataFrame(
        [
            {
                "Column": column,
                "Stored dtype": str(dataframe[column].dtype),
                "Python type of value": type(dataframe[column].dropna().iloc[0]).__name__,
                "Memory (KB)": round(memory_by_column[column] / 1024, 1),
            }
            for column in dataframe.columns
        ]
    )
