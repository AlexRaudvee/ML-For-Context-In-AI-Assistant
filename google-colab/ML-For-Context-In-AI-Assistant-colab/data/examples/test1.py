from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


def read_csv_pandas(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file using pandas and return a DataFrame.

    Example
    -------
    >>> df = read_csv_pandas("data.csv")
    >>> print(df.head())
    """
    return pd.read_csv(path)


def write_csv_pandas(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """
    Write a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : str | Path
        Destination CSV path.
    index : bool
        Whether to write the row index.

    Example
    -------
    >>> write_csv_pandas(df, "out.csv", index=False)
    """
    df.to_csv(path, index=index)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load JSON from a file path.

    Example
    -------
    >>> data = load_json("config.json")
    >>> print(data.get("name"))
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """
    Save a dictionary as JSON to disk.

    Example
    -------
    >>> save_json({"ok": True}, "out.json", indent=2)
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
