from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from multiband_qso.config import ensure_parent, load_config

DEFAULT_SQL_ENDPOINT = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
SDSS_CLASSES = ("STAR", "GALAXY", "QSO")


PHOTOMETRY_COLUMNS = [
    "psfMag_u",
    "psfMag_g",
    "psfMag_r",
    "psfMag_i",
    "psfMag_z",
    "modelMag_u",
    "modelMag_g",
    "modelMag_r",
    "modelMag_i",
    "modelMag_z",
    "extinction_u",
    "extinction_g",
    "extinction_r",
    "extinction_i",
    "extinction_z",
]


def build_catalog_query(
    class_name: str,
    limit: int,
    *,
    science_primary: bool = True,
    zwarning_zero: bool = True,
    clean_photometry: bool = True,
) -> str:
    """Build the SDSS DR17 SQL query for one spectroscopic class."""
    normalized_class = class_name.upper()
    if normalized_class not in SDSS_CLASSES:
        raise ValueError(f"Unsupported SDSS class: {class_name}")
    if limit <= 0:
        raise ValueError("limit must be positive")

    filters = [f"s.class = '{normalized_class}'", "s.bestObjID > 0"]
    if science_primary:
        filters.append("s.sciencePrimary = 1")
    if zwarning_zero:
        filters.append("s.zWarning = 0")
    if clean_photometry:
        filters.extend(
            [
                "p.clean = 1",
                "p.mode = 1",
                "p.psfMag_u BETWEEN 0 AND 30",
                "p.psfMag_g BETWEEN 0 AND 30",
                "p.psfMag_r BETWEEN 0 AND 30",
                "p.psfMag_i BETWEEN 0 AND 30",
                "p.psfMag_z BETWEEN 0 AND 30",
            ]
        )

    selected_columns = [
        "p.objID AS object_id",
        "s.specObjID AS spec_obj_id",
        "p.ra AS ra",
        "p.dec AS dec",
        "s.class AS class",
        "s.z AS spec_z",
        *[f"p.{column} AS {column}" for column in PHOTOMETRY_COLUMNS],
    ]
    where_clause = "\n      AND ".join(filters)

    return f"""
SELECT TOP {limit}
      {", ".join(selected_columns)}
FROM SpecObj AS s
JOIN PhotoObj AS p ON s.bestObjID = p.objID
WHERE {where_clause}
ORDER BY p.objID
""".strip()


def query_sdss_sql(
    query: str,
    *,
    endpoint: str = DEFAULT_SQL_ENDPOINT,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Execute a SkyServer SQL query and return a dataframe."""
    client = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 multiband-qso-classification/0.1"}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.get(
                endpoint,
                params={"cmd": query, "format": "csv"},
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                return pd.DataFrame()
            lower_text = text.lower()
            if "error" in lower_text and ("sql" in lower_text or "exception" in lower_text):
                raise RuntimeError(f"SDSS SQL query failed: {text[:1000]}")
            if text.startswith("#Table"):
                text = "\n".join(text.splitlines()[1:])
            return pd.read_csv(StringIO(text))
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"SDSS SQL query failed after {retries} attempts: {last_error}")

def fetch_candidates(
    *,
    output_csv: str | Path,
    endpoint: str = DEFAULT_SQL_ENDPOINT,
    classes: Iterable[str] = SDSS_CLASSES,
    rows_per_class: int = 10000,
    science_primary: bool = True,
    zwarning_zero: bool = True,
    clean_photometry: bool = True,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    """Fetch candidate objects for each class and cache them as CSV."""
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for class_name in classes:
            query = build_catalog_query(
                class_name,
                rows_per_class,
                science_primary=science_primary,
                zwarning_zero=zwarning_zero,
                clean_photometry=clean_photometry,
            )
            frame = query_sdss_sql(
                query,
                endpoint=endpoint,
                timeout_seconds=timeout_seconds,
                session=session,
            )
            if frame.empty:
                raise RuntimeError(f"SDSS returned no rows for class {class_name}")
            frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)
    output_path = ensure_parent(output_csv)
    candidates.to_csv(output_path, index=False)
    return candidates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch SDSS DR17 candidate catalog rows.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--rows-per-class", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    sdss_config = config["sdss"]
    paths_config = config["paths"]
    output_csv = args.output_csv or paths_config["candidates_csv"]
    rows_per_class = args.rows_per_class or sdss_config["candidate_rows_per_class"]

    fetch_candidates(
        output_csv=output_csv,
        endpoint=sdss_config.get("sql_endpoint", DEFAULT_SQL_ENDPOINT),
        classes=sdss_config.get("classes", SDSS_CLASSES),
        rows_per_class=rows_per_class,
        science_primary=sdss_config.get("science_primary", True),
        zwarning_zero=sdss_config.get("zwarning_zero", True),
        clean_photometry=sdss_config.get("clean_photometry", True),
    )


if __name__ == "__main__":
    main()
