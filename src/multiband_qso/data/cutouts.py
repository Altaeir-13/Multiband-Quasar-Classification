from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from multiband_qso.config import ensure_parent, load_config

DEFAULT_IMGCUTOUT_ENDPOINT = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"


def build_cutout_url(
    *,
    ra: float,
    dec: float,
    width: int = 128,
    height: int = 128,
    scale: float = 0.396,
    endpoint: str = DEFAULT_IMGCUTOUT_ENDPOINT,
    opt: str = "",
) -> str:
    """Build a SkyServer ImgCutout getjpeg URL."""
    params = {
        "TaskName": "Skyserver.Chart.Image",
        "ra": ra,
        "dec": dec,
        "scale": scale,
        "width": width,
        "height": height,
        "opt": opt,
    }
    return f"{endpoint}?{urlencode(params)}"


def is_valid_image(path: str | Path) -> bool:
    image_path = Path(path)
    if not image_path.exists() or image_path.stat().st_size == 0:
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def download_cutout(
    *,
    ra: float,
    dec: float,
    output_path: str | Path,
    width: int = 128,
    height: int = 128,
    scale: float = 0.396,
    endpoint: str = DEFAULT_IMGCUTOUT_ENDPOINT,
    opt: str = "",
    retries: int = 3,
    timeout_seconds: int = 30,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> str:
    """Download and validate one SDSS cutout.

    Returns one of: ``skipped_existing`` or ``downloaded``.
    """
    destination = ensure_parent(output_path)
    if destination.exists() and not overwrite and is_valid_image(destination):
        return "skipped_existing"

    client = session or requests.Session()
    url = build_cutout_url(
        ra=ra,
        dec=dec,
        width=width,
        height=height,
        scale=scale,
        endpoint=endpoint,
        opt=opt,
    )

    temp_destination = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 multiband-qso-classification/0.1"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            temp_destination.write_bytes(response.content)
            if is_valid_image(temp_destination):
                temp_destination.replace(destination)
                return "downloaded"
            last_error = RuntimeError("Downloaded file is not a valid image")
        except Exception as exc:  # noqa: BLE001 - keep downloader resilient and report later.
            last_error = exc
        finally:
            if temp_destination.exists() and not is_valid_image(temp_destination):
                temp_destination.unlink()
        if attempt < retries:
            time.sleep(min(2**attempt, 10))

    if temp_destination.exists():
        temp_destination.unlink()
    raise RuntimeError(f"Failed to download cutout {url}: {last_error}")


def download_cutouts_from_metadata(
    *,
    metadata_csv: str | Path,
    output_metadata_csv: str | Path | None = None,
    image_root: str | Path | None = None,
    width: int = 128,
    height: int = 128,
    scale: float = 0.396,
    endpoint: str = DEFAULT_IMGCUTOUT_ENDPOINT,
    opt: str = "",
    retries: int = 3,
    timeout_seconds: int = 30,
    request_sleep_seconds: float = 0.15,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download all cutouts referenced by metadata and mark failures."""
    metadata = pd.read_csv(metadata_csv)
    required = {"object_id", "ra", "dec", "class", "image_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns required for cutouts: {sorted(missing)}")

    if image_root is not None:
        root = Path(image_root)
        metadata["image_path"] = [
            str(root / class_name / f"{object_id}.jpg")
            for class_name, object_id in zip(metadata["class"], metadata["object_id"], strict=True)
        ]

    statuses: list[str] = []
    errors: list[str] = []
    with requests.Session() as session:
        for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="cutouts"):
            try:
                status = download_cutout(
                    ra=float(row.ra),
                    dec=float(row.dec),
                    output_path=row.image_path,
                    width=width,
                    height=height,
                    scale=scale,
                    endpoint=endpoint,
                    opt=opt,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    overwrite=overwrite,
                    session=session,
                )
                statuses.append(status)
                errors.append("")
            except RuntimeError as exc:
                statuses.append("failed")
                errors.append(str(exc))
            if request_sleep_seconds > 0:
                time.sleep(request_sleep_seconds)

    metadata["image_download_status"] = statuses
    metadata["image_download_error"] = errors
    output_path = ensure_parent(output_metadata_csv or metadata_csv)
    metadata.to_csv(output_path, index=False)
    print(f"Cutout download summary: {dict(Counter(statuses))}")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download SDSS cutouts from metadata.csv.")
    parser.add_argument("--config", default="configs/image_benchmark.yaml")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    paths_config = config["paths"]
    cutout_config = config["cutouts"]
    sdss_config = config["sdss"]

    download_cutouts_from_metadata(
        metadata_csv=args.metadata_csv or paths_config["metadata_csv"],
        image_root=paths_config["image_root"],
        width=cutout_config.get("width", 128),
        height=cutout_config.get("height", 128),
        scale=cutout_config.get("scale", 0.396),
        endpoint=sdss_config.get("imgcutout_endpoint", DEFAULT_IMGCUTOUT_ENDPOINT),
        opt=cutout_config.get("opt", ""),
        retries=cutout_config.get("retries", 3),
        timeout_seconds=cutout_config.get("timeout_seconds", 30),
        request_sleep_seconds=cutout_config.get("request_sleep_seconds", 0.15),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()