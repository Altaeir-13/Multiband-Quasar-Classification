from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pandas as pd
from PIL import Image

from multiband_qso.data.cutouts import (
    build_cutout_url,
    download_cutouts_from_metadata,
    is_valid_image,
)


def test_build_cutout_url_contains_sdss_parameters() -> None:
    url = build_cutout_url(ra=180.0, dec=0.25, width=128, height=128, scale=0.396)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert "skyserver.sdss.org" in parsed.netloc
    assert parsed.path.endswith("/SkyServerWS/ImgCutout/getjpeg")
    assert params["ra"] == ["180.0"]
    assert params["dec"] == ["0.25"]
    assert params["width"] == ["128"]
    assert params["height"] == ["128"]
    assert params["scale"] == ["0.396"]
    assert params["TaskName"] == ["Skyserver.Chart.Image"]


def test_downloader_records_failure_without_deleting_existing_valid_image(tmp_path) -> None:
    valid_path = tmp_path / "STAR" / "1.jpg"
    valid_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(120, 80, 40)).save(valid_path)
    missing_target = tmp_path / "QSO" / "2.jpg"

    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "object_id": 1,
                "ra": 180.0,
                "dec": 0.1,
                "class": "STAR",
                "image_path": str(valid_path),
            },
            {
                "object_id": 2,
                "ra": 180.0,
                "dec": 0.2,
                "class": "QSO",
                "image_path": str(missing_target),
            },
        ]
    ).to_csv(metadata_path, index=False)

    result = download_cutouts_from_metadata(
        metadata_csv=metadata_path,
        endpoint="http://127.0.0.1:1/does-not-exist",
        retries=1,
        timeout_seconds=1,
        request_sleep_seconds=0,
    )

    assert is_valid_image(valid_path)
    assert result.loc[0, "image_download_status"] == "skipped_existing"
    assert result.loc[0, "image_download_error"] == ""
    assert result.loc[1, "image_download_status"] == "failed"
    assert result.loc[1, "image_download_error"]