from __future__ import annotations

import os

import pytest

from multiband_qso.data.sdss import build_catalog_query, query_sdss_sql


@pytest.mark.live_sdss
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SDSS_TESTS") != "1",
    reason="Set RUN_LIVE_SDSS_TESTS=1 to run live SDSS API checks.",
)
def test_live_sdss_sql_returns_one_star_row() -> None:
    query = build_catalog_query("STAR", 1)
    frame = query_sdss_sql(query, timeout_seconds=30)
    assert len(frame) == 1
    assert str(frame.loc[0, "class"]).upper() == "STAR"
