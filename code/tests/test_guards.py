"""Guards that stop a script instead of writing or scoring a wrong file."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from winsorize_stored_forecasts import check_volatility_scale  # noqa: E402


def _write(folder, name, fc):
    idx = pd.bdate_range("2020-01-01", periods=len(fc)).strftime("%Y-%m-%d")
    pd.DataFrame({"date": idx, "actual": 0.01, "forecast": fc}).to_csv(folder / name, index=False)


def test_reclip_accepts_volatility_files(tmp_path):
    _write(tmp_path, "ttm_AAPL_h1.csv", [0.01, 0.02, 0.0])
    check_volatility_scale([tmp_path])


def test_reclip_refuses_log_volatility_before_writing(tmp_path):
    _write(tmp_path, "ttm_AAPL_h1.csv", [0.01, 0.02, 0.03])
    _write(tmp_path, "toto_AAPL_h1.csv", [-4.6, -4.5, -4.4])
    before = (tmp_path / "ttm_AAPL_h1.csv").read_bytes()
    with pytest.raises(SystemExit):
        check_volatility_scale([tmp_path])
    assert (tmp_path / "ttm_AAPL_h1.csv").read_bytes() == before
