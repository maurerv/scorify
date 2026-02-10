import tempfile
from pathlib import Path

import pandas as pd
import pytest
import starfile

from scorify.parsing import (
    _COLUMN_MAP,
    _REVERSE_MAP,
    _sanitize_name,
    _standardize,
    export_star,
    load_star_files,
)


class TestSanitizeName:
    def test_strips_pickle_extension(self):
        assert _sanitize_name("TS_001.pickle") == "TS_001"

    def test_strips_star_extension(self):
        assert _sanitize_name("TS_001.star") == "TS_001"

    def test_strips_tomostar_extension(self):
        assert _sanitize_name("TS_001.tomostar") == "TS_001"

    def test_strips_chained_pickle_star(self):
        assert _sanitize_name("TS_001_10.73Apx.pickle.star") == "TS_001"

    def test_strips_apx_suffix(self):
        assert _sanitize_name("TS_001_10.73Apx.pickle") == "TS_001"

    def test_strips_apx_without_extension(self):
        assert _sanitize_name("TS_001_10.73Apx") == "TS_001"

    def test_no_op_on_clean_name(self):
        assert _sanitize_name("TS_001") == "TS_001"

    def test_extracts_basename_from_path(self):
        assert _sanitize_name("/path/to/TS_001_10.73Apx.pickle") == "TS_001"

    def test_preserves_name_when_stripping_would_empty(self):
        # Edge case: name is entirely an Apx pattern
        assert _sanitize_name("_10.73Apx.pickle") != ""

    def test_strips_mrc_extension(self):
        assert _sanitize_name("TS_001.mrc") == "TS_001"

    def test_various_pixel_sizes(self):
        assert _sanitize_name("TS_001_5.0Apx.pickle") == "TS_001"
        assert _sanitize_name("TS_001_13.48Apx.star") == "TS_001"


class TestStandardize:
    def test_renames_relion_columns(self, relion_df):
        result = _standardize(relion_df, "test.star")
        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "rot" in result.columns
        assert "tilt" in result.columns
        assert "psi" in result.columns
        assert "score" in result.columns

    def test_maps_pytme_score(self, pytme_df):
        result = _standardize(pytme_df, "test.star")
        assert "score" in result.columns
        assert list(result["score"]) == [0.1, 0.2, 0.3]

    def test_infers_tomogram_from_filename(self, pytme_df):
        result = _standardize(pytme_df, "TS_005_10.73Apx.pickle.star")
        assert "tomogram" in result.columns
        assert (result["tomogram"] == "TS_005").all()

    def test_sanitizes_tomogram_from_column(self, relion_df):
        result = _standardize(relion_df, "ignored.star")
        assert (result["tomogram"] == "TS_001").all()

    def test_fallback_score_detection(self):
        df = pd.DataFrame({"myCustomScore": [0.5], "rlnCoordinateX": [1.0]})
        result = _standardize(df, "test.star")
        assert "score" in result.columns

    def test_no_score_column_survives(self):
        df = pd.DataFrame({"rlnCoordinateX": [1.0], "someOther": [42]})
        result = _standardize(df, "test.star")
        assert "score" not in result.columns


class TestReverseMap:
    def test_maps_internal_to_relion(self):
        assert _REVERSE_MAP["x"] == "rlnCoordinateX"
        assert _REVERSE_MAP["score"] == "pytmeScore"
        assert _REVERSE_MAP["tomogram"] == "rlnMicrographName"

    def test_no_underscore_prefixed_keys(self):
        for v in _REVERSE_MAP.values():
            assert not v.startswith("_")


class TestExportStar:
    def test_returns_bytes(self, sample_df):
        result = export_star(sample_df)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_contains_relion_headers(self, sample_df):
        text = export_star(sample_df).decode()
        assert "rlnCoordinateX" in text
        assert "rlnAngleRot" in text
        assert "pytmeScore" in text

    def test_round_trip_preserves_rows(self, sample_df):
        data = export_star(sample_df)
        with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as f:
            f.write(data)
            f.flush()
            reloaded = starfile.read(f.name)
        assert len(reloaded) == len(sample_df)


class _FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


class TestLoadStarFiles:
    def _write_star(self, df: pd.DataFrame, path: Path):
        starfile.write(df, str(path))

    def test_loads_single_file(self, relion_df, tmp_path):
        p = tmp_path / "test.star"
        self._write_star(relion_df, p)
        upload = _FakeUpload("TS_001_10.73Apx.pickle.star", p.read_bytes())
        result = load_star_files([upload])
        assert len(result) == 2
        assert "score" in result.columns

    def test_loads_multiple_files(self, relion_df, tmp_path):
        uploads = []
        for name in ["TS_001.star", "TS_002.star"]:
            p = tmp_path / name
            self._write_star(relion_df, p)
            uploads.append(_FakeUpload(name, p.read_bytes()))
        result = load_star_files(uploads)
        assert len(result) == 4

    def test_empty_input(self):
        result = load_star_files([])
        assert result.empty

    def test_score_is_numeric(self, relion_df, tmp_path):
        p = tmp_path / "test.star"
        self._write_star(relion_df, p)
        upload = _FakeUpload("test.star", p.read_bytes())
        result = load_star_files([upload])
        assert pd.api.types.is_numeric_dtype(result["score"])
