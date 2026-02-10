import re
import tempfile
from pathlib import Path

import pandas as pd
import starfile

# Internal column names mapped from RELION / pytme conventions
_COLUMN_MAP = {
    "rlnCoordinateX": "x",
    "rlnCoordinateY": "y",
    "rlnCoordinateZ": "z",
    "rlnAngleRot": "rot",
    "rlnAngleTilt": "tilt",
    "rlnAnglePsi": "psi",
    "rlnMicrographName": "tomogram",
    "rlnLCCmax": "score",
    "rlnMetricValue": "score",
    "pytmeScore": "score",
}

ANGLE_COLUMNS = {"rot", "tilt", "psi"}
POSITION_COLUMNS = {"x", "y", "z"}

_KNOWN_EXTENSIONS = (".pickle", ".star", ".mrc", ".tomostar")
_APX_PATTERN = re.compile(r"_[\d.]+Apx$")


def _sanitize_name(raw: str) -> str:
    """Strip Warp/M pixel-size tags and pytme extensions from a name."""
    name = Path(raw).name
    changed = True
    while changed:
        changed = False
        for ext in _KNOWN_EXTENSIONS:
            if name.endswith(ext):
                name = name[: -len(ext)]
                changed = True
    cleaned = _APX_PATTERN.sub("", name)
    return cleaned if cleaned else name


def _standardize(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Rename columns to internal names, infer tomogram if missing."""
    renamed = df.rename(columns=_COLUMN_MAP)

    if "tomogram" not in renamed.columns:
        renamed["tomogram"] = _sanitize_name(filename)
    else:
        renamed["tomogram"] = renamed["tomogram"].apply(_sanitize_name)

    if "score" not in renamed.columns:
        score_candidates = [
            c
            for c in renamed.columns
            if "score" in c.lower() or "lcc" in c.lower() or "correlation" in c.lower()
        ]
        if score_candidates:
            renamed = renamed.rename(columns={score_candidates[0]: "score"})

    return renamed


_REVERSE_MAP = {v: k for k, v in _COLUMN_MAP.items() if not k.startswith("_")}


def export_star(df: pd.DataFrame) -> bytes:
    """Convert internal DataFrame back to RELION star file bytes."""
    out = df.rename(columns=_REVERSE_MAP)
    with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as tmp:
        starfile.write(out, tmp.name, overwrite=True)
    return Path(tmp.name).read_bytes()


def load_star_files(uploaded_files) -> pd.DataFrame:
    """Read uploaded star files and return a single concatenated DataFrame."""
    frames = []
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(suffix=".star", delete=False) as tmp:
            tmp.write(uf.getvalue())
            tmp.flush()
            data = starfile.read(tmp.name)

        if isinstance(data, dict):
            data = max(data.values(), key=len)

        df = _standardize(data, uf.name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if "score" in combined.columns:
        combined["score"] = pd.to_numeric(combined["score"], errors="coerce")

    return combined
