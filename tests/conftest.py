import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_df():
    """DataFrame mimicking parsed star file output with two tomograms."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 1000, n),
            "y": rng.uniform(0, 1000, n),
            "z": rng.uniform(0, 500, n),
            "rot": rng.uniform(-180, 180, n),
            "tilt": rng.uniform(0, 180, n),
            "psi": rng.uniform(-180, 180, n),
            "score": np.concatenate(
                [rng.uniform(0.15, 0.30, n // 2), rng.uniform(0.20, 0.35, n // 2)]
            ),
            "tomogram": ["TS_001"] * (n // 2) + ["TS_002"] * (n // 2),
        }
    )


@pytest.fixture()
def relion_df():
    """DataFrame with RELION-style column names (no underscore prefix)."""
    return pd.DataFrame(
        {
            "rlnCoordinateX": [100.0, 200.0],
            "rlnCoordinateY": [150.0, 250.0],
            "rlnCoordinateZ": [50.0, 60.0],
            "rlnAngleRot": [10.0, 20.0],
            "rlnAngleTilt": [30.0, 40.0],
            "rlnAnglePsi": [50.0, 60.0],
            "rlnMicrographName": ["TS_001.tomostar", "TS_001.tomostar"],
            "pytmeScore": [0.25, 0.30],
        }
    )


@pytest.fixture()
def pytme_df():
    """DataFrame with pytmeScore column and no rlnLCCmax."""
    return pd.DataFrame(
        {
            "rlnCoordinateX": [1.0, 2.0, 3.0],
            "rlnCoordinateY": [4.0, 5.0, 6.0],
            "rlnCoordinateZ": [7.0, 8.0, 9.0],
            "pytmeScore": [0.1, 0.2, 0.3],
        }
    )
