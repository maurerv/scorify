import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent / "app.py"
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    )


if __name__ == "__main__":
    main()
