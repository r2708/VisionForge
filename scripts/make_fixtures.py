"""Generate the synthetic fixture image folder used by the tests and the demo.

    python scripts/make_fixtures.py OUTPUT_DIR
"""
import sys
from pathlib import Path

from app.tests.fixtures import build_fixture_dataset

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    if target.exists() and any(target.iterdir()):
        sys.exit(f"{target} already exists and is not empty")
    info = build_fixture_dataset(target)
    print(f"Wrote {info['supported_files']} image files ({info['valid']} valid, {info['invalid']} deliberately broken) to {target}")
