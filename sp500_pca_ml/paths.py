from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
print(f"DATA_DIR: {DATA_DIR}")

# create a data directory if it does not exist
