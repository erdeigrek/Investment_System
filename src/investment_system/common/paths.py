from pathlib import Path


def project_root() -> Path:
    """
    Finds project root by looking for README.md or data directory.
    """
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists() or (parent / "data").exists():
            return parent

    raise RuntimeError("Project root not found")


# Root directory
ROOT_DIR = project_root()

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
