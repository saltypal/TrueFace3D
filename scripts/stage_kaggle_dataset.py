"""Stage the private Kaggle dataset (dry run by default). Run from the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggle.staging import main  # noqa: E402

if __name__ == "__main__":
    main()
