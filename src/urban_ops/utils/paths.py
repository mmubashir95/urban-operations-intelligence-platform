"""Project-relative paths used by notebooks and reports."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_overview.ipynb"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_report_directories() -> None:
    """Create report output directories required by analysis notebooks."""
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
