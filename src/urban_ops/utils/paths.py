"""Project-relative paths used by notebooks and reports."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_overview.ipynb"
REPORTS_DIR = PROJECT_ROOT / "reports"


def notebook_report_paths(notebook_slug: str) -> tuple[Path, Path, Path]:
    """Return report, tables, and figures directories for a notebook slug."""
    report_dir = REPORTS_DIR / notebook_slug
    return report_dir, report_dir / "tables", report_dir / "figures"


def ensure_report_directories(notebook_slug: str) -> None:
    """Create per-notebook report output directories."""
    _, tables_dir, figures_dir = notebook_report_paths(notebook_slug)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
