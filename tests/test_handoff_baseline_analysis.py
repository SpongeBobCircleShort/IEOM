import subprocess
import sys
from pathlib import Path


def test_handoff_scrape_and_analysis(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    summary_csv = out_dir / "handoff_baseline_summary.csv"
    report_path = out_dir / "handoff_next_steps.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/scrape_handoff_baseline_data.py",
            "--output-dir",
            str(out_dir),
            "--sqlite-path",
            str(out_dir / "handoff_baseline.db"),
            "--human-speed",
            "0.8",
            "--robot-speeds",
            "0.4",
            "0.6",
            "0.75",
        ],
        check=True,
    )

    assert summary_csv.exists()

    run = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_handoff_baseline.py",
            "--summary-csv",
            str(summary_csv),
            "--report-path",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Baseline Handoff: What to Do Next" in text
    assert "Introduce hesitation" in text
    assert "aggressive" in text
    assert "Wrote report" in run.stdout
