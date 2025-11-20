# vqc_sims/tests/test_unify.py | Phase 1.2.44 Multiple Inputs Test Enhancement
# Builds on 1.2.43. Changes:
#   • Enhanced multi-file test: assert >=150 rows (3×50), robust to future extra rows (>=100 safe)
#   • Cleaner column stripping, explicit fidelity check, precise source uniqueness
#   • Improved assertion messages and printout
#   • All prior 1.2.43 quoting fixes retained
# Targets: 3/3 unify tests green, multi-input aggregation robust, fidelity + source verified

import sys
import os
import pytest
import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Robust root import: ensure tests can discover project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import ONLY the CSV generator — PDF function removed upstream
from run_all import generate_unified_metrics_csv


def unify_outputs(output_dir: str, l_max: int):
    """Test-compatible wrapper that triggers unified CSV generation only (CSV-only since 1.2.42)."""
    generate_unified_metrics_csv(output_dir, l_max)


@pytest.fixture
def tmp_outputs(tmp_path: Path):
    """Create temporary output directories mirroring real runtime structure."""
    for subdir in ["tables", "pdfs", "figures"]:
        (tmp_path / subdir).mkdir(exist_ok=True)
    return tmp_path


def test_unify_csv(tmp_outputs: Path):
    """Test unified metrics CSV aggregation with proper quoting and clean column reading."""
    tables_dir = tmp_outputs / "tables"

    # Create mock input file that the real function would aggregate
    mock_df = pd.DataFrame({
        "fidelity": np.random.rand(100),
        "energy": np.random.randn(100)
    })
    mock_df.to_csv(tables_dir / "mock_fid_L5.csv", index=False)

    # Run the actual unification logic
    generate_unified_metrics_csv(str(tmp_outputs), l_max=5)

    # Verify output existence and content
    unified_files = glob.glob(str(tables_dir / "vqc_metrics_L5.csv"))
    assert len(unified_files) == 1, "Unified CSV not generated"

    # Read with QUOTE_ALL (mirrors how downstream tools may read it)
    df = pd.read_csv(unified_files[0], quoting=3)  # csv.QUOTE_ALL

    # FIXED in 1.2.43: Strip surrounding quotes from column names (caused by QUOTE_ALL on write)
    df.columns = df.columns.str.replace('"', '').str.strip()

    assert df.shape[0] >= 100, "Unified CSV missing expected rows"
    assert "fidelity" in df.columns, "Column 'fidelity' missing or misquoted"
    assert "energy" in df.columns, "Column 'energy' missing or misquoted"
    assert "source" in df.columns, "Column 'source' (added by aggregator) missing"

    print("Unified metrics CSV: generated + QUOTE_ALL + quote-stripped columns verified")


def test_unify_outputs_call(tmp_outputs: Path):
    """End-to-end test: unify_outputs wrapper produces the unified CSV correctly."""
    tables_dir = tmp_outputs / "tables"

    # Minimal mock input required by the real aggregation logic
    pd.DataFrame({"test_col": [42]}).to_csv(tables_dir / "mock_L5.csv", index=False)

    # Execute wrapper (mirrors real runtime usage — CSV-only)
    unify_outputs(str(tmp_outputs), l_max=5)

    # === CSV verification only ===
    csv_path = tmp_outputs / "tables" / "vqc_metrics_L5.csv"
    assert csv_path.exists(), "Unified CSV missing after unify_outputs call"

    df = pd.read_csv(csv_path, quoting=3)
    df.columns = df.columns.str.replace('"', '').str.strip()  # Consistent fix applied

    assert len(df) > 0, "Unified CSV is empty"
    assert "source" in df.columns, "Aggregator 'source' column missing"

    print(f"unify_outputs wrapper: CSV generated + columns clean — Phase 1.2.44 ready")


def test_unify_csv_multiple_inputs(tmp_outputs: Path):
    """Ensure multiple input files are correctly aggregated (real-world scenario)."""
    tables_dir = tmp_outputs / "tables"

    for i in range(3):
        df = pd.DataFrame({
            "fidelity": np.random.rand(50),
            "energy": np.random.randn(50)
        })
        df.to_csv(tables_dir / f"mock_fid_L5_part{i}.csv", index=False)

    generate_unified_metrics_csv(str(tmp_outputs), l_max=5)

    unified_path = tables_dir / "vqc_metrics_L5.csv"
    assert unified_path.exists(), "Unified CSV missing"

    df = pd.read_csv(unified_path, quoting=3)  # csv.QUOTE_ALL
    df.columns = df.columns.str.replace('"', '')  # Strip QUOTE_ALL artifacts

    assert df.shape[0] >= 150, f"Expected >=150 rows from 3×50 files, got {df.shape[0]}"
    assert "fidelity" in df.columns, "Column 'fidelity' missing after aggregation"
    assert len(df['source'].unique()) == 3, f"Expected 3 unique sources, found {df['source'].nunique()}"

    print(f"Unified multiple inputs: {df.shape[0]} rows from 3 files (wildcard glob verified) — PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])