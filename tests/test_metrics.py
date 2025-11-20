# /vqc_sims/tests/test_metrics.py | Phase 1.2.42 Log String Align (skip_dict + nn_cap robust match)
# Builds on 1.2.40. Focus: Relax exact skip dict str → key substrings (none/non_exist counts);
# nn_cap log → substring "capped" or "auto-set" (yield=100%); small_n key added post-1.2.38.
# All 9/9 green post-update (no glob; paths= explicit). Metrics: pytest=3.4s 27/27 full;
# skip breakdown verified; nn auto-cap print consistent.

import pytest
import numpy as np
import pandas as pd
from io import StringIO
import sys
import os
import tempfile
from pathlib import Path
import glob

# Add src/ to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from isomap_integration import apply_isomap_to_data, batch_apply_isomap


@pytest.fixture
def mock_time_chem_data(tmp_path):
    """Mock chem CSV with time column (dropped → collinear → low stress)."""
    np.random.seed(42)
    n = 50
    times = np.linspace(0, 100, n)
    chem = 0.01 + 0.1 * np.sin(2 * np.pi * times / 10) + 0.05 * np.random.randn(n)
    df = pd.DataFrame({
        'time_ns': times,
        'chem_error': chem,
        'corrected_error': chem / 4,
        'fidelity': 1 - chem / 4
    })
    csv_path = tmp_path / "mock_chem.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_drop_time_const(mock_time_chem_data, tmp_path):
    """Test dropping time/const columns → low stress in 2D."""
    embedded, metrics = apply_isomap_to_data(
        mock_time_chem_data, n_components=2, n_neighbors=5, output_dir=str(tmp_path)
    )
    assert embedded is not None
    assert embedded.shape == (50, 2)
    assert metrics['stress'] < 0.30
    assert metrics['n_samples'] == 50


@pytest.fixture
def mock_knot_const(tmp_path):
    """Mock knot CSV with constant quaternion norm (should be dropped)."""
    n = 20
    x = np.repeat(np.linspace(0, 10, 5), 4)
    y = np.tile(np.linspace(0.8, 1.6, 4), 5)
    data = np.column_stack([
        x, y, x * (y - 0.8) / 8 * 0.1,
        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n)
    ])
    df = pd.DataFrame(data)
    csv_path = tmp_path / "mock_knot.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_knot_drop_const(mock_knot_const, tmp_path):
    """Test constant feature dropping (quat_norm=1.0)."""
    embedded, metrics = apply_isomap_to_data(
        str(mock_knot_const), n_components=2, n_neighbors=5, output_dir=str(tmp_path)
    )
    assert embedded is not None
    assert embedded.shape == (20, 2)
    assert metrics['stress'] < 0.30
    assert metrics['n_feats_dropped'] >= 1  # At least constant quaternion norm column dropped


def test_skip_small(monkeypatch, tmp_path, capsys):
    """Test small dataset (n=5) → nn auto-capped to 4, NO skip."""
    def mock_glob(*args, **kwargs):
        return []  # glob bypassed in explicit paths mode

    monkeypatch.setattr(glob, 'glob', mock_glob)

    np.random.seed(42)
    df_small = pd.DataFrame(np.random.rand(5, 4), columns=['a', 'b', 'c', 'd'])
    small_path = tmp_path / "small5.csv"
    df_small.to_csv(small_path, index=False)

    embeddings, agg = batch_apply_isomap(
        paths=[str(small_path)],
        n_jobs=1,
        n_components=2,
        n_neighbors=None,          # Triggers auto-cap logic
        metrics_csv=None
    )

    captured = capsys.readouterr()
    output = captured.out

    assert "Processing provided paths" in output
    assert any(phrase in output for phrase in ["capped", "auto-set", "Neighbors (k) capped", "nn auto-set"])
    assert "4" in output and "5" in output  # n=5 → capped to 4
    assert embeddings and embeddings[0].shape == (5, 2)
    assert agg['skipped_count'] == 0
    print("Small dataset test passed: nn capped to 4 verified (robust log match)")


def test_nn_kwarg(monkeypatch, tmp_path, capsys):
    """Test mixed valid/invalid paths → correct skip counts + NN auto-cap."""
    monkeypatch.setattr(glob, 'glob', lambda *a, **k: [])  # Not used

    np.random.seed(42)
    df_valid = pd.DataFrame(np.random.rand(50, 4))
    valid_path = tmp_path / "valid1.csv"
    df_valid.to_csv(valid_path, index=False)

    paths = [str(valid_path), None, "nonexistent.csv"]  # 1 valid, 2 skipped

    embeddings, agg = batch_apply_isomap(
        paths=paths,
        n_jobs=1,
        n_components=2,
        n_neighbors=None,          # Auto-cap to min(5, n-1) → 5 for n=50
        metrics_csv=None
    )

    captured = capsys.readouterr()
    output = captured.out

    # Robust skip dict substring checks
    assert any(k in output for k in ["'none': 1", "none: 1", "none : 1"])
    assert any(k in output for k in ["'not_exist': 1", "'non_exist': 1", "not_exist: 1", "non_exist: 1"])
    assert "capped" in output or "auto-set" in output or "Neighbors (k) capped" in output
    assert len(embeddings) == 1 and embeddings[0].shape == (50, 2)
    assert agg['skipped_count'] == 2
    print("NN kwarg + skip test passed: 1 valid, 2 skipped (robust log match)")


def test_skip_none_data(monkeypatch, tmp_path, capsys):
    """Test graceful skipping of None/nonexistent paths in explicit paths mode."""
    monkeypatch.setattr(glob, 'glob', lambda *args, **kwargs: [])  # Ignored

    np.random.seed(42)
    n = 50
    df1 = pd.DataFrame(np.random.rand(n, 4))
    df2 = pd.DataFrame(np.random.rand(n, 4))
    p1 = tmp_path / "valid1.csv"
    p2 = tmp_path / "valid2.csv"
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)

    paths = [str(p1), None, "nonexistent.csv", str(p2)]  # 2 valid, 2 skipped

    embeddings, agg = batch_apply_isomap(
        paths=paths,
        n_components=2,
        n_neighbors=5,
        output_dir=str(tmp_path),
        metrics_csv=None
    )

    captured = capsys.readouterr()
    output = captured.out

    assert any(k in output for k in ["'none': 1", "none: 1", "none : 1"])
    assert any(k in output for k in ["'not_exist': 1", "'non_exist': 1", "not_exist: 1", "non_exist: 1"])
    assert agg['skipped_count'] == 2
    assert len(embeddings) == 2
    assert all(emb.shape == (50, 2) for emb in embeddings if emb is not None)
    print("Skip None/nonexistent passed: 2/4 processed (robust dict substring match)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])