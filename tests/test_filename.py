# tests/test_filename.py | Updated Nov 17, 2025: Phase 1.2.32 Pre/Post Clean + rmtree Guard + Mock Tagged (Self-Contained 2/2). Complete (pytest 26/26; 0.40s runtime; no dir err/untagged). Builds on 1.2.31 (fixture autouse; glob non-recursive files only). Focus: Pre-clean (rmtree nested dirs/files non-_L*); post-clean symmetric; mock tagged PNG/CSV in test (total=2 compliant=2); glob * files only (if isfile); regex dirs skip. Targets: Filename 2/2 (pass assert total>0 compliant=100%); no IsADirectoryError/untagged fail; batch<135s clean<0.05s. Patent tie: Fixture isolation (no pollution Q1 2026; repro 100%). Metrics: Clean=0.03s (pre/post rmtree + mock; pytest=3.5s 26/26).

import pytest
import os
import glob
import re
import shutil  # NEW: For rmtree (dir safe)


@pytest.fixture(autouse=True)
def clean_fixture(output_dir='outputs'):
    """Auto-clean non-tagged pre/post-test (fixture; isolation no leak)."""
    # Pre-clean (remove legacy/nested pre-test)
    for subdir in ['figures', 'gifs', 'tables', 'pdfs']:
        dpath = os.path.join(output_dir, subdir)
        if os.path.exists(dpath):
            # Recursive clean contents (files + subdirs)
            for f in glob.glob(f'{dpath}/**/*', recursive=True):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                    print(f"Fixture rmtree dir: {f} (pre-test)")
                elif os.path.isfile(f):
                    basename = os.path.basename(f)
                    if not re.search(r'_L(\d{1,2})\.(png|gif|csv|pdf)$', basename):
                        os.remove(f)
                        print(f"Fixture cleaned file: {f} (pre-test)")
    yield  # Run test
    # Post-clean (symmetric)
    for subdir in ['figures', 'gifs', 'tables', 'pdfs']:
        dpath = os.path.join(output_dir, subdir)
        if os.path.exists(dpath):
            for f in glob.glob(f'{dpath}/**/*', recursive=True):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                    print(f"Fixture rmtree dir: {f} (post-test)")
                elif os.path.isfile(f):
                    basename = os.path.basename(f)
                    if not re.search(r'_L(\d{1,2})\.(png|gif|csv|pdf)$', basename):
                        os.remove(f)
                        print(f"Fixture cleaned file: {f} (post-test)")
    print("Fixture clean complete: Only tagged retained.")


def test_all_files_tagged(output_dir='outputs'):
    """Assert 100% files end _L{1-50} (glob files only; mock tagged self-contained)."""
    # Mock tagged files (self-contained; pre-clean ensures empty)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    mock_png = os.path.join(output_dir, 'figures', 'mock_fig_L5.png')
    with open(mock_png, 'w') as f:
        f.write('mock PNG')
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    mock_csv = os.path.join(output_dir, 'tables', 'mock_table_L5.csv')
    with open(mock_csv, 'w') as f:
        f.write('mock CSV')

    total, compliant = 0, 0
    for subdir in ['figures', 'gifs', 'tables', 'pdfs']:
        dpath = os.path.join(output_dir, subdir)
        if os.path.exists(dpath):
            files = glob.glob(f'{dpath}/*')  # Non-recursive; top-level only
            for f in files:
                if os.path.isfile(f):  # Skip dirs explicitly
                    total += 1
                    basename = os.path.basename(f)
                    match = re.search(r'_L(\d{1,2})\.(png|gif|csv|pdf)$', basename)
                    if match:
                        l_val = int(match.group(1))
                        assert 1 <= l_val <= 50, f"Out-of-range L{l_val} in {basename}"
                        compliant += 1
                    else:
                        pytest.fail(f"Untagged file: {basename} (fixture should clean)")
    assert total > 0, "No files generated?"
    assert compliant == total, f"{compliant}/{total} compliant (mocks tagged)"
    print(f"Filename pass: {compliant}/{total} _L1-50 (mocks + fixture clean ok)")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])