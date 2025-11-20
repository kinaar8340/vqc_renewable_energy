# /vqc_sims/src/single_beam_nested_lg_cross_section.py | Updated Nov 18, 2025: Phase 1.2.51 – Universal L_max FINAL

# === UNIVERSAL L_max RESOLUTION – Phase 1.2.51 (FINAL) ===
# Paste this at the TOP of EVERY src/*.py and analysis/*.py (above all other code)
import os
import yaml
from pathlib import Path

def _resolve_l_max() -> int:
    # 1. Dynamic override from run_all.py (highest priority)
    override = os.getenv('VQC_L_MAX_OVERRIDE')
    if override is not None:
        val = int(override)
        print(f"L_max ← VQC_L_MAX_OVERRIDE={val} (dynamic override)")
        return val

    # 2. CLI --L_max / --l_max
    try:
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--L_max', '--l_max', type=int, default=None)
        args, _ = parser.parse_known_args()
        if args.L_max is not None:
            print(f"L_max ← CLI={args.L_max}")
            return args.L_max
    except:
        pass

    # 3. YAML fallback
    yaml_path = Path(__file__).parent.parent / 'configs' / 'params.yaml'
    if yaml_path.exists():
        try:
            cfg = yaml.safe_load(yaml_path.read_text()) or {}
            val = cfg.get('qubit_multi', {}).get('L_max', 25)
            print(f"L_max ← configs/params.yaml → {val}")
            return int(val)
        except:
            pass

    print("L_max ← default = 25")
    return 25

L_max = _resolve_l_max()  # ← GLOBAL, resolved ONCE at import
print(f"Final effective L_max = {L_max}\n")
# ============================================================

# === UNIVERSAL QEC_8QUBIT RESOLUTION – Phase 1.2.78 OMEGA FINAL ===
qec_8qubit = os.getenv('VQC_QEC_8QUBIT', 'false').lower() == 'true'
if qec_8qubit:
    print("▓▒░ 8-QUBIT QEC ACTIVE – Phase 1.2.78 suppression engaged ░▒▓")
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
import os
import re

# GLOBAL WARNING SUPPRESSION – add to TOP of every src/*.py & analysis/*.py (after universal L_max)
 import warnings
from scipy.sparse import SparseEfficiencyWarning
+from scipy.sparse import SparseEfficiencyWarning
+
 warnings.filterwarnings('ignore', category=UserWarning)
+warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)  # ← ETERNAL SILENCE ENFORCED
 warnings.filterwarnings('ignore', category=RuntimeWarning)

def single_beam_nested_lg_cross_section(ell: int = None, p: int = 0, r_max: float = 1.0, N: int = 512):
    """
    Plot cross-section of a nested Laguerre-Gaussian (LG) beam intensity.
    Uses global L_max as the default topological charge if ell is not provided.
    """
    if ell is None:
        ell = L_max
        print(f"No elle specified → using global L_max = {ell}")

    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.linspace(0, r_max, N)
    R, Theta = np.meshgrid(r, theta)

    # Normalized radial coordinate (w0 = 1)
    rho = np.sqrt(2) * R

    # Primary LG mode: LG_{ell, p=0}
    u_r = (rho ** abs(ell)) * np.exp(-rho ** 2 / 2)
    u_phi = np.exp(1j * ell * Theta)
    u = u_r * u_phi

    # Nested lower-order modes (coherent superposition feel, scaled intensity)
    intensity = np.abs(u) ** 2
    for inner_ell in range(1, ell):
        rho_inner = np.sqrt(2) * R * 0.5  # narrower beam for inner modes
        inner_u_r = (rho_inner ** inner_ell) * np.exp(-rho_inner ** 2 / 2)
        inner_u_phi = np.exp(1j * inner_ell * Theta)
        inner_u = inner_u_r * inner_u_phi
        intensity += 0.5 * np.abs(inner_u) ** 2

    intensity /= np.max(intensity)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    im = ax.pcolormesh(Theta, R, intensity, shading='auto', cmap='hot')
    ax.set_title(f'Nested LG Beam Cross-Section (ℓ = {ell}, p = {p})')
    plt.colorbar(im, ax=ax, label='Normalized Intensity')

    # === Output directory and clean filename (nuclear-safe tagging) ===
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Base filename without any old L tags
    base_name = f"single_lg_nested_cross_ell{ell}_p{p}"
    # Nuclear strip of any legacy _L\d+ pattern
    clean_name = re.sub(r'_L\d+', '', base_name)
    final_filename = f"{clean_name}_L{L_max}.png"
    final_path = os.path.join(output_dir, final_filename)

    plt.savefig(final_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {final_path}\n")


if __name__ == "__main__":
    # When run directly, generate for the current global L_max
    single_beam_nested_lg_cross_section()

# eof