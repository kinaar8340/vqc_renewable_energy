# Vortex Quaternion Conduit (VQC) OAM Simulations
Ultra-high-density quantum data compression/transfer via OAM-flux qubits and quaternion encoding.  
Provisional patent filed Oct 28, 2025: US 63/913,110 | Amendment Nov 15, 2025

## Public Release Notes – Phase 1.2.40 (Nov 19, 2025)
This repository contains the complete simulation framework (QuTiP + PySCF + scikit-learn) achieving:
- L_max = 1999 (101 orthogonal OAM states per photon)
- 8-qubit QEC enabled
- Isomap stress 0.041 at 7D quaternion augmentation
- 100% yield on batch sweeps (pytest 26/26)

Real simulation results (data/L## folders) are withheld for patent enablement.
Run your own sweeps – everything needed is here.

**Dependencies**  
pip install pytest-mock  # required only for tests/test_dashboard.py (no runtime impact)

Configuration
Edit only one place: configs/params.yaml → set qubit_multi.L_max (now the single source of truth).Standard recommended run (respects YAML config, parallelized, auto-archives results):bash

OMP_NUM_THREADS=16 python run_all.py --n_jobs 8  # CLI execution command

OMP_NUM_THREADS=16 python run_all.py --L_max 1999 --n_jobs 8   # Override L_max from CLI, ignores YAML

OMP_NUM_THREADS=16 python run_all.py --n_jobs 8 --keep-outputs  # Keep transient outputs for inspection (otherwise auto-wiped after archiving):bash

VQC_QEC_4QUBIT=true OMP_NUM_THREADS=16 python run_all.py --L_max 1999 --n_jobs 16  # sets 4 QuBits for sims

VQC_QEC_8QUBIT=true OMP_NUM_THREADS=16 python run_all.py --L_max 1999 --n_jobs 16  # sets 8 QuBits for sims

Outputs  During run: written to transient outputs/{figures,gifs,tables,pdfs}/ with tag _L##  
Post-run: automatically archived to persistent data/L##/ and outputs/ cleaned (unless --keep-outputs)  
Full batch time (L=25): ~256 s (ntraj=1, OMP=16, 90% utilization on Dell PowerEdge 630)

Standalone Scripts (all accept --L_max; overridden by run_all.py when orchestrated)bash

python src/qubit_dynamics.py --L_max 25          # CSVs/PNGs _L25; FID=0.999 post-QEC
python src/photonics.py --L_max 25               # CSV _L25; chunked n=5100
python src/demixing.py --L_max 25                # CSV _L25; post=0.378 [tune pending]
python src/chem_error_corr.py --L_max 25         # CSV _L25; fid=0.991
python src/knots.py --L_max 25                   # sweep/summary _L25; PDF >50KB
python src/isomap_integration.py --L_max 25      # metrics _L25; stress=0.041
python analysis/fidelity_sweep.py --L_max 25     # PNG/CSV _L25
python analysis/isomap_anim.py --csv_path outputs/tables/chem_qec_L25.csv --l_max 25  # GIF _L25; 30-frame chirp+rot
streamlit run analysis/dashboard.py              # auto-detects latest _L##; <2.5s load

Testingbash

pytest tests/ -v                              # 26/26 passed; dashboard uses mocked fixtures
pytest tests/test_metrics.py -v               # filename regex + tempfile GIF checks; 9/9

Project Structure (Phase 1.2.40)

vqc_sims/
├── run_all.py                  # orchestrator; L_max from YAML → CLI optional; auto-archive + cleanup
├── configs/
│   └── params.yaml             # SINGLE SOURCE of L_max (qubit_multi.L_max) + demix tuning
├── src/                        # core simulation modules (all append _L## tags)
├── analysis/                   # visualization & post-processing
├── tests/                      # comprehensive tests (mocks, filename glob, cleanup)
├── outputs/                    # ← transient, auto-wiped after successful run
└── data/                       # ← **persistent archive**, one folder per L value

Contact: kinaar0@protonmail.com
#VQC #QuantumComms #PatentPending
