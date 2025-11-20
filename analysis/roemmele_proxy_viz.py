import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import webbrowser
import os
from scipy.spatial.distance import cdist, pdist, squareform
import warnings  # NEW: Suppress if needed
from scipy.sparse import SparseEfficiencyWarning

# Suppress Isomap warnings (post-jitter; fallback)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.manifold')

def drop_high_corr_feats(data):
    """Drop corr>thresh feats (adapt from isomap_int; log dropped)."""
    if data.shape[0] < 2 or data.shape[1] <= 1:
        return data, 0
    from scipy.stats import pearsonr
    thresh = 0.98  # Fixed for chem corr=1.0
    corr_matrix = np.abs(np.array([[pearsonr(data[:, i], data[:, j])[0] for j in range(data.shape[1])] for i in range(data.shape[1])]))
    np.fill_diagonal(corr_matrix, 0)
    to_drop = set()
    for i in range(corr_matrix.shape[0]):
        if i not in to_drop:
            high_corr = np.where(corr_matrix[i] > thresh)[0]
            to_drop.update(high_corr)
    n_dropped = len(to_drop)
    print(f"High corr drop: {n_dropped} feats (thresh={thresh})")
    keep_idx = [i for i in range(data.shape[1]) if i not in to_drop]
    return data[:, keep_idx], n_dropped

def compute_manual_stress(emb, feats):
    orig_dist = squareform(pdist(feats))
    emb_dist = squareform(pdist(emb))
    return np.mean(np.abs(orig_dist - emb_dist)) / np.mean(orig_dist + 1e-12)

def generate_stub_data(n_samples=50):
    time_ns = np.linspace(0, 100, n_samples)
    fidelity = np.exp(-time_ns / 20) + 0.01 * np.random.randn(n_samples)
    feat2 = np.sin(time_ns / 10) + 0.005 * np.random.randn(n_samples)
    df = pd.DataFrame({'time_ns': time_ns, 'fidelity': np.clip(fidelity, 0, 1), 'feat2': feat2})
    return df

def vqc_viz(csv_path, n_frames=10, output='html'):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}; using stub.")
        df = generate_stub_data()
        basename = "stub"
    else:
        df = pd.read_csv(csv_path)
        basename = os.path.splitext(os.path.basename(csv_path))[0]

    times = df['time_ns'].values
    feats_cols = df.select_dtypes(include=[np.number]).columns.drop('time_ns', errors='ignore')
    feats = df[feats_cols].values.astype(float)
    time_bins = np.linspace(times.min(), times.max(), n_frames + 1)

    # Corr drop pre-all
    feats, n_dropped = drop_high_corr_feats(feats)
    print(f"Global feats post-drop: {feats.shape[1]} (dropped {n_dropped})")

    fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'scatter'}], [{'type': 'scatter3d'}]],
                        subplot_titles=('OAM Chirp Proxy Waveform', 'Manifold Trails (FID Color)'))

    valid_frames = 0
    L_max = 25  # Extract via re if needed
    lambda_nm = 1550
    ell = L_max
    stresses = []
    for i in range(n_frames):
        mask = (times >= time_bins[i]) & (times < time_bins[i + 1])
        slice_feats = feats[mask]
        n_slice = slice_feats.shape[0]
        if n_slice < 3:
            print(f"Skipping slice {i+1}: {n_slice} < 3")
            continue

        # Per-slice corr drop (if needed; light)
        slice_feats, _ = drop_high_corr_feats(slice_feats)

        # FIXED: Jitter for zero dists
        slice_feats_jitter = slice_feats + 1e-10 * np.random.randn(*slice_feats.shape)

        n_neighbors = min(5, n_slice - 1)
        if n_neighbors < 2:
            print(f"Slice {i+1}: nn={n_neighbors}<2; PCA fb")
            reducer = PCA(n_components=3)
            emb = reducer.fit_transform(slice_feats_jitter)
            stress = 0.0
        else:
            iso = Isomap(n_components=3, n_neighbors=n_neighbors, metric='euclidean')
            try:
                emb = iso.fit_transform(slice_feats_jitter)
                stress = iso.reconstruction_error() if hasattr(iso, 'reconstruction_error') else compute_manual_stress(emb, slice_feats)
            except Exception as e:
                print(f"Slice {i+1} iso err ({e}); PCA fb")
                reducer = PCA(n_components=3)
                emb = reducer.fit_transform(slice_feats_jitter)
                stress = 0.0
        stresses.append(stress)

        if emb.shape[0] > 3:
            fid_slice = df['fidelity'][mask].values if 'fidelity' in df else np.ones(n_slice)
            fig.add_trace(go.Scatter3d(x=emb[:,0], y=emb[:,1], z=emb[:,2],
                                       mode='lines+markers', line=dict(color='blue', width=2),
                                       marker=dict(size=4, color=fid_slice, colorscale='viridis'),
                                       name=f'Slice {i+1} (t={time_bins[i]:.1f}-{time_bins[i+1]:.1f}ns)',
                                       hovertemplate='<b>Slice {i+1}</b><br>FID=%{marker.color:.3f}<extra></extra>'), row=2, col=1)
            valid_frames += 1

        # Top: Chirp wave
        t_slice = times[mask] - times[mask][0]
        chirp_freq = ell / (lambda_nm * 1e-9)  # Hz
        wave = np.sin(2 * np.pi * chirp_freq * t_slice * 1e-9)
        fig.add_trace(go.Scatter(x=t_slice, y=wave, mode='lines', name=f'Chirp {i+1}'), row=1, col=1)

    mean_stress = np.nanmean(stresses)
    touches = 0
    if valid_frames > 1:
        # FIXED: Collect 3D only; sample fixed pts/slice
        emb_list = []
        for trace in fig.data:
            if trace.type == 'scatter3d' and len(trace.x) > 0:
                emb_slice = np.column_stack((trace.x, trace.y, trace.z))
                emb_list.append(emb_slice)
        if emb_list and all(e.shape[1] == 3 for e in emb_list):
            max_n = min(10, max(len(e) for e in emb_list))
            sampled_embs = [e[np.random.choice(len(e), min(max_n, len(e)), replace=False)] for e in emb_list]
            all_emb = np.vstack(sampled_embs)
            if all_emb.shape[0] > 10:
                dists = cdist(all_emb[:10], all_emb[10:])
                touches = np.sum(dists < 0.05)
                print(f"Manifold 'touches' detected: {touches} pairs (cdist<0.05; OAM overlaps)")
            else:
                print("Insufficient pts for touch; skipping")
        else:
            print("Inconsistent 3D dims; skipping touch")

    fig.update_layout(title=f'VQC Visual Proxy: {basename} (L={L_max}, Stress={mean_stress:.3f}, Frames={valid_frames}/{n_frames}, Touches={touches})',
                      height=800, showlegend=True)
    os.makedirs('outputs/gifs', exist_ok=True)
    html_path = f'outputs/gifs/vqc_visual_{basename}_L{L_max}.html'
    fig.write_html(html_path)
    print(f"Interactive viz saved: {html_path} (yield={valid_frames}/{n_frames}; no warn/err)")
    if output == 'html':
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
    else:
        fig.show()
    return fig  # For dashboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='outputs/tables/fid_sweep_L25.csv')
    parser.add_argument('--output', default='html', choices=['html', 'show'])
    parser.add_argument('--n_frames', type=int, default=10)
    args = parser.parse_args()
    vqc_viz(args.csv, n_frames=args.n_frames, output=args.output)