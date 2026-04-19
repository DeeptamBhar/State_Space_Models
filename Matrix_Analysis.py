import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.linalg import svd, subspace_angles, eig, solve, lstsq
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

DATASETS = {"DNA": "extracted_matrices/dna_sequence.npz", "INDUCTION": "extracted_matrices/induction_heads.npz", "COPY": "extracted_matrices/selective_copying.npz"}

DATASET_LAYERS = {"DNA": 4, "INDUCTION": 2, "COPY": 2}

N = 16
OUT_DIR = "mamba_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)


def hippo_legs(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i > k:
                A[i, k] = (2*i + 1)**0.5 * (2*k + 1)**0.5
            elif i == k:
                A[i, k] = i + 1
    return -A


def hippo_legt(n: int, theta: float = 1.0) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i > k:
                A[i, k] = (2*i + 1)**0.5 * (2*k + 1)**0.5 / theta
            elif i == k:
                A[i, k] = (i + 1) / theta
    return -A


def hippo_lagt(n: int, alpha: float = 0.0) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1):
            if i == k:
                A[i, k] = -0.5 * (alpha + 1)
            else:
                A[i, k] = -1.0
    return A


HIPPO_MATRICES = {"LegS": hippo_legs(N), "LegT": hippo_legt(N), "LagT": hippo_lagt(N)}


def load_dataset(name: str, path: str, n_layers: int) -> dict:
    data = np.load(path, allow_pickle=True)
    layers = {}
    for l in range(n_layers):
        prefix = f"layer_{l}__"
        layer_data = {k.replace(prefix, ""): data[k] for k in data.files if k.startswith(prefix)}
        layers[l] = layer_data
    print(f"[{name}] Loaded {n_layers} layers. Keys per layer: {list(list(layers.values())[0].keys())}")
    return layers


def collapse_method1_projection(A_bar: np.ndarray) -> np.ndarray:
    if A_bar.ndim == 3:
        A_avg = A_bar.mean(axis=0)
    else:
        A_avg = A_bar
    d_in, d_st = A_avg.shape
    sz = max(d_in, d_st, N)
    A_sq = np.zeros((sz, sz))
    A_sq[:d_in, :d_st] = A_avg
    rng = np.random.default_rng(42)
    P_init = rng.standard_normal((sz, N))

    def loss(p_flat):
        P = p_flat.reshape(sz, N)
        A_small = np.linalg.lstsq(P, A_sq @ P, rcond=None)[0]
        return np.linalg.norm(A_sq @ P - P @ A_small, 'fro')**2

    res = minimize(loss, P_init.ravel(), method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9})
    P_opt = res.x.reshape(sz, N)
    A_16 = np.linalg.lstsq(P_opt, A_sq @ P_opt, rcond=None)[0]
    return A_16


def collapse_method2_mean(A_bar: np.ndarray) -> np.ndarray:
    if A_bar.ndim == 3:
        A_avg = A_bar.mean(axis=0)
    else:
        A_avg = A_bar
    r = min(A_avg.shape[0], N)
    c = min(A_avg.shape[1], N)
    A_16 = np.zeros((N, N))
    A_16[:r, :c] = A_avg[:r, :c]
    return A_16


def collapse_method3_svd(A_bar: np.ndarray) -> np.ndarray:
    if A_bar.ndim == 3:
        A_avg = A_bar.mean(axis=0)
    else:
        A_avg = A_bar
    U, S, Vt = svd(A_avg, full_matrices=False)
    k = min(N, len(S))
    A_16 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    r = min(A_16.shape[0], N)
    c = min(A_16.shape[1], N)
    out = np.zeros((N, N))
    out[:r, :c] = A_16[:r, :c]
    return out


COLLAPSE_METHODS = {"Projection": collapse_method1_projection, "Mean": collapse_method2_mean, "SVD": collapse_method3_svd}


def get_collapsed_matrices(layers: dict) -> dict:
    result = {}
    for l, ldata in layers.items():
        result[l] = {}
        A_bar = ldata.get("A_bar")
        if A_bar is None:
            print(f"  Warning: no A_bar in layer {l}")
            continue
        for mname, mfunc in COLLAPSE_METHODS.items():
            try:
                result[l][mname] = mfunc(A_bar)
            except Exception as e:
                print(f"  Warning: collapse {mname} layer {l} failed: {e}")
                result[l][mname] = np.zeros((N, N))
    return result


def compute_subspace_angles(A_m: np.ndarray, A_h: np.ndarray) -> np.ndarray:
    try:
        angles = subspace_angles(A_m, A_h)
        return np.degrees(angles)
    except Exception:
        return np.full(N, np.nan)


def frobenius_distance(A_m: np.ndarray, A_h: np.ndarray) -> float:
    return float(np.linalg.norm(A_m - A_h, 'fro'))


def compute_eigenvalues(A: np.ndarray):
    try:
        return np.linalg.eigvals(A)
    except Exception:
        return np.array([])


def projection_onto_hippo(A_m: np.ndarray, A_h: np.ndarray):
    a_m = A_m.ravel()
    a_h = A_h.ravel()
    norm_h = np.linalg.norm(a_h)
    if norm_h < 1e-12:
        return 0.0, np.linalg.norm(a_m), np.zeros_like(A_m)
    coeff = np.dot(a_m, a_h) / (norm_h**2)
    proj = coeff * A_h
    resid = np.linalg.norm(A_m - proj, 'fro')
    return float(coeff), float(resid), proj


def projection_cosine_similarity(A_m: np.ndarray, A_h: np.ndarray) -> float:
    a_m = A_m.ravel()
    a_h = A_h.ravel()
    denom = (np.linalg.norm(a_m) * np.linalg.norm(a_h))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_m, a_h) / denom)


def operator_shift(A_ref: np.ndarray, A_target: np.ndarray):
    delta = A_target - A_ref
    U, S, _ = svd(delta)
    return delta, float(S[0]), U[:, 0]


PALETTE = {
    "DNA": "#E63946",
    "INDUCTION": "#2A9D8F",
    "COPY": "#F4A261",
    "LegS": "#457B9D",
    "LegT": "#6A4C93",
    "LagT": "#1D7874",
    "Projection": "#E9C46A",
    "Mean": "#F4A261",
    "SVD": "#E76F51",
}

MARKER = {"DNA": "o", "INDUCTION": "s", "COPY": "^"}

plt.rcParams.update({"font.family": "monospace", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  -> saved {path}")


def run_analysis():
    all_layers = {}
    all_collapsed = {}

    for dname, dpath in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Dataset: {dname}")
        print(f"{'='*60}")
        nl = DATASET_LAYERS[dname]
        layers = load_dataset(dname, dpath, nl)
        all_layers[dname] = layers
        collapsed = get_collapsed_matrices(layers)
        all_collapsed[dname] = collapsed
        print(f"  Collapsed layers: {list(collapsed.keys())}")

    pca_labels = []
    pca_vectors = []
    results = {}

    for dname, collapsed in all_collapsed.items():
        results[dname] = {}
        for layer, method_dict in collapsed.items():
            results[dname][layer] = {}
            for mname, A_16 in method_dict.items():
                results[dname][layer][mname] = {}
                for hname, A_h in HIPPO_MATRICES.items():
                    angles = compute_subspace_angles(A_16, A_h)
                    frob = frobenius_distance(A_16, A_h)
                    coeff, resid, proj_mat = projection_onto_hippo(A_16, A_h)
                    cosine = projection_cosine_similarity(A_16, A_h)
                    results[dname][layer][mname][hname] = {"subspace_angles": angles, "frobenius": frob, "proj_coeff": coeff, "proj_residual": resid, "cosine_sim": cosine, "A_16": A_16}
                if mname == "SVD":
                    pca_vectors.append(A_16.ravel())
                    pca_labels.append(f"{dname}_L{layer}")

    for hname, A_h in HIPPO_MATRICES.items():
        pca_vectors.append(A_h.ravel())
        pca_labels.append(f"HiPPO_{hname}")

    print("\n[Plot 1] Frobenius Distance Heatmaps")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Frobenius Distance: A_mamba(SVD) vs HiPPO variants", fontsize=13, fontweight="bold")
    for ax, (dname, ddata) in zip(axes, results.items()):
        n_layers = len(ddata)
        hnames = list(HIPPO_MATRICES.keys())
        mat = np.zeros((n_layers, len(hnames)))
        for li, layer in enumerate(sorted(ddata.keys())):
            for hi, hn in enumerate(hnames):
                mat[li, hi] = ddata[layer]["SVD"][hn]["frobenius"]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(hnames)))
        ax.set_xticklabels(hnames, fontsize=10)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"Layer {l}" for l in sorted(ddata.keys())], fontsize=9)
        ax.set_title(dname, fontweight="bold")
        for i in range(n_layers):
            for j in range(len(hnames)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig("01_frobenius_heatmap.png")

    print("[Plot 2] Principal Angles")
    fig, axes = plt.subplots(len(HIPPO_MATRICES), 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Subspace Principal Angles: A_mamba vs HiPPO", fontsize=13, fontweight="bold")
    for ax, hname in zip(axes, HIPPO_MATRICES):
        for dname, ddata in results.items():
            for layer in sorted(ddata.keys()):
                angles = ddata[layer]["SVD"][hname]["subspace_angles"]
                ax.plot(angles, marker=".", label=f"{dname} L{layer}", color=PALETTE[dname], alpha=0.7, linestyle=["-", "--", ":", "-."][layer % 4])
        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"vs HiPPO-{hname}", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Principal Angle Index")
    plt.tight_layout()
    savefig("02_subspace_angles.png")

    print("[Plot 3] Eigenvalue Spectra")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    ax = axes[0]
    ax.set_title("Mamba A_mamba eigenvalues (SVD collapse)", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    for dname, ddata in results.items():
        for layer in sorted(ddata.keys()):
            A_16 = ddata[layer]["SVD"]["LegS"]["A_16"]
            ev = compute_eigenvalues(A_16)
            ax.scatter(ev.real, ev.imag, s=30, alpha=0.7, color=PALETTE[dname], marker=MARKER[dname], label=f"{dname} L{layer}" if layer == 0 else "")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.legend(fontsize=7)
    for ax_i, (hname, A_h) in enumerate(HIPPO_MATRICES.items(), start=1):
        ax = axes[ax_i]
        ev_h = compute_eigenvalues(A_h)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.scatter(ev_h.real, ev_h.imag, s=60, zorder=5, color=PALETTE[hname], edgecolors="black", linewidths=0.5, label=f"HiPPO-{hname}", marker="D")
        for dname, ddata in results.items():
            for layer in sorted(ddata.keys()):
                A_16 = ddata[layer]["SVD"][hname]["A_16"]
                ev = compute_eigenvalues(A_16)
                ax.scatter(ev.real, ev.imag, s=18, alpha=0.35, color=PALETTE[dname], marker=MARKER[dname])
        ax.set_title(f"Mamba vs HiPPO-{hname}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.legend(fontsize=8)
    plt.suptitle("Eigenvalue Spectrum Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("03_eigenvalue_spectra.png")

    print("[Plot 4] Projection onto HiPPO Basis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cosine Similarity: A_mamba -> HiPPO Basis", fontsize=13, fontweight="bold", color="#E63946")
    for ax, mname in zip(axes, COLLAPSE_METHODS):
        all_ds_rows = []
        row_labels = []
        for dname, ddata in results.items():
            for layer in sorted(ddata.keys()):
                row = [ddata[layer][mname][hn]["cosine_sim"] for hn in HIPPO_MATRICES]
                all_ds_rows.append(row)
                row_labels.append(f"{dname} L{layer}")
        mat_cos = np.array(all_ds_rows)
        im = ax.imshow(mat_cos, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(HIPPO_MATRICES)))
        ax.set_xticklabels(list(HIPPO_MATRICES.keys()), fontsize=10)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(f"Collapse: {mname}", fontweight="bold")
        for i in range(len(row_labels)):
            for j in range(len(HIPPO_MATRICES)):
                ax.text(j, i, f"{mat_cos[i,j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig("04a_projection_cosine_heatmap.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Projection Coefficient onto HiPPO Bases (SVD collapse)", fontsize=12, fontweight="bold", color="#E63946")
    hnames = list(HIPPO_MATRICES.keys())
    x = np.arange(len(hnames))
    width = 0.22
    for ax, dname in zip(axes, results):
        ddata = results[dname]
        for li, layer in enumerate(sorted(ddata.keys())):
            coeffs = [ddata[layer]["SVD"][hn]["proj_coeff"] for hn in hnames]
            ax.bar(x + li * width, coeffs, width, label=f"Layer {layer}", color=PALETTE[dname], alpha=0.7 - 0.1 * li, edgecolor="black", linewidth=0.6)
        ax.set_title(dname, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(hnames)
        ax.set_ylabel("Projection Coefficient")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
    plt.tight_layout()
    savefig("04b_projection_coefficients.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Projection Residual ||A_m - alpha*A_h||_F  (lower = more HiPPO-like)", fontsize=12, fontweight="bold")
    for ax, dname in zip(axes, results):
        ddata = results[dname]
        for li, layer in enumerate(sorted(ddata.keys())):
            resids = [ddata[layer]["SVD"][hn]["proj_residual"] for hn in hnames]
            ax.plot(hnames, resids, marker="o", label=f"Layer {layer}", color=PALETTE[dname], linestyle=["-", "--", ":", "-."][layer % 4])
        ax.set_title(dname, fontweight="bold")
        ax.set_ylabel("Residual (Frobenius)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("04c_projection_residuals.png")

    print("[Plot 5] Dataset-Conditioned Operator Shift")
    ref_ds, ref_layer = "DNA", 0
    ref_A = all_collapsed[ref_ds][ref_layer]["SVD"]
    shift_records = []
    for dname, collapsed in all_collapsed.items():
        for layer in sorted(collapsed.keys()):
            A_t = collapsed[layer]["SVD"]
            delta, shift_mag, _ = operator_shift(ref_A, A_t)
            shift_records.append({"label": f"{dname} L{layer}", "dataset": dname, "layer": layer, "shift": shift_mag, "delta": delta})

    labels_s = [r["label"] for r in shift_records]
    shifts_s = [r["shift"] for r in shift_records]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Dataset-Conditioned Operator Shift  (ref = {ref_ds} L{ref_layer})", fontsize=12, fontweight="bold", color="#E63946")
    ax = axes[0]
    colors_bar = [PALETTE[r["dataset"]] for r in shift_records]
    bars = ax.barh(labels_s, shifts_s, color=colors_bar, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("||delta_A||_2  (spectral norm of shift)")
    ax.set_title("Operator Shift Magnitude from Reference")
    ax.axvline(0, color="black", lw=1)
    for bar, val in zip(bars, shifts_s):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    ax = axes[1]
    n_show = min(4, len(shift_records))
    delta_stack = np.hstack([shift_records[i]["delta"] for i in range(n_show)])
    im = ax.imshow(delta_stack, aspect="auto", cmap="RdBu_r", vmin=-np.percentile(np.abs(delta_stack), 95), vmax=np.percentile(np.abs(delta_stack), 95))
    ax.set_title(f"delta_A matrices (first {n_show} operators)")
    tick_pos = [N//2 + i*N for i in range(n_show)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([shift_records[i]["label"] for i in range(n_show)], fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    savefig("05a_operator_shift.png")

    print("[Plot 5b] Cross-dataset pairwise shift matrix")
    ds_keys = list(all_collapsed.keys())
    n_ds = len(ds_keys)
    pairwise = np.zeros((n_ds, n_ds))
    for i, d1 in enumerate(ds_keys):
        for j, d2 in enumerate(ds_keys):
            A1 = all_collapsed[d1][0]["SVD"]
            A2 = all_collapsed[d2][0]["SVD"]
            _, s, _ = operator_shift(A1, A2)
            pairwise[i, j] = s
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pairwise, cmap="plasma")
    ax.set_xticks(range(n_ds)); ax.set_xticklabels(ds_keys)
    ax.set_yticks(range(n_ds)); ax.set_yticklabels(ds_keys)
    ax.set_title("Pairwise Operator Shift (Layer 0, SVD)", fontweight="bold", color="#E63946")
    for i in range(n_ds):
        for j in range(n_ds):
            ax.text(j, i, f"{pairwise[i,j]:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if pairwise[i,j] > pairwise.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    savefig("05b_pairwise_shift.png")

    print("[Plot 5c] Shift directions in HiPPO subspace")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Operator Shift Direction Alignment with HiPPO Bases", fontsize=12, fontweight="bold")
    for ax, hname in zip(axes, HIPPO_MATRICES):
        A_h_flat = HIPPO_MATRICES[hname].ravel()
        alignments = []
        lbls = []
        for r in shift_records:
            d_flat = r["delta"].ravel()
            norm_d = np.linalg.norm(d_flat)
            if norm_d < 1e-12:
                alignments.append(0.0)
            else:
                alignments.append(float(np.dot(d_flat, A_h_flat) / (norm_d * np.linalg.norm(A_h_flat) + 1e-12)))
            lbls.append(r["label"])
        colors_align = [PALETTE[r["dataset"]] for r in shift_records]
        ax.barh(lbls, alignments, color=colors_align, edgecolor="black", linewidth=0.6)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"Shift -> HiPPO-{hname}", fontweight="bold")
        ax.set_xlabel("Cosine alignment of delta_A with HiPPO")
    plt.tight_layout()
    savefig("05c_shift_hippo_alignment.png")

    print("[Plot 6] PCA Clustering")
    X = np.array(pca_vectors)
    X_sc = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(4, X_sc.shape[0] - 1))
    X_pca = pca.fit_transform(X_sc)
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("PCA Clustering of Mamba A Matrices + HiPPO Matrices", fontsize=14, fontweight="bold")
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, lbl in enumerate(pca_labels):
        is_hippo = lbl.startswith("HiPPO")
        ds_key = lbl.split("_")[0]
        color = PALETTE.get(ds_key, "#888888")
        size = 120 if is_hippo else 60
        marker = "D" if is_hippo else MARKER.get(ds_key, "o")
        ec = "black" if is_hippo else "none"
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], s=size, c=color, marker=marker, edgecolors=ec, linewidths=1.0, zorder=3 if is_hippo else 2)
        ax1.annotate(lbl, (X_pca[idx, 0], X_pca[idx, 1]), fontsize=7, ha="left", va="bottom", alpha=0.85)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("PC1 vs PC2")
    ax2 = fig.add_subplot(gs[0, 1])
    if X_pca.shape[1] >= 4:
        for idx, lbl in enumerate(pca_labels):
            is_hippo = lbl.startswith("HiPPO")
            ds_key = lbl.split("_")[0]
            color = PALETTE.get(ds_key, "#888888")
            size = 120 if is_hippo else 60
            marker = "D" if is_hippo else MARKER.get(ds_key, "o")
            ec = "black" if is_hippo else "none"
            ax2.scatter(X_pca[idx, 2], X_pca[idx, 3], s=size, c=color, marker=marker, edgecolors=ec, linewidths=1.0)
            ax2.annotate(lbl, (X_pca[idx, 2], X_pca[idx, 3]), fontsize=7, ha="left", va="bottom", alpha=0.85)
        ax2.set_xlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax2.set_ylabel(f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)")
        ax2.set_title("PC3 vs PC4")
    ax3 = fig.add_subplot(gs[1, 0])
    n_comps = pca.n_components_
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax3.bar(range(1, n_comps + 1), pca.explained_variance_ratio_ * 100, color="#457B9D", edgecolor="black", linewidth=0.6)
    ax3.plot(range(1, n_comps + 1), cumvar * 100, "r-o", ms=6, label="Cumulative")
    ax3.set_xlabel("Principal Component")
    ax3.set_ylabel("% Variance Explained")
    ax3.set_title("Scree Plot")
    ax3.legend()
    ax4 = fig.add_subplot(gs[1, 1])
    dist_mat = np.zeros((len(pca_labels), len(pca_labels)))
    for i in range(len(pca_labels)):
        for j in range(len(pca_labels)):
            dist_mat[i, j] = np.linalg.norm(X_pca[i] - X_pca[j])
    im = ax4.imshow(dist_mat, cmap="viridis", aspect="auto")
    ax4.set_xticks(range(len(pca_labels)))
    ax4.set_yticks(range(len(pca_labels)))
    ax4.set_xticklabels(pca_labels, rotation=45, ha="right", fontsize=7)
    ax4.set_yticklabels(pca_labels, fontsize=7)
    ax4.set_title("Pairwise Distance in PCA Space")
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    savefig("06_pca_clustering.png")

    print("[Plot 7] Collapse Method Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Collapse Method Comparison: Frobenius vs HiPPO-LegS", fontsize=12, fontweight="bold")
    for ax, dname in zip(axes, results):
        ddata = results[dname]
        x_pos = np.arange(len(COLLAPSE_METHODS))
        width = 0.25
        for li, layer in enumerate(sorted(ddata.keys())):
            frobs = [ddata[layer][mname]["LegS"]["frobenius"] for mname in COLLAPSE_METHODS]
            ax.bar(x_pos + li * width, frobs, width, label=f"Layer {layer}", edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(list(COLLAPSE_METHODS.keys()))
        ax.set_title(dname, fontweight="bold")
        ax.set_ylabel("Frobenius Distance")
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("07_collapse_method_comparison.png")

    print("[Plot 8] A_16 Matrix Heatmaps")
    n_cols = 3 + len(HIPPO_MATRICES)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle("Collapsed A_16 matrices (SVD) vs HiPPO matrices", fontsize=12, fontweight="bold")
    for ax, (hname, A_h) in zip(axes[:len(HIPPO_MATRICES)], HIPPO_MATRICES.items()):
        im = ax.imshow(A_h, cmap="RdBu_r", vmax=np.percentile(np.abs(A_h), 98), vmin=-np.percentile(np.abs(A_h), 98))
        ax.set_title(f"HiPPO-{hname}", fontweight="bold")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, fraction=0.046)
    offset = len(HIPPO_MATRICES)
    for ax, dname in zip(axes[offset:], results):
        A_16 = all_collapsed[dname][0]["SVD"]
        v = np.percentile(np.abs(A_16), 98) + 1e-9
        im = ax.imshow(A_16, cmap="RdBu_r", vmax=v, vmin=-v)
        ax.set_title(f"{dname} L0 (SVD)", fontweight="bold")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    savefig("08_matrix_heatmaps.png")

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("MAMBA vs HiPPO  -  NUMERICAL SUMMARY")
    summary_lines.append("=" * 80)
    for dname, ddata in results.items():
        summary_lines.append(f"\nDataset: {dname}")
        summary_lines.append("-" * 60)
        for layer in sorted(ddata.keys()):
            for mname in COLLAPSE_METHODS:
                summary_lines.append(f"  Layer {layer}  |  Collapse: {mname}")
                for hname in HIPPO_MATRICES:
                    r = ddata[layer][mname][hname]
                    mean_ang = np.nanmean(r["subspace_angles"])
                    summary_lines.append(f"    vs {hname:4s} -> Frob={r['frobenius']:7.4f}  CosSim={r['cosine_sim']:+.4f}  ProjCoeff={r['proj_coeff']:+.4f}  Residual={r['proj_residual']:7.4f}  MeanAngle={mean_ang:6.2f} deg")
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("OPERATOR SHIFT (ref = DNA Layer 0, SVD collapse)")
    summary_lines.append("=" * 80)
    for r in shift_records:
        summary_lines.append(f"  {r['label']:20s}  shift_magnitude = {r['shift']:.6f}")
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\n  -> Summary saved to {summary_path}")
    print(f"\n{'='*60}")
    print(f"  All outputs saved to: ./{OUT_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")
    print("=" * 60)


if __name__ == "__main__":
    run_analysis()