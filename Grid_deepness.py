import os
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F
import glob

# Project modules
from data_loader import load_mnist
from nmf_models import Deep_NMF_2W_toN

from visualizations import (
    plot_nmf_results, 
    plot_H_signatures, 
    plot_mnist_reconstruction,
    plot_mnist_reconstruction_nmf
)

from metrics import (
    exp_effective_rank_torch,
    nuclear_over_operator_norm_torch,
)

import glob
import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    """Torch Tensor (CPU/GPU) -> numpy, otherwise return as is."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x

def save_run_toN(out_dir, run_name, Ws, H, logs, meta):
    """
    Save a Deep NMF run to a compressed .npz file.
    Contains :
      - list of matrices Ws
      - final factor H
      - logs (errors, ranks, etc.)
      - meta (params, temps, etc.)
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}.npz")

    # Convert Ws to numpy safely (list of arrays)
    Ws_np = [to_numpy(W) for W in Ws]

    np.savez_compressed(
        path,
        Ws=np.array(Ws_np, dtype=object),   # IMPORTANT: keep list structure
        H=to_numpy(H),
        errorsGD=np.array(logs["errorsGD"], dtype=np.float32),
        rankGD=np.array(logs["rankGD"], dtype=np.float32),
        nuclearrankGD=np.array(logs["nuclearrankGD"], dtype=np.float32),
        SVGD1=np.array(logs["SVGD1"], dtype=np.float32),
        SVGD2=np.array(logs["SVGD2"], dtype=np.float32),
        meta=np.array([meta], dtype=object),
    )
    return path

def save_all_figures(base_dir):
    """
    Save all open matplotlib figures to the specified directory.
    Returns the directory where figures were saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    for i, fig_num in enumerate(plt.get_fignums(), 1):
        fig = plt.figure(fig_num)
        path = os.path.join(save_dir, f"figure_{i}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")

    return save_dir

def prod_Ws(Ws, nonneg=True):
    """Compute W_prod = W0 @ W1 @ ... @ W_{L-1} (optionally with ReLU)."""
    Wp = Ws[0]
    if nonneg:
        Wp = torch.relu(Wp)
    for W in Ws[1:]:
        Wp = Wp @ (torch.relu(W) if nonneg else W)
    return Wp

def compute_metrics_from_snapshots_toN(
    snap_dir,
    A,
    eps=1e-12
):
    A = torch.tensor(A, dtype=torch.float32)  # CPU
    fro_Y = torch.norm(A, p="fro") ** 2

    epochs_metrics = []
    errorsGD = []
    rankGD = []
    nuclearrankGD = []
    SVGD1 = []
    SVGD2 = []

    for pt_path in sorted(glob.glob(os.path.join(snap_dir, "snapshot_*.pt"))):
        ckpt = torch.load(pt_path, map_location="cpu")
        epoch = ckpt["epoch"]

        Ws = ckpt["Ws"]          # list of tensors
        H_out = ckpt["H_out"]    # tensor

        # Deep product using the canonical helper
        W_prod = prod_Ws(Ws, nonneg=True)
        H_pos = torch.relu(H_out)

        WH = W_prod @ H_pos

        # relative error
        err = torch.norm(A - WH, p="fro") ** 2
        rel_err = (err / fro_Y).item()

        # spectral metrics
        eff_rank = exp_effective_rank_torch(WH, eps=eps)
        nuclear_rank = nuclear_over_operator_norm_torch(WH)

        # singular values (for logging)
        s = torch.linalg.svdvals(WH)
        sv1 = s[0].item() if s.numel() > 0 else 0.0
        sv2 = s[1].item() if s.numel() > 1 else 0.0

        # Append to lists
        epochs_metrics.append(epoch)
        errorsGD.append(rel_err)
        rankGD.append(eff_rank)
        nuclearrankGD.append(nuclear_rank)
        SVGD1.append(sv1)
        SVGD2.append(sv2)

    return errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, epochs_metrics

def plot_mnist_reconstruction_toN(A, Ws, H, index=0, image_shape=(28, 28)):
    """
    Uses plot_mnist_reconstruction_nmf for deep NMF:
    A ≈ (W0 @ ... @ WL-1) @ H
    Assumes A is (D, N) where D=784 and N=#samples.
    """
    # Effective factors
    W_eff = prod_Ws(Ws, nonneg=True)     # (D, r_last)
    H_eff = torch.relu(H)               # (r_last, N)

    plot_mnist_reconstruction_nmf(
        A, W_eff, H_eff,
        index=index,
        image_shape=image_shape
    )

def main():
    # ------------------------- CONFIG SWEEP -------------------------
    # Is the deepness sweep
    x_train_list = [2000]      # Is the deepness sweep
    epochs_list  = [100000]      # Tested number of epochs

    r_list = [60, 40, 30, 20, 10]   # listes de rangs profonds testées


    init = "random"
    

    model_tag = "DeepNMF_2W_deepness"
    # ------------------------- DATA -------------------------
    X, dataset = load_mnist(resize=(28, 28))

    # ------------------------- SHUFFLE REPRODUCTIBLE -------------------------
    seed = 112 # To have 7 as the reproduction image
    rng = np.random.default_rng(seed)

    perm = rng.permutation(X.shape[0])
    X = X[perm]   # Reproducible shuffle
    
    # ------------------------- OUTPUT DIR -------------------------
    out_dir = os.path.join("./sweeps", model_tag)
    os.makedirs(out_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(out_dir, exist_ok=True)

    results = []

    snapshot_every  = 500
    

    # ------------------------- SWEEP -------------------------
    total_runs = len(r_list)

    run_idx = 0

    for x_n in x_train_list:
        X_train = X[:x_n, :]

        D = X_train.shape[0] 

        for epochs in epochs_list:
            for i in range(1, len(r_list) + 1):

                
                
                r_l = [D] + r_list[-i:]
                print(f"\n-- Deep NMF with r_list={r_l}, epochs={epochs} --")

                run_idx += 1
                r_tag = "x".join(map(str, r_list[-i:]))
                run_name = f"x{x_n}_e{epochs}_r{r_tag}_init{init}_{run_time}"
                print(f"\n[{run_idx}/{total_runs}] {run_name}")

                run_dir = os.path.join(out_dir, run_name)
                snap_dir = os.path.join(run_dir, "snapshots")


                # ----------------- TRAIN -----------------
                start = time.perf_counter()
                Ws, H = Deep_NMF_2W_toN(
                    X_train,
                    init=init,
                    end="matrix",
                    epochs=epochs,
                    seed=seed,
                    save_dir=snap_dir,
                    snapshot_every=snapshot_every,
                    r_list=r_l
                )
                seconds = time.perf_counter() - start

                errorsGD_s, rankGD, nuclearrankGD, SVGD1, SVGD2, epochs_metrics_s = compute_metrics_from_snapshots_toN(
                    snap_dir, X_train
                )

                # ------------------------- VISUALISATIONS -------------------------
                W_prod = prod_Ws(Ws, nonneg=True)

                plot_nmf_results(W_prod, H, errorsGD_s, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X, epochs_metrics=epochs_metrics_s,limit=epochs)
                plot_H_signatures(H, "deep", image_shape=(28, 28), n_show=H.shape[0])
                plot_mnist_reconstruction_toN(X_train, Ws, H, index=0, image_shape=(28, 28))

                # ----- Signatures over time -----
                sig_dir = os.path.join(run_dir, "figures", "signatures_over_time_H")
                os.makedirs(sig_dir, exist_ok=True)

                snapshots = sorted(glob.glob(os.path.join(glob.escape(snap_dir), "snapshot_*.pt")))
                keep = np.linspace(0, len(snapshots)-1, 100, dtype=int)
                keep = np.unique(keep)

                for i in keep:

                    pt_path = snapshots[i]
                    ckpt = torch.load(pt_path, map_location="cpu")


                    epoch_k = ckpt["epoch"]
                    H_k = ckpt["H_out"]

                    # --- H_out ---
                    plot_H_signatures(H_k, title=f"H_out (epoch={epoch_k})", image_shape=(28, 28), n_show=H_k.shape[0])
                    plt.savefig(os.path.join(sig_dir, f"signatures_epoch_{epoch_k:06d}.png"),
                                dpi=200, bbox_inches="tight")
                    plt.close()

                # ------------------------- SAUVEGARDE FIGURES -------------------------
                fig_dir = save_all_figures(
                    base_dir=os.path.join(run_dir, "figures")
                )
                plt.close("all")

                final_error = float(errorsGD_s[-1]) if len(errorsGD_s) else np.nan

                logs = {
                    "errorsGD": errorsGD_s,
                    "rankGD": rankGD,
                    "nuclearrankGD": nuclearrankGD,
                    "SVGD1": SVGD1,
                    "SVGD2": SVGD2
                }
                meta = {
                    "x_train": x_n,
                    "epochs": epochs,
                    "r_list": r_list, 
                    "init": init,
                    "seconds": seconds,
                    "final_error": final_error
                }

                saved_path = save_run_toN(run_dir, "factors_and_logs",Ws, H, logs, meta)

                results.append({
                    "run": run_name,
                    "x_train": x_n,
                    "epochs": epochs,
                    "r_list": r_list,
                    "depth": len(r_list) - 1,
                    "seconds": seconds,
                    "final_error": final_error,
                    "npz": saved_path
                })

                # Cleanup snapshot directory
                shutil.rmtree(snap_dir)

    # ------------------------- SUMMARY + OPTIMAL -------------------------
    df = pd.DataFrame(results).sort_values(by="final_error", ascending=True)

    # Optional: add helpful derived columns
    if "r_list" in df.columns:
        df["depth"] = df["r_list"].apply(lambda x: len(x) - 1 if isinstance(x, (list, tuple)) else np.nan)
        df["r_last"] = df["r_list"].apply(lambda x: x[-1] if isinstance(x, (list, tuple)) and len(x) else np.nan)

    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()