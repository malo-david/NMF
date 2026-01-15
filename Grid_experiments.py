import os
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F

# Tes modules
from data_loader import load_mnist
from nmf_models import Deep_NMF_2W

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
    """Torch Tensor (CPU/GPU) -> numpy, sinon retourne tel quel."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x

def save_run(out_dir, run_name, W1, W2, H, logs, meta):
    """
    Sauve:
      - facteurs W1, W2, H
      - logs (errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2)
      - meta (params, temps, etc.)
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}.npz")

    np.savez_compressed(
        path,
        W1=to_numpy(W1),
        W2=to_numpy(W2),
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
    Sauvegarde toutes les figures matplotlib ouvertes
    dans un dossier horodaté.
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

def pick_optimal(df, rel_tol=0.01):
    """
    'Optimal' = parmi les runs dont l'erreur finale est à <= (1+rel_tol)*best,
    on choisit celui avec le plus petit temps, puis le plus petit modèle.
    """
    best = df["final_error"].min()
    candidates = df[df["final_error"] <= best * (1.0 + rel_tol)].copy()

    # Tie-breakers: temps puis complexité (r1*r2 puis r2)
    candidates = candidates.sort_values(
        by=["seconds", "r1xr2", "r2", "r1", "epochs", "x_train"],
        ascending=[True, True, True, True, True, True]
    )
    return candidates.iloc[0], best

def compute_metrics_from_snapshots(
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

        W1 = ckpt["W1"]
        W2 = ckpt["W2"]
        H_out = ckpt["H_out"]

        # Non-négativité NMF
        W1 = F.relu(W1)
        W2 = F.relu(W2)
        H_out = F.relu(H_out)

        WH = W1 @ W2 @ H_out

        # erreur relative
        err = torch.norm(A - WH, p="fro") ** 2
        rel_err = (err / fro_Y).item()

        # métriques spectrales
        eff_rank = exp_effective_rank_torch(WH, eps=eps)
        nuclear_rank = nuclear_over_operator_norm_torch(WH)

        # singular values (pour logging)
        s = torch.linalg.svdvals(WH)
        sv1 = s[0].item()
        sv2 = s[1].item() if s.numel() > 1 else 0.0

        # stockage
        epochs_metrics.append(epoch)
        errorsGD.append(rel_err)
        rankGD.append(eff_rank)
        nuclearrankGD.append(nuclear_rank)
        SVGD1.append(sv1)
        SVGD2.append(sv2)

    return errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, epochs_metrics


def main():
    # ------------------------- CONFIG SWEEP -------------------------
    # Tu peux ajuster ces listes librement.
    x_train_list = [2000]      # tailles de X_train testées
    epochs_list  = [40000]      # epochs testées

    r1_list = [20, 40]
    r2_list = [10, 20, 30]

    l1_lambda_list = [0.0, 0.03, 0.1]
    l1_cos_list = [0.0, 1e-4, 1e-2]

    init = "random"
    rel_tol_for_optimal = 0.01            # 1% de la meilleure erreur = "aussi bon"

    model_tag = "DeepNMF_2W"
    # ------------------------- DATA -------------------------
    X, dataset = load_mnist(resize=(28, 28))

    # ------------------------- SHUFFLE REPRODUCTIBLE -------------------------
    seed = 112 # Pour avoir un 7 en reconstruction
    rng = np.random.default_rng(seed)

    perm = rng.permutation(X.shape[0])
    X = X[perm]   # on remplace X par sa version shuffled

    # ------------------------- OUTPUT DIR -------------------------
    out_dir = os.path.join("./sweeps", model_tag)
    os.makedirs(out_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(out_dir, exist_ok=True)

    results = []

    snapshot_every  = 200
    

    # ------------------------- SWEEP -------------------------
    total_runs = 0
    for r1 in r1_list:
        for r2 in r2_list:
            if r2 <= r1:
                total_runs += 1
    total_runs *= len(x_train_list) * len(epochs_list) * len(l1_lambda_list) * len(l1_cos_list)

    run_idx = 0

    for x_n in x_train_list:
        X_train = X[:x_n, :]

        for epochs in epochs_list:
            for r1 in r1_list:
                for r2 in r2_list:
                    for l1_lambda in l1_lambda_list:
                        for l1_cos in l1_cos_list:
                            if r2 > r1:
                                # souvent inutile d'avoir r2 > r1 dans W1@W2
                                continue

                            run_idx += 1
                            run_name = f"x{x_n}_e{epochs}_r1{r1}_r2{r2}__l1_lambda{l1_lambda}_l1_cos{l1_cos}_init{init}_{run_time}"
                            print(f"\n[{run_idx}/{total_runs}] {run_name}")

                            run_dir = os.path.join(out_dir, run_name)
                            snap_dir = os.path.join(run_dir, "snapshots")


                            # ----------------- TRAIN -----------------
                            start = time.perf_counter()
                            W1, W2, H = Deep_NMF_2W(
                                X_train,
                                r1=r1,
                                r2=r2,
                                init=init,
                                end="matrix",
                                epochs=epochs,
                                seed=seed,
                                save_dir=snap_dir,
                                snapshot_every=snapshot_every,
                                l1_lambda=l1_lambda,
                                l1_cos=l1_cos
                            )
                            seconds = time.perf_counter() - start

                            errorsGD_s, rankGD, nuclearrankGD, SVGD1, SVGD2, epochs_metrics_s = compute_metrics_from_snapshots(snap_dir, X_train)

                            # ------------------------- VISUALISATIONS -------------------------
                            plot_nmf_results(W1@W2, H, errorsGD_s, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X, epochs_metrics=epochs_metrics_s)
                            plot_H_signatures(H, "deep", image_shape=(28, 28), n_show=H.shape[0])
                            plot_mnist_reconstruction(X, W1, W2, H, index=0, image_shape=(28, 28))

                            # ----- Signatures over time -----
                            sig_dir = os.path.join(run_dir, "figures", "signatures_over_time_H")
                            os.makedirs(sig_dir, exist_ok=True)

                            snapshots = sorted(glob.glob(os.path.join(snap_dir, "snapshot_*.pt")))
                            keep = np.linspace(0, len(snapshots)-1, 100, dtype=int)
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
                                "r1": r1,
                                "r2": r2,
                                "init": init,
                                "seconds": seconds,
                                "final_error": final_error
                            }

                            saved_path = save_run(run_dir, "factors_and_logs", W1, W2, H, logs, meta)

                            results.append({
                                "run": run_name,
                                "x_train": x_n,
                                "epochs": epochs,
                                "r1": r1,
                                "r2": r2,
                                "r1xr2": r1 * r2,
                                "seconds": seconds,
                                "final_error": final_error,
                                "npz": saved_path
                            })

                            # Nettoyage snapshots
                            shutil.rmtree(snap_dir)

    # ------------------------- SUMMARY + OPTIMAL -------------------------
    df = pd.DataFrame(results).sort_values(by="final_error", ascending=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False)

    chosen, best = pick_optimal(df, rel_tol=rel_tol_for_optimal)

    print("\n==================== RÉSULTATS ====================")
    print(f"Meilleure erreur finale: {best:.6f}")
    print(f"Choix 'optimal' (<= {100*rel_tol_for_optimal:.1f}% du best, + plus rapide):")
    print(chosen[["run", "x_train", "epochs", "r1", "r2", "seconds", "final_error", "npz"]])

    print(f"\nRésumé sauvegardé: {csv_path}")
    print(f"Dossier sweep: {out_dir}")

if __name__ == "__main__":
    main()
