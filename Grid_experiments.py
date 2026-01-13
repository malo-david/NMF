import os
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

# Tes modules
from data_loader import load_mnist
from nmf_models import Deep_NMF_2W
from nmf_models import Deep_NMF_2W_Malopetitchou

from visualizations import (
    plot_nmf_results, 
    plot_H_signatures, 
    plot_mnist_reconstruction,
    plot_mnist_reconstruction_nmf
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

def main():
    # ------------------------- CONFIG SWEEP -------------------------
    # Tu peux ajuster ces listes librement.
    x_train_list = [2000]      # tailles de X_train testées
    epochs_list  = [80000]      # epochs testées

    r1_list = [20, 40, 60]
    r2_list = [5, 10, 20, 30]

    init = "random"
    rel_tol_for_optimal = 0.01            # 1% de la meilleure erreur = "aussi bon"

    model_tag = "DeepNMF_2W"
    # ------------------------- DATA -------------------------
    X, dataset = load_mnist(resize=(28, 28))

    # ------------------------- SHUFFLE REPRODUCTIBLE -------------------------
    seed = 65
    rng = np.random.default_rng(seed)

    perm = rng.permutation(X.shape[0])
    X = X[perm]   # on remplace X par sa version shuffled

    # ------------------------- OUTPUT DIR -------------------------
    out_dir = os.path.join("./sweeps", model_tag)
    os.makedirs(out_dir, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(out_dir, exist_ok=True)

    results = []

    patience = 1000
    

    # ------------------------- SWEEP -------------------------
    total_runs = len(x_train_list) * len(epochs_list) * len(r1_list) * len(r2_list)
    run_idx = 0

    for x_n in x_train_list:
        X_train = X[:x_n, :]

        for epochs in epochs_list:
            for r1 in r1_list:
                for r2 in r2_list:
                    if r2 > r1:
                        # souvent inutile d'avoir r2 > r1 dans W1@W2
                        continue

                    run_idx += 1
                    run_name = f"x{x_n}_e{epochs}_r1{r1}_r2{r2}_init{init}_{run_time}"
                    print(f"\n[{run_idx}/{total_runs}] {run_name}")

                    run_dir = os.path.join(out_dir, run_name)
                    snap_dir = os.path.join(run_dir, "snapshots")

                    # ----------------- TRAIN -----------------
                    start = time.perf_counter()
                    W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 , epochs_metrics = Deep_NMF_2W(
                        X_train,
                        r1=r1,
                        r2=r2,
                        init=init,
                        end="all",
                        epochs=epochs,
                        seed=seed,
                        save_dir=snap_dir,
                        snapshot_every=patience
                    )
                    seconds = time.perf_counter() - start

                    final_error = float(errorsGD[-1]) if len(errorsGD) else np.nan

                    # ------------------------- VISUALISATIONS -------------------------
                    plot_nmf_results(W1@W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X, epochs_metrics=epochs_metrics)
                    plot_H_signatures(H, "deep", image_shape=(28, 28), n_show=9)
                    plot_mnist_reconstruction(X, W1, W2, H, index=0, image_shape=(28, 28))

                    # ----- Signatures over time -----
                    sig_dir = os.path.join(run_dir, "figures", "signatures_over_time")
                    os.makedirs(sig_dir, exist_ok=True)

                    for pt_path in sorted(glob.glob(os.path.join(snap_dir, "H_epoch_*.pt"))):
                        ckpt = torch.load(pt_path, map_location="cpu")
                        epoch_k = ckpt["epoch"]
                        H_k = ckpt["H"]

                        plot_H_signatures(H_k, title=f" (epoch={epoch_k})", image_shape=(28, 28), n_show=9)
                        plt.savefig(os.path.join(sig_dir, f"signatures_epoch_{epoch_k:06d}.png"),
                                    dpi=200, bbox_inches="tight")
                        plt.close()
                    # ------------------------- SAUVEGARDE FIGURES -------------------------
                    fig_dir = save_all_figures(
                        base_dir=os.path.join(run_dir, "figures")
                    )

                    plt.close("all")

                    logs = {
                        "errorsGD": errorsGD,
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
