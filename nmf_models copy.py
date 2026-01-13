import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import exp_effective_rank_torch, nuclear_over_operator_norm_torch, cosine_separation_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Deep_NMF_2W(A, r1, r2, init='random', end='matrix', epochs=6000, seed=None, save_dir=None, snapshot_every=None):
    """
    Deep NMF avec 2 couches W : A ≈ W1 @ W2 @ H
    """

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m, n = np.shape(A)


    lr = 1e-2

    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    # Initialisation
    if init == 'eye':
        W1 = torch.eye(m, r1, device=device, requires_grad=True)
        W2 = torch.eye(r1, r2, device=device, requires_grad=True)
        H  = torch.eye(r2, n, device=device, requires_grad=True)
    elif init == 'random':
        W1_init = torch.empty((m, r1), device=device).uniform_(0, 1)
        W1 = W1_init.clone().detach().requires_grad_(True)  
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        H  = torch.rand((r2, n), device=device, requires_grad=True)
        #print(H)
        #print(W1)
    elif init == 'ssvd':
        W1, W2, H = deep_nmf_init_torch(A_tensor, r1, r2, device=device)

    optimizer = torch.optim.Adam([W1, W2, H], lr=lr)

    errorsGD, nuclearrankGD, fullerrorsGD, rankGD = [], [], [], []
    SVGD1, SVGD2 = [], []

    epochs_metrics = []

    fro_Y = torch.norm(A_tensor, p='fro') ** 2 

    metric_every = max(1, epochs // 100)
    
    if snapshot_every is None:
        snapshot_every = 0  # 0 = désactivé

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Reconstruction via W1 * W2 * H
        WH = F.relu(W1) @ F.relu(W2) @ F.relu(H)
        l1_lambda = 0.3
        l1_cos = 0.00
        loss = torch.norm(A_tensor - WH, p='fro')**2 + l1_lambda * torch.norm(W1, p = 1) + l1_cos * cosine_separation_loss(H)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Clamp (optionnel, au cas où relu ne suffit pas)
            W1.clamp_(min=1e-2)
            W2.clamp_(min=1e-2)
            H.clamp_(min=1e-2)
            WH_detached = F.relu(W1) @ F.relu(W2) @ F.relu(H)
            rel_error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2).item()

            errorsGD.append(rel_error)
            fullerrorsGD.append(error)

            if epoch % metric_every == 0 or epoch == epochs - 1:
                epochs_metrics.append(epoch)
                rankGD.append(exp_effective_rank_torch(WH_detached))
                nuclearrankGD.append(nuclear_over_operator_norm_torch(WH_detached))
                if epoch % 1000 == 0 or epoch == epochs - 1:
                    print(f"Époque {epoch},  erreur relative : {rel_error:.4f}, norme A : {torch.norm(A_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")

                s = torch.linalg.svdvals(WH_detached)
                SVGD1.append(s[0].item())
                SVGD2.append(s[1].item() if s.numel() > 1 else 0.0)

            # ----- Snapshot de H pour plot signatures -----
            if save_dir is not None and snapshot_every and (epoch % snapshot_every == 0 or epoch == epochs - 1):
                # on sauve H en CPU pour éviter GPU->CPU plus tard
                torch.save(
                    {"epoch": epoch, "H": H.detach().cpu()},
                    os.path.join(save_dir, f"H_epoch_{epoch:06d}.pt")
                )

    if end == 'lists':
        return W2, H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H
    elif end == 'all':
        return W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, epochs_metrics


def Deep_NMF_Article(
    A,
    r1,
    r2,
    init="random",
    end="matrix",
    epochs=6000,
    seed=None,
    save_dir=None,
    snapshot_every=None,
    lr=1e-2,
    lambda_l1=1,
    lambda_l2=1
):
    """
    Deep NMF Article (2W + 2H)
    Optimisation :
        ||A - ReLU(W1) @ H_mid||_F^2
      + ||H_mid - W2 @ H_out||_F^2
    """

    # ---------- Seed ----------
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()

    m, n = A.shape
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)

    # ---------- Init ----------
    if init == "eye":
        W1 = torch.eye(m, r1, device=device, requires_grad=True)
        W2 = torch.eye(r1, r2, device=device, requires_grad=True)
        H_mid = torch.eye(r1, n, device=device, requires_grad=True)
        H_out = torch.eye(r2, n, device=device, requires_grad=True)

    elif init == "random":
        W1 = torch.rand((m, r1), device=device, requires_grad=True)
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        H_mid = torch.rand((r1, n), device=device, requires_grad=True)
        H_out = torch.rand((r2, n), device=device, requires_grad=True)

    elif init == "ssvd":
        W1, W2, H_out = deep_nmf_init_torch(A_tensor, r1, r2, device=device)
        H_mid = torch.rand((r1, n), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([W1, W2, H_mid, H_out], lr=lr)

    # ---------- Logs ----------
    errorsGD, fullerrorsGD = [], []
    rankGD, nuclearrankGD = [], []
    SVGD1, SVGD2 = [], []
    epochs_metrics = []

    fro_Y = torch.norm(A_tensor, p="fro") ** 2
    metric_every = max(1, epochs // 300)

    if snapshot_every is None:
        snapshot_every = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ---------- Training ----------
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        loss = (
            lambda_l1 * torch.norm(A_tensor - F.relu(W1) @ H_mid, p="fro") ** 2
            + lambda_l2 * torch.norm(H_mid - W2 @ H_out, p="fro") ** 2
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % metric_every == 0 or epoch == epochs - 1:
                W1.clamp_(min=0.0)
                W2.clamp_(min=0.0)
                H_mid.clamp_(min=0.0)
                H_out.clamp_(min=0.0)

                WH = F.relu(W1) @ F.relu(W2) @ F.relu(H_out)

                err = torch.norm(A_tensor - WH, p="fro") ** 2
                rel_err = (err / fro_Y).item()

                errorsGD.append(rel_err)
                fullerrorsGD.append(err.item())

                epochs_metrics.append(epoch)
                rankGD.append(exp_effective_rank_torch(WH))
                nuclearrankGD.append(nuclear_over_operator_norm_torch(WH))

                s = torch.linalg.svdvals(WH.detach().cpu())
                SVGD1.append(s[0].item())
                SVGD2.append(s[1].item() if s.numel() > 1 else 0.0)

        # ---------- Snapshots ----------
        if (
            save_dir is not None
            and snapshot_every > 0
            and (epoch % snapshot_every == 0 or epoch == epochs - 1)
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "W1": W1.detach().cpu(),
                    "W2": W2.detach().cpu(),
                    "H_mid": H_mid.detach().cpu(),
                    "H_out": H_out.detach().cpu(),
                },
                os.path.join(save_dir, f"snapshot_{epoch:06d}.pt"),
            )

    # ---------- Output ----------
    if end == "matrix":
        return W1, W2, H_mid, H_out
    elif end == "lists":
        return errorsGD, rankGD, SVGD1, SVGD2
    elif end == "all":
        return (
            W1,
            W2,
            H_mid,
            H_out,
            errorsGD,
            rankGD,
            nuclearrankGD,
            SVGD1,
            SVGD2,
            epochs_metrics
        )
