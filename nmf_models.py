import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from metrics import exp_effective_rank_torch, nuclear_over_operator_norm_torch, cosine_separation_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Deep_NMF_2W(
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
    l1_lambda=0.3,
    l1_cos=0.0
):
    """
    Deep NMF with two W layers: A ≈ W1 @ W2 @ H.
    """

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m, n = np.shape(A)



    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
        m, n = A.shape
    
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


    
    if snapshot_every is None:
        snapshot_every = 0  # 0 = désactivé

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Reconstruction via W1 * W2 * H
        WH = F.relu(W1) @ F.relu(W2) @ F.relu(H)
        loss = torch.norm(A_tensor - WH, p='fro')**2 + l1_lambda * torch.norm(W1, p = 1) + l1_cos * cosine_separation_loss(H)

        #Penalty term encouraging orthogonality between rows of H.

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            W1.clamp_(min=0.0)
            W2.clamp_(min=0.0)
            H.clamp_(min=0.0)

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
                    "H_out": H.detach().cpu(),
                },
                os.path.join(save_dir, f"snapshot_{epoch:06d}.pt"),
            )

    if end == 'lists':
        return W2, H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H


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
        ||A - ReLU(W1) @ ReLU(H_mid)||_F^2
      + ||ReLU(H_mid) - ReLU(W2) @ ReLU(H_out)||_F^2
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
            lambda_l1 * torch.norm(A_tensor - F.relu(W1) @ F.relu(H_mid), p="fro") ** 2
            + lambda_l2 * torch.norm(F.relu(H_mid) - F.relu(W2) @ F.relu(H_out), p="fro") ** 2
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            W1.clamp_(min=0.0)
            W2.clamp_(min=0.0)
            H_mid.clamp_(min=0.0)
            H_out.clamp_(min=0.0)

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
    elif end == "light":
        return W1, W2, H_mid, H_out, errorsGD, epochs_metrics


def Deep_NMF_2W_toN(
    A,
    init="random",
    end="matrix",
    epochs=6000,
    seed=None,
    save_dir=None,
    snapshot_every=None,
    lr=1e-2,
    r_list=[]
):
    """
    Deep NMF with two W layers: A ≈ W1 @ W2 @ H
    """

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m, n = np.shape(A)



    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
        m, n = A.shape
    
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    if init == 'random':
        Ws = nn.ParameterList()
        for i in range(len(r_list) - 1):
            Ws.append(
                nn.Parameter(torch.rand((r_list[i], r_list[i+1]), device=device))
            )
        H = nn.Parameter(torch.rand((r_list[-1], n), device=device))
        #print(H)
        #print(W1)

    optimizer = torch.optim.Adam(
        list(Ws.parameters()) + [H],
        lr=lr
    )

    errorsGD, nuclearrankGD, fullerrorsGD, rankGD = [], [], [], []
    SVGD1, SVGD2 = [], []


    
    if snapshot_every is None:
        snapshot_every = 0  # 0 = désactivé

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Reconstruction via W1 * W2 * H
        WH = F.relu(Ws[0])
        for W in Ws[1:]:
            WH = WH @ F.relu(W)
        WH = WH @ F.relu(H)

        loss = torch.norm(A_tensor - WH, p='fro')**2 

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for W in Ws:
                W.clamp_(min=0.0)
            H.clamp_(min=0.0)

        # ---------- Snapshots ----------
        if (
            save_dir is not None
            and snapshot_every > 0
            and (epoch % snapshot_every == 0 or epoch == epochs - 1)
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "Ws": [W.detach().cpu() for W in Ws],
                    "H_out": H.detach().cpu()
                },
                os.path.join(save_dir, f"snapshot_{epoch:06d}.pt"),
            )

    if end == 'matrix':
        return Ws, H