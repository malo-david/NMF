import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import exp_effective_rank_torch, nuclear_over_operator_norm_torch, cosine_separation_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Deep_NMF_2W(A, r1, r2, init='random', end='matrix', epochs=6000, seed=None):
    """
    Deep NMF avec 2 couches W : A ≈ W1 @ W2 @ H
    """
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
        print(H)
        print(W1)
    elif init == 'ssvd':
        W1, W2, H = deep_nmf_init_torch(A_tensor, r1, r2, device=device)

    optimizer = torch.optim.Adam([W1, W2, H], lr=lr)

    errorsGD, nuclearrankGD, fullerrorsGD, rankGD = [], [], [], []
    SVGD1, SVGD2, SVGD3 = [], [], []
    coef1, coef2, coef3, coef4 = [], [], [], []
    SBS2 = []
    SBS7a = []
    SBS40a = []
    SBS40b = []
    SBS13 = []
    fro_Y = torch.norm(A_tensor, p='fro') ** 2 

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
            rankGD.append(exp_effective_rank_torch(WH))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH))
            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"Époque {epoch},  erreur relative : {rel_error:.4f}, norme A : {torch.norm(A_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")

            singular_values = torch.linalg.svdvals(WH_detached).detach().cpu().numpy()
            SVGD1.append(singular_values[0])
            SVGD2.append(singular_values[1])
            coef1.append(W1[0, 0].cpu().numpy())
            coef2.append(W2[0, 0].cpu().numpy())
            coef3.append(H[0, 0].cpu().numpy())
            coef4.append(H[1, 1].cpu().numpy())

    if end == 'lists':
        return W2, H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H
    elif end == 'all':
        return W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2


def NMF_for_r_comparison(A, r, init, end, epochs):
    """
    NMF classique avec descente de gradient (Adam)
    """
    m, n = np.shape(A)
    lr = 1e-2

    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    
    # Initialisation des facteurs pour descente de gradient
    if init == 'eye':
        W = torch.eye(m, r, device = device)
        W.requires_grad_()
        H = torch.eye(r, n, device = device)
        H.requires_grad_()  
    elif init == 'random': 
        W = torch.rand(m,r, device = device, requires_grad=True)
        H = torch.rand(r,n, device = device, requires_grad=True)
    elif init == 'ssvd':
        W, H = nndsvd_init_torch(A, r, device=device)

    optimizer = torch.optim.Adam([W, H], lr=lr)
    errors = []
    Y_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    epsilon = 1e-10
    errorsGD = []
    nuclearrankGD = []
    fullerrorsGD = []
    SVGD1 = []
    SVGD2 = []
    SVGD3 = []
    SVGD4 = []
    coef1 = []
    coef2 = []
    coef3 = []
    coef4 = []
    rankGD = []
    fro_Y = torch.norm(Y_tensor, p='fro') ** 2 

    # Training
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        WH = torch.matmul(F.relu(W), F.relu(H))
        loss = torch.norm(Y_tensor - WH, p='fro')**2 + 0 * torch.sum(torch.abs(H))

        loss.backward()
        optimizer.step()

        # Calcul erreur relative (detach pour ne pas traquer le gradient)
        with torch.no_grad():
            W.clamp_(min=0)
            H.clamp_(min=0)
            WH_detached = torch.matmul(F.relu(W), F.relu(H))
            rel_error = (torch.norm(Y_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(Y_tensor - WH_detached, p='fro') ** 2).item()

            errorsGD.append(rel_error)
            fullerrorsGD.append(error)
            rankGD.append(exp_effective_rank_torch(WH))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH))
            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(torch.norm(WH_detached))
                print(f"Époque {epoch}, erreur relative : {rel_error:.4f}, norme A : {torch.norm(Y_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
                print(torch.all(W >= 0))
                print(torch.all(H >= 0))
            if torch.all(W>=0) == False:
                print("ERROR ON W")
            if torch.all(H>=0) == False:
                print("ERROR ON H")

            singular_values = torch.linalg.svdvals(W@H)
            s_np = singular_values.detach().cpu().numpy()

            SVGD1.append(s_np[0])
            SVGD2.append(s_np[1])

    if end == 'lists':
        return H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W,H
    elif end == 'all':
        return W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2


def NMF_for_r_comparison_MU(A, r, init, end, epochs):
    """
    NMF avec Multiplicative Updates (Lee & Seung 2001)
    """
    m, n = np.shape(A)

    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    
    # Initialisation des facteurs
    if init == 'eye':
        W = torch.eye(m, r, device=device)
        H = torch.eye(r, n, device=device)
    elif init == 'random':
        W = torch.rand(m, r, device=device)
        H = torch.rand(r, n, device=device)
    elif init == 'ssvd':
        W, H = nndsvd_init_torch(A, r, device=device)

    # Données
    Y_tensor = torch.tensor(A, dtype=torch.float32).to(device)  
    fro_Y = torch.norm(Y_tensor, p='fro') ** 2 

    # Logs
    epsilon = 1e-10
    errorsGD = []
    nuclearrankGD = []
    fullerrorsGD = []
    SVGD1 = []
    SVGD2 = []
    rankGD = []

    # Training
    for epoch in tqdm(range(epochs)):

        # Multiplicative Updates (Lee & Seung 2001)
        # H <- H * (W^T Y) / (W^T W H)
        WH = W @ H
        numerator_H = W.T @ Y_tensor
        denominator_H = (W.T @ WH) + epsilon
        H = H * (numerator_H / denominator_H)

        # W <- W * (Y H^T) / (W H H^T)
        WH = W @ H
        numerator_W = Y_tensor @ H.T
        denominator_W = (W @ (H @ H.T)) + epsilon
        W = W * (numerator_W / denominator_W)

        # Logs
        with torch.no_grad():
            WH_detached = W @ H
            rel_error = (torch.norm(Y_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(Y_tensor - WH_detached, p='fro') ** 2).item()

            errorsGD.append(rel_error)
            fullerrorsGD.append(error)
            rankGD.append(exp_effective_rank_torch(WH_detached))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH_detached))

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(torch.norm(WH_detached))
                print(f"Époque {epoch}, erreur relative : {rel_error:.4f}, norme A : {torch.norm(Y_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
                print(torch.all(W >= 0))
                print(torch.all(H >= 0))

            singular_values = torch.linalg.svdvals(WH_detached)
            s_np = singular_values.detach().cpu().numpy()
            SVGD1.append(s_np[0])
            if len(s_np) > 1:
                SVGD2.append(s_np[1])

    # Outputs
    if end == 'lists':
        return H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W, H
    elif end == 'all':
        return W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2
