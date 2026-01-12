import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import gzip
import pyreadr
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms
from scipy.stats import mode
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from collections import Counter
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import os


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42):
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

# Exemple d'utilisation
set_seed(123)





# #-------------------------DATASET-------------------------

# # Transformations : redimensionnement + conversion en tenseur
# # (tu peux adapter pour ton pipeline NMF)
# # transform = transforms.Compose([
# #     transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
# #     transforms.ToTensor()
# # ])

# # Charger le dataset
# dataset = datasets.Flowers102(
#     root="./data",
#     split="train",
#     download=True,
#     transform=transform
# )

# # Transformations : redimensionnement + conversion en tenseur
# # (tu peux adapter pour ton pipeline NMF)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
#     transforms.ToTensor()
# ])

# # Charger le dataset
# dataset = datasets.Flowers102(
#     root="./data",
#     split="train",
#     download=True,
#     transform=transform
# )

# # Transformations : redimensionnement + conversion en tenseur
# # (tu peux adapter pour ton pipeline NMF)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
#     transforms.ToTensor()
# ])

# # Charger le dataset
# dataset = datasets.Flowers102(
#     root="./data",
#     split="train",
#     download=True,
#     transform=transform
# )



# # Convertir en matrice numpy
# X_list = []

# for img, _ in dataset:
#     arr = img.numpy().reshape(-1)   # aplatissement
#     X_list.append(arr)

# X = np.stack(X_list)   # Matrice finale (N, D)

import numpy as np
from torchvision import datasets, transforms

# ------------------------- DATASET -------------------------

# Transformations : redimensionnement + conversion en tenseur
# (adapté pour NMF)
transform = transforms.Compose([
    transforms.Resize((28, 28)),   # upscale MNIST (28x28 → 64x64)
    transforms.ToTensor()
])

# Charger le dataset MNIST
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# ------------------------- MATRICE POUR NMF -------------------------

X_list = []

for img, _ in dataset:
    arr = img.numpy().reshape(-1)   # aplatissement
    X_list.append(arr)

X = np.stack(X_list)   # Matrice finale (N, D)


#----------------FONCTION SECONDAIRE-----------------------------
def exp_effective_rank_torch(A, eps=1e-12):
    # A : matrice 2D tensor
    # calcul des valeurs singulières
    S = torch.linalg.svdvals(A)
    S = S[S > eps]  # éviter les très petites valeurs

    p = S / S.sum()  # distribution de probabilité normalisée
    entropy = -torch.sum(p * torch.log(p))
    effective_rank = torch.exp(entropy)
    return effective_rank.cpu().numpy()


def nuclear_over_operator_norm_torch(A):
    s = torch.linalg.svdvals(A)
    nuclear_norm = s.sum()
    operator_norm = s.max()
    return (nuclear_norm / operator_norm).item()

def cosine_separation_loss(H, eps=1e-8):
    Hn = H / (torch.norm(H, dim=1, keepdim=True) + eps)
    G = Hn @ Hn.T
    I = torch.eye(G.size(0), device=H.device)
    return - torch.norm(G - I, p='fro')**2





def plot_nmf_results(W, H, titre, errorsGD, rankGD, nuclearrankGD,
                     SVGD1, SVGD2, image_shape=(64, 64), X=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # --- Convertir torch -> numpy si nécessaire
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    if X is not None and isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # ==========================================================
    # FIGURE 1 : Heatmaps W et H
    # ==========================================================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Heatmap W, epoch = " + titre)
    plt.imshow(W, aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Heatmap H, epoch = " + titre)
    plt.imshow(H, aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.tight_layout()

    # ==========================================================
    # FIGURE 2 : Courbes de suivi
    # ==========================================================
    plt.figure(figsize=(12, 8))

    plots = [errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2]
    titles = ["errorsGD", "rankGD", "nuclearrankGD", "SVGD1", "SVGD2"]

    for i, (t, y) in enumerate(zip(titles, plots)):
        plt.subplot(3, 2, i + 1)
        plt.plot(y)
        plt.title(t)
        plt.grid(True)

    plt.tight_layout()

    # # ==========================================================
    # # FIGURE 3 : Composantes NMF (lignes de H) → images MNIST
    # # ==========================================================
    # num_components = H.shape[0]   # doit être 10
    # n_cols = 5
    # n_rows = int(np.ceil(num_components / n_cols))

    # plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))

    # for i in range(num_components):
    #     comp = H[i]

    #     img = comp.reshape(image_shape)
    #     img = np.clip(img, 0, 1)

    #     plt.subplot(n_rows, n_cols, i + 1)
    #     plt.imshow(img, cmap="gray")
    #     plt.axis("off")
    #     plt.title(f"Composante H[{i}]", fontsize=9)

    # plt.suptitle("Composantes NMF (H ∈ ℝ^{10×D}) – MNIST", fontsize=14)
    # plt.tight_layout()


    # # ==========================================================
    # # FIGURE 4 : Images originales MNIST (X)
    # # ==========================================================
    # if X is not None:
    #     num_X = min(25, X.shape[0])
    #     n_cols = 5
    #     n_rows = int(np.ceil(num_X / n_cols))

    #     plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))

    #     for i in range(num_X):
    #         img = X[i].reshape(image_shape)
    #         img = np.clip(img, 0, 1)

    #         plt.subplot(n_rows, n_cols, i + 1)
    #         plt.imshow(img, cmap="gray")
    #         plt.axis("off")
    #         plt.title(f"X[{i}]", fontsize=8)

    #     plt.suptitle("Images originales MNIST", fontsize=14)
    #     plt.tight_layout()




def plot_H_signatures(H,title, image_shape=(28, 28), n_show=5):
    """
    Affiche n_show signatures (lignes) de H sous forme d'images MNIST
    H : (r2, D)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    n_show = min(n_show, H.shape[0])

    plt.figure(figsize=(2.5 * n_show, 3))

    for i in range(n_show):
        img = H[i].reshape(image_shape)
        img = np.clip(img, 0, 1)

        plt.subplot(1, n_show, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"H[{i}]")

    plt.suptitle("Signatures NMF (lignes de H)" + " epoch = " + title, fontsize=14)
    plt.tight_layout()

def plot_mnist_reconstruction(A, W1, W2, H, title, index=0, image_shape=(28, 28)):
    """
    Compare une image MNIST originale et sa reconstruction NMF
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # --- Conversion numpy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(W1, torch.Tensor):
        W1 = W1.detach().cpu().numpy()
    if isinstance(W2, torch.Tensor):
        W2 = W2.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    # --- Original
    x_true = A[index]

    # --- Reconstruction NMF
    W_eff = W1 @ W2                # (m × r2)
    coeffs = W_eff[index]          # (r2,)
    x_rec = coeffs @ H             # (D,)

    # --- Reshape
    img_true = x_true.reshape(image_shape)
    img_rec = x_rec.reshape(image_shape)
    img_err = np.abs(img_true - img_rec)

    # --- Normalisation sécurité
    img_rec = np.clip(img_rec, 0, 1)
    img_err = img_err / img_err.max() if img_err.max() > 0 else img_err

    # --- Plot
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img_true, cmap="gray")
    plt.title("Original MNIST")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_rec, cmap="gray")
    plt.title("Reconstruction NMF")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_err, cmap="hot")
    plt.title("|Erreur|")
    plt.axis("off")

    plt.suptitle(f"MNIST – reconstruction NMF (index={index})" + "epoch = " + title, fontsize=14)
    plt.tight_layout()
def plot_mnist_reconstruction_nmf(A, W, H, titre, index=0, image_shape=(28, 28)):
    """
    Compare une image MNIST originale et sa reconstruction avec une NMF simple
    A : (N, D) images originales
    W : (N, r) coefficients NMF
    H : (r, D) signatures NMF
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # --- Conversion numpy si nécessaire
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    # --- Image originale
    x_true = A[index]

    # --- Reconstruction NMF simple
    x_rec = W[index] @ H   # (D,)

    # --- Reshape
    img_true = x_true.reshape(image_shape)
    img_rec = np.clip(x_rec.reshape(image_shape), 0, 1)
    img_err = np.abs(img_true - img_rec)
    img_err = img_err / img_err.max() if img_err.max() > 0 else img_err

    # --- Plot
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img_true, cmap="gray")
    plt.title("Original MNIST")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_rec, cmap="gray")
    plt.title("Reconstruction NMF")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_err, cmap="hot")
    plt.title("|Erreur|")
    plt.axis("off")

    plt.suptitle(f"MNIST – reconstruction NMF simple (index={index})" + "epoch = " + titre, fontsize=14)
    plt.tight_layout()




#----------CODE EXPLICITE DEEP NMF-------------
def Deep_NMF_2W(A, r1, r2, init='random', end='matrix', epochs=6000, seed=None):
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
        # Sauvegarde de l'état RNG global
        #rng_state = torch.get_rng_state()
        
        # Fixe un seed local pour rendre l'init déterministe
        #torch.manual_seed(seed)

        # Initialisation déterministe
        W1_init = torch.empty((m, r1), device=device).uniform_(0, 1)
        W1 = W1_init.clone().detach().requires_grad_(True)  
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        H  = torch.rand((r1, n), device=device, requires_grad=True)
        H2  = torch.rand((r2, n), device=device, requires_grad=True)

        # Restaure l'état RNG global (ne pollue pas l'extérieur)
        #torch.set_rng_state(rng_state)
    elif init == 'ssvd':
        W1, W2, H = deep_nmf_init_torch(A_tensor, r1, r2, device=device)

    optimizer = torch.optim.Adam([W1, W2, H, H2], lr=lr)

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
        WH = F.relu(W1) @ F.relu(W2) @ F.relu(H2)
        l1_lambda = 0.0
        l1_cos = 0.00
        loss = torch.norm(A_tensor - F.relu(W1)@H, p='fro')**2 + torch.norm(H - W2@H2)**2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Clamp (optionnel, au cas où relu ne suffit pas)
            W1.clamp_(min=1e-2)
            W2.clamp_(min=1e-2)
            H.clamp_(min=1e-2)
            H2.clamp(min=1e-2)
            WH_detached = F.relu(W1) @ F.relu(W2) @ F.relu(H2)
            rel_error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2).item()

            errorsGD.append(rel_error)
            fullerrorsGD.append(error)
            rankGD.append(exp_effective_rank_torch(WH))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH))
            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"Époque {epoch},  erreur relative : {rel_error:.4f}, norme A : {torch.norm(A_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
                # print("Lignes correspondantes W1 : " + str(lignes_correspondantes(W1, D1)))
                # print("Lignes correspondantes W2 : " + str(lignes_correspondantes(W2, D2)))
                # print("Lignes correspondantes H : " + str(lignes_correspondantes(H, HD)))
                plot_H_signatures(H2,"epoch = " + str(epoch), image_shape=(28, 28), n_show=18)
                plot_H_signatures(H,"epoch = " + str(epoch), image_shape=(28, 28), n_show=18)
                plot_nmf_results(W1@W2, H, "epoch = " + str(epoch), errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X)
            singular_values = torch.linalg.svdvals(WH_detached).detach().cpu().numpy()
            SVGD1.append(singular_values[0])
            SVGD2.append(singular_values[1])
            coef1.append(W1[0, 0].cpu().numpy())
            coef2.append(W2[0, 0].cpu().numpy())
            coef3.append(H[0, 0].cpu().numpy())
            coef4.append(H[1, 1].cpu().numpy())
    # W2, H = normalize_H_scale_W_diag_torch(W2, H)
    # W1, W2 = normalize_H_scale_W_diag_torch(W1,W2)
    if end == 'lists':
        return W2, H2, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H2
    elif end == 'all':
        return W1, W2, H, H2, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2


def Deep_NMF_2W_Malopetitchou(A, r1, r2, init='random', end='matrix', epochs=6000, seed=None):
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
        # Sauvegarde de l'état RNG global
        #rng_state = torch.get_rng_state()
        
        # Fixe un seed local pour rendre l'init déterministe
        #torch.manual_seed(seed)

        # Initialisation déterministe
        W1_init = torch.empty((m, r1), device=device).uniform_(0, 1)
        W1 = W1_init.clone().detach().requires_grad_(True)  
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        H  = torch.rand((r2, n), device=device, requires_grad=True)

        # Restaure l'état RNG global (ne pollue pas l'extérieur)
        #torch.set_rng_state(rng_state)
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
            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"Époque {epoch},  erreur relative : {rel_error:.4f}, norme A : {torch.norm(A_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
                # print("Lignes correspondantes W1 : " + str(lignes_correspondantes(W1, D1)))
                # print("Lignes correspondantes W2 : " + str(lignes_correspondantes(W2, D2)))
                # print("Lignes correspondantes H : " + str(lignes_correspondantes(H, HD)))
                plot_H_signatures(H,"epoch = " + str(epoch) + "rank = " + str(exp_effective_rank_torch(WH)), image_shape=(28, 28), n_show=45)

            singular_values = torch.linalg.svdvals(WH_detached).detach().cpu().numpy()
            SVGD1.append(singular_values[0])
            SVGD2.append(singular_values[1])
            coef1.append(W1[0, 0].cpu().numpy())
            coef2.append(W2[0, 0].cpu().numpy())
            coef3.append(H[0, 0].cpu().numpy())
            coef4.append(H[1, 1].cpu().numpy())
    # W2, H = normalize_H_scale_W_diag_torch(W2, H)
    # W1, W2 = normalize_H_scale_W_diag_torch(W1,W2)
    if end == 'lists':
        return W2, H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H
    elif end == 'all':
        return W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2

def NMF_for_r_comparison(A, r, init, end, epochs):
    m, n = np.shape(A)
    lr = 1e-2
    # # ----- Génération d'une matrice V non-négative -----
    # true_W = torch.rand((m, r), device=device)
    # true_H = torch.rand((r, n), device=device)
    # V = torch.matmul(true_W, true_H)
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    
    # ----- Initialisation des facteurs pour descente de gradient -----
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
    Y_tensor = torch.tensor(A, dtype=torch.float32).to(device)  # si besoin

    # ----- Initialization -----
    
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

    # ----- Training -----

    for epoch in tqdm(range(epochs)):
        # ----- Entraînement de la descente de gradient avec Torch ----- 
        optimizer.zero_grad()

        WH = torch.matmul(F.relu(W), F.relu(H))  # Projection non-négative via ReLU
        loss = torch.norm(Y_tensor - WH, p='fro')**2 + 0 * torch.sum(torch.abs(H))
 # perte de Frobenius

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
    m, n = np.shape(A)

    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    
    # ----- Initialisation des facteurs -----
    if init == 'eye':
        W = torch.eye(m, r, device=device)
        H = torch.eye(r, n, device=device)

    elif init == 'random':
        W = torch.rand(m, r, device=device)
        H = torch.rand(r, n, device=device)

    elif init == 'ssvd':
        W, H = nndsvd_init_torch(A, r, device=device)

    # ----- Données -----
    Y_tensor = torch.tensor(A, dtype=torch.float32).to(device)  
    fro_Y = torch.norm(Y_tensor, p='fro') ** 2 

    # ----- Logs -----
    epsilon = 1e-10
    errorsGD = []
    nuclearrankGD = []
    fullerrorsGD = []
    SVGD1 = []
    SVGD2 = []
    rankGD = []

    # ----- Training -----
    for epoch in tqdm(range(epochs)):

        # ---- Multiplicative Updates (Lee & Seung 2001) ----
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

        # ---- Logs ----
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
                print(f"Epoque {epoch}, erreur relative : {rel_error:.4f}, norme A : {torch.norm(Y_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
                print(torch.all(W >= 0))
                print(torch.all(H >= 0))

            singular_values = torch.linalg.svdvals(WH_detached)
            s_np = singular_values.detach().cpu().numpy()
            SVGD1.append(s_np[0])
            if len(s_np) > 1:
                SVGD2.append(s_np[1])

    # ----- Outputs -----
    if end == 'lists':
        return H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W, H
    elif end == 'all':
        return W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2


def plot_heatmap(matrix, title="Heatmap"):
    """
    Affiche une heatmap matplotlib propre à partir d'un numpy array
    ou d'un torch tensor.

    Parameters
    ----------
    matrix : numpy.ndarray ou torch.Tensor
        Matrice 2D à afficher.
    title : str
        Titre de la heatmap.
    """

    # Conversion éventuelle torch → numpy
    try:
        import torch
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
    except ImportError:
        pass  # torch non installé, on ignore

    # Vérification que la matrice est bien 2D
    if matrix.ndim != 2:
        raise ValueError("La matrice doit être 2D pour une heatmap.")

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Valeur")
    plt.title(title)
    plt.xlabel("Colonnes")
    plt.ylabel("Lignes")
    plt.tight_layout()




#---------------APPEL ET RESULTATS---------------------

W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 = Deep_NMF_2W_Malopetitchou(X[100:300, :], 30, 10, 'random', 'all', 35000)
#W_N, H_N, errorsGD_N, rankGD_N, nuclearrankGD_N, SVGD1_N, SVGD2_N = NMF_for_r_comparison_MU(X[100:350, :], 10, 'random', 'all', 5)

plot_nmf_results(W1@W2, H, "deep", errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X)
#plot_nmf_results(W_N, H_N, errorsGD_N, rankGD_N, nuclearrankGD_N, SVGD1_N, SVGD2_N, (3,28,28), X)

# plot_H_signatures(H2,"deep", image_shape=(28, 28), n_show=18)
plot_H_signatures(H,"deep", image_shape=(28, 28), n_show=45)
#plot_H_signatures(H_N,"MU", image_shape=(28, 28), n_show=9)

plot_mnist_reconstruction(X, W1, W2, H, "deep", index=0, image_shape=(28, 28))
#plot_mnist_reconstruction_nmf(X, W_N, H_N, index=0, image_shape=(28, 28))
plot_heatmap(W1, "Heatmap W1")
plot_heatmap(W2, "Heatmap W2")
plot_heatmap(H, "Heatmap H")
# plot_heatmap(H2, "Heatmap H2")

save_dir = r"C:\Users\thoma\Desktop\COde\NMF Graphiques\Advanced ML"

path = os.path.join(save_dir, f"nmf_component.png")
# Base folder
base_dir = r"C:\Users\thoma\Desktop\COde\NMF Graphiques\Advanced ML"
from datetime import datetime

# Dossier unique par exécution
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + str(" Test mALO IMAGES 80000")
save_dir = os.path.join(base_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

# --- Récupère toutes les figures ouvertes et sauvegarde automatiquement
for i, fig_num in enumerate(plt.get_fignums(), 1):
    fig = plt.figure(fig_num)
    path = os.path.join(save_dir, f"figure_{i}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")

# --- Affiche tout à la fin
plt.show()