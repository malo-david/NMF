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
print(np.__version__)

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import os


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

cosmic_df = pd.read_csv("C:/Users/thoma/Python/COSMIC/COSMIC_GRCh37.txt", sep="\t", index_col=0)
# Conversion en matrice NumPy (sans en-têtes ni index)
cosmic_df = pd.read_csv("C:/Users/thoma/Python/COSMIC/COSMIC_GRCh37.txt", sep="\t", index_col=0)
cosmic_rows = cosmic_df.index.tolist()
cosmic_cols = cosmic_df.columns.tolist()

print(cosmic_rows)
cosmic_matrix = cosmic_df.to_numpy().T
print(cosmic_matrix)
print('MIAOU')
print(len(['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5', 'SBS6', 'SBS7a', 'SBS7b',
       'SBS7c', 'SBS7d', 'SBS8', 'SBS9', 'SBS10a', 'SBS10b', 'SBS10c',
       'SBS10d', 'SBS11', 'SBS12', 'SBS13', 'SBS14', 'SBS15', 'SBS16',
       'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21', 'SBS22a',
       'SBS22b', 'SBS23', 'SBS24', 'SBS25', 'SBS26', 'SBS27', 'SBS28', 'SBS29',
       'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34', 'SBS35', 'SBS36', 'SBS37',
       'SBS38', 'SBS39', 'SBS40a', 'SBS40b', 'SBS40c', 'SBS41', 'SBS42',
       'SBS43', 'SBS44', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50',
       'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56', 'SBS57', 'SBS58',
       'SBS59', 'SBS60', 'SBS84', 'SBS85', 'SBS86', 'SBS87', 'SBS88', 'SBS89',
       'SBS90', 'SBS91', 'SBS92', 'SBS93', 'SBS94', 'SBS95', 'SBS96', 'SBS97',
       'SBS98', 'SBS99']))

# Remplace par le chemin de ton fichier .RData
result = pyreadr.read_r("C:/Users/thoma/Python/Re__Data_bases/pcawg_public/pcawg_public/filtered_counts.RData")
result2 = pyreadr.read_r("C:/Users/thoma/Python/Re__Data_bases/pcawg_public/pcawg_public/filtered_counts_plus_oeso.RData")
# Affiche les noms des objets chargés
print("Colonnes COSMIC:", cosmic_df.columns)
print("Lignes COSMIC:", cosmic_rows)
print("Nombre Lignes COSMIC:", len(cosmic_rows))

# Récupère chaque matrice
melacounts = result["melacounts3"]
brcacounts = result["brcacounts3"]
livercounts = result["livercounts3"]
oesocounts = result2["oesocounts3"]
#Exemple : affichage des tailles et des premières lignes
print("Melanoma :\n", melacounts.shape)
print(melacounts.head())
print(melacounts.columns.tolist())

print("\nBreast cancer :\n", brcacounts.shape)
print(brcacounts.head())

print("\nLiver cancer :\n", livercounts.shape)
print(livercounts.head())

B1 = np.array([[10, 0, 1], [14, 1, 1], [12, 0, 0], [13, 1, 0], [17, 0, 1], [1, 10, 0], [0, 15, 0], [0, 20, 1], [0, 17, 1], [0, 22, 2], [1, 16, 0], [0, 15, 0], [0, 19, 1], [1, 17, 1], [0, 24, 2], [0, 2, 34], [1, 2, 25], [1, 0, 12], [1, 0, 16], [0, 5, 16], [2, 2, 24], [1, 0, 34], [0, 1, 11]])
B2 = np.array([[0.7, 0.3, 0], [0.25, 0.75, 0], [0.2, 0.1, 0.7]])
H3 = np.array([[0.5, 0.4, 0, 0, 0.1], [0.1, 0, 0, 0.2, 0.7], [0, 0, 0.8, 0, 0.2]])

D1 = B1
D2 = np.array([[0.7, 0.3, 0], [0.25, 0.75, 0], [0.2, 0.1, 0.7]])
# Indices des lignes à extraire
indices = [0, 14, 27]

# Sous-matrice
HD1 = cosmic_matrix[indices, :]



# Transformations : redimensionnement + conversion en tenseur
# (tu peux adapter pour ton pipeline NMF)
transform = transforms.Compose([
    transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
    transforms.ToTensor()
])

# Charger le dataset
dataset = datasets.Flowers102(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

# Transformations : redimensionnement + conversion en tenseur
# (tu peux adapter pour ton pipeline NMF)
transform = transforms.Compose([
    transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
    transforms.ToTensor()
])

# Charger le dataset
dataset = datasets.Flowers102(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

# Transformations : redimensionnement + conversion en tenseur
# (tu peux adapter pour ton pipeline NMF)
transform = transforms.Compose([
    transforms.Resize((64, 64)),     # résolutions plus petites pour NMF
    transforms.ToTensor()
])

# Charger le dataset
dataset = datasets.Flowers102(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

# Convertir en matrice numpy
X_list = []

for img, _ in dataset:
    arr = img.numpy().reshape(-1)   # aplatissement
    X_list.append(arr)

X = np.stack(X_list)   # Matrice finale (N, D)

print("Shape de X :", X.shape)
print(X)

def normalize_rows_sum1(matrix):
    """
    Normalise chaque ligne pour que la somme des coefficients soit égale à 1.

    Args:
        matrix (np.ndarray): matrice 2D à normaliser.

    Returns:
        np.ndarray: matrice normalisée.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Éviter division par zéro
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
print("\nOeso cancer :\n", oesocounts.shape)
print(oesocounts.head())
# Réordonner les colonnes de chaque cancer selon COSMIC
melacounts = melacounts[cosmic_rows]
brcacounts = brcacounts[cosmic_rows]
livercounts = livercounts[cosmic_rows]
oesocounts = oesocounts[cosmic_rows]
# Conversion en numpy avec colonnes dans l'ordre de COSMIC
A1 = melacounts.values
A2 = brcacounts.values
A3 = livercounts.values
A4 = oesocounts.values
# Empilement
matricecancer = np.vstack((A1, A4))
matricecancer = normalize_rows_sum1(matricecancer)
matricefoie = A3
matricefoie = normalize_rows_sum1(matricefoie)
matricemela = normalize_rows_sum1(A1)


cosmic_df = pd.read_csv("C:/Users/thoma/Python/COSMIC/COSMIC_GRCh37.txt", sep="\t", index_col=0)
# Conversion en matrice NumPy (sans en-têtes ni index)
cosmic_matrix = cosmic_df.to_numpy().T

print(cosmic_matrix)
print(type(cosmic_matrix))  # <class 'numpy.ndarray'>
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(np.shape(cosmic_matrix))
print("Device utilisé :", device)
def heatmap_cosine_similarity(H, COSMIC, 
                              figsize=(10, 8), 
                              cmap='viridis',
                              xticklabels=None,
                              yticklabels=None,
                              title="Cosine similarity heatmap between H rows and COSMIC columns"):
    """
    Calcule la similarité cosinus entre les lignes de H et les colonnes de COSMIC
    et affiche une heatmap.

    Arguments :
    - H : np.array de forme (n_signatures, n_mutations)
    - COSMIC : np.array de forme (n_mutations, m_signatures) ou similaire (ici on considère colonnes = mutations)
               Attention ici, on veut comparer lignes de H à colonnes de COSMIC, donc COSMIC doit être transposée si besoin
    - figsize : taille de la figure matplotlib
    - cmap : palette de couleurs pour la heatmap
    - xticklabels : labels des colonnes (signatures COSMIC)
    - yticklabels : labels des lignes (signatures H)
    - title : titre de la heatmap

    Retourne la matrice de similarité (np.array) et affiche la heatmap.
    """


    # Maintenant H shape: (n_signatures, n_mutations)
    # COSMIC shape: (n_mutations, n_signatures_cosmic)

    # On veut comparer chaque ligne de H avec chaque colonne de COSMIC
    # Donc on transpose COSMIC pour avoir shape (n_signatures_cosmic, n_mutations)
    # Calcul similarité cosinus entre lignes de H (n x d) et lignes de COSMIC_t (m x d)
    # cosine_similarity attend (n_samples, n_features), donc c'est parfait

    similarity_matrix = cosine_similarity(normalize_rows_sum1(H), normalize_rows_sum1(COSMIC))  # shape (n_signatures_H, n_signatures_COSMIC)
    H_rows = [f"H_{i}" for i in range(H.shape[0])]  # labels H (ou une liste custom si tu as les noms)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, cmap=cmap, xticklabels=cosmic_cols, yticklabels=H_rows)
    plt.title(title)
    plt.xlabel('Signatures COSMIC')
    plt.ylabel('Signatures H')
    plt.tight_layout()

    return similarity_matrix

def split_matrix_and_labels_80_20(X, labels, random_state=None):
    """
    Divise X et labels en deux sets train/test (80%/20%) en gardant la correspondance.
    
    Args:
        X (np.ndarray): matrice 2D à diviser (n_samples, n_features)
        labels (list ou np.ndarray): labels associés à chaque ligne de X
        random_state (int, optionnel): graine pour la reproductibilité
    
    Returns:
        X_train, X_test, labels_train, labels_test
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]
    
    return X_train, X_test, labels_train, labels_test



def cosine_similarity(v1, v2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    # Convertir en numpy si nécessaire
    if isinstance(v1, torch.Tensor):
        v1 = v1.detach().cpu().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.detach().cpu().numpy()
        
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot / (norm_v1 * norm_v2)


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


M1 = np.array([[750, 150], [99, 10], [155, 34], [97, 25], [100, 15], [99.3, 4], [92, 5], [98, 450], [0, 100], [1, 99], [2, 98], [15, 97]])
M2 = np.array([[0.7, 0.3], [0.25, 0.75]])
H1 = np.array([[0.5, 0.4, 0, 0, 0.1], [0.1, 0, 0, 0.2, 0.7]])

A1 = np.array([[500, 150], [500, 94], [450, 130], [480, 140], [520, 40], [750, 250], [780, 120], [15, 300], [10, 550], [8, 560], [1, 340]])
A2 = np.array([[0.7, 0.3], [0.25, 0.75]])
H2 = np.array([[0.5, 0.4, 0, 0, 0.1], [0.1, 0, 0, 0.2, 0.7]])


B1 = np.array([[10, 0, 1], [14, 1, 1], [12, 0, 0], [13, 1, 0], [17, 0, 1], [1, 10, 0], [0, 15, 0], [0, 20, 1], [0, 17, 1], [0, 22, 2], [1, 16, 0], [0, 15, 0], [0, 19, 1], [1, 17, 1], [0, 24, 2], [0, 2, 34], [1, 2, 25], [1, 0, 12], [1, 0, 16], [0, 5, 16], [2, 2, 24], [1, 0, 34], [0, 1, 11]])
B2 = np.array([[0.7, 0.3, 0], [0.25, 0.75, 0], [0.2, 0.1, 0.7]])
H3 = np.array([[0.5, 0.4, 0, 0, 0.1], [0.1, 0, 0, 0.2, 0.7], [0, 0, 0.8, 0, 0.2]])

D1 = B1
D2 = np.array([[0.7, 0.3, 0], [0.25, 0.75, 0], [0.2, 0.1, 0.7]])
# Indices des lignes à extraire
indices = [0, 14, 27]

# Sous-matrice
HD = cosmic_matrix[indices, :]
print("HD : ")
print(HD)

def gaussian_noise_matrix(shape, mean=0.0, std=1.0, use_torch=False):
    """
    Génère une matrice de bruit gaussien.
    
    Args:
        shape (tuple): dimensions de la matrice, par ex. (3, 4)
        mean (float): moyenne du bruit
        std (float): écart-type
        use_torch (bool): si True, renvoie un torch.Tensor sinon un np.ndarray
    """
    if use_torch:
        return torch.normal(mean=mean, std=std, size=shape)
    else:
        return np.random.normal(loc=mean, scale=std, size=shape)


B = gaussian_noise_matrix((12, 5), 3, 0.5)

#M_poisson = np.random.poisson(lam=M)


def plot_top3_cosmic(H, COSMIC_matrix, COSMIC_names):
    import matplotlib.pyplot as plt
    
    # S'assurer que COSMIC_matrix a les signatures en lignes
    if COSMIC_matrix.shape[1] != H.shape[1]:
        COSMIC_matrix = COSMIC_matrix.T
    
    for i, h_vector in enumerate(H):
        h_vector = h_vector.reshape(1, -1)
        sims = cosine_similarity(h_vector, COSMIC_matrix)
        top3_idx = sims.argsort()[-3:][::-1]
        top3_sims = sims[top3_idx]
        top3_names = [COSMIC_names[j] for j in top3_idx]
        
        plt.figure(figsize=(6,4))
        plt.bar(top3_names, top3_sims)
        plt.ylim(0,1)
        plt.ylabel("Cosine similarity")
        plt.title(f"H row {i}")
        for j, val in enumerate(top3_sims):
            plt.text(j, val+0.01, f"{val:.3f}", ha='center')



def best_cosmic_matches(H, COSMIC_df):
    """
    Pour chaque ligne de H, retourne la signature COSMIC la plus similaire et le score associé.
    
    Args:
        H: np.array, shape (n_samples, n_features)
        COSMIC_df: pandas DataFrame, shape (n_features, n_signatures)
                   Les colonnes contiennent les noms des signatures COSMIC
    
    Returns:
        best_matches: liste de tuples (nom_signature, score_cosine) pour chaque ligne de H
    """
    COSMIC_matrix = COSMIC_df.values.T  # Transposer pour que les signatures soient en lignes
    COSMIC_names = COSMIC_df.columns.tolist()
    
    best_matches = []
    for h_vector in H:
        h_vector = h_vector.reshape(1, -1)
        # Cosine similarity
        sims = np.dot(h_vector, COSMIC_matrix.T) / (
            np.linalg.norm(h_vector, axis=1)[:, None] * np.linalg.norm(COSMIC_matrix, axis=1)
        )
        sims = sims.flatten()
        top_idx = sims.argmax()
        best_matches.append((COSMIC_names[top_idx], sims[top_idx]))
    
    return best_matches
def supervised_clustering_loss_vectorized(W, labels_tensor, num_classes):
    # W: (n, r2), labels_tensor: (n,)
    centroids = torch.zeros((num_classes, W.shape[1]), device=W.device)  # (num_classes, r2)

    for c in range(num_classes):
        mask = (labels_tensor == c)  # mask shape: (n,)
        if mask.any():
            centroids[c] = W[mask, :].mean(dim=0)  # moyenne sur les samples de classe c

    assigned_centroids = centroids[labels_tensor]  # shape: (n, r2)

    loss = torch.mean((W - assigned_centroids) ** 2)
    return loss
def nndsvd_init_torch(X, rank, device=device, requires_grad=True, dtype=torch.float32):
    """
    Initialisation NNDSVD (basée sur la SVD) pour NMF en PyTorch.
    
    Paramètres
    ----------
    X : torch.Tensor (m x n), non-négatif
        Matrice de données.
    rank : int
        Rang (dimension des facteurs latents).
    device : torch.device ou str
        'cpu' ou 'cuda'. Si None -> même device que X.
    requires_grad : bool
        Si True, renvoie des tensors avec gradient activé.
    dtype : torch.dtype
        Type numérique (par défaut torch.float32).
    
    Retourne
    --------
    W, H : torch.Tensor
        Matrices de facteurs initiaux pour NMF.
    """
    if device is None:
        device = X.device
    
    # SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    
    # Tronquer
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    
    m, n = X.shape
    W = torch.zeros((m, rank), device=device, dtype=dtype)
    H = torch.zeros((rank, n), device=device, dtype=dtype)
    
    # Premier vecteur singulier
    W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
    H[0, :] = torch.sqrt(S[0]) * torch.abs(Vh[0, :])
    
    # Les suivants
    for j in range(1, rank):
        u = U[:, j]
        v = Vh[j, :]
        
        u_p = torch.clamp(u, min=0)
        u_n = torch.clamp(-u, min=0)
        v_p = torch.clamp(v, min=0)
        v_n = torch.clamp(-v, min=0)
        
        u_p_norm = torch.linalg.norm(u_p)
        v_p_norm = torch.linalg.norm(v_p)
        u_n_norm = torch.linalg.norm(u_n)
        v_n_norm = torch.linalg.norm(v_n)
        
        term_p = u_p_norm * v_p_norm
        term_n = u_n_norm * v_n_norm
        
        if term_p >= term_n:
            W[:, j] = torch.sqrt(S[j] * term_p) * (u_p / (u_p_norm + 1e-10))
            H[j, :] = torch.sqrt(S[j] * term_p) * (v_p / (v_p_norm + 1e-10))
        else:
            W[:, j] = torch.sqrt(S[j] * term_n) * (u_n / (u_n_norm + 1e-10))
            H[j, :] = torch.sqrt(S[j] * term_n) * (v_n / (v_n_norm + 1e-10))
    
    # Activer require_grad si demandé
    W.requires_grad_(requires_grad)
    H.requires_grad_(requires_grad)
    
    return W, H



def lignes_correspondantes(matrice1, matrice2):
    """
    Compare les lignes de deux matrices et renvoie le nombre de lignes
    pour lesquelles la ligne la plus proche par cosine similarity est celle de même indice.
    Fonction compatible NumPy arrays et Torch tensors.
    """
    # Convertir en numpy si nécessaire
    if isinstance(matrice1, torch.Tensor):
        matrice1 = matrice1.detach().cpu().numpy()
    if isinstance(matrice2, torch.Tensor):
        matrice2 = matrice2.detach().cpu().numpy()
    
    assert matrice1.shape == matrice2.shape, "Les matrices doivent avoir la même forme"
    n_lignes = matrice1.shape[0]
    score_total = 0
    
    for i in range(n_lignes):
        similitudes = [cosine_similarity(matrice1[i], matrice2[j]) for j in range(n_lignes)]
        index_max = np.argmax(similitudes)
        if index_max == i:
            score_total += 1
    
    return score_total


def mean_cosine_similarity(torch_mat: torch.Tensor, numpy_mat: np.ndarray) -> float:
    """
    Calcule la moyenne de la cosine similarity entre les lignes correspondantes 
    d'une matrice PyTorch et d'une matrice NumPy.
    
    Args:
        torch_mat (torch.Tensor): Matrice PyTorch de forme (n, d)
        numpy_mat (np.ndarray): Matrice NumPy de forme (n, d)
    
    Returns:
        float: Moyenne des cosine similarities.
    """
    # Vérifier que les dimensions correspondent
    if torch_mat.shape != numpy_mat.shape:
        raise ValueError("Les deux matrices doivent avoir la même forme.")
    
    # Convertir numpy -> torch (même type que torch_mat)
    numpy_as_torch = torch.from_numpy(numpy_mat).to(torch_mat.dtype).to(torch_mat.device)
    
    # Normalisation des lignes pour la cosine similarity
    torch_norm = torch.nn.functional.normalize(torch_mat, p=2, dim=1)
    numpy_norm = torch.nn.functional.normalize(numpy_as_torch, p=2, dim=1)
    
    # Cosine similarity ligne par ligne (produit scalaire après normalisation)
    similarities = torch.sum(torch_norm * numpy_norm, dim=1)
    
    # Retourner la moyenne
    return similarities.mean().item()
def nndsvd_init(A, r, device='cpu'):
    """
    Initialisation NNDSVD pour NMF.
    
    Args:
        A : matrice non-négative (numpy ou torch tensor)
        r : rang
    Returns:
        W, H : matrices initialisées non négatives
    """
    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy()
    else:
        A_np = A

    # SVD
    U, S, Vt = np.linalg.svd(A_np, full_matrices=False)
    
    W = np.zeros((A_np.shape[0], r))
    H = np.zeros((r, A_np.shape[1]))
    
    # 1ère composante
    W[:,0] = np.sqrt(S[0]) * np.maximum(0, U[:,0])
    H[0,:] = np.sqrt(S[0]) * np.maximum(0, Vt[0,:])
    
    for j in range(1, r):
        u = U[:,j]
        v = Vt[j,:]
        
        u_pos = np.maximum(u, 0)
        u_neg = np.maximum(-u, 0)
        v_pos = np.maximum(v, 0)
        v_neg = np.maximum(-v, 0)
        
        u_pos_norm = np.linalg.norm(u_pos)
        v_pos_norm = np.linalg.norm(v_pos)
        u_neg_norm = np.linalg.norm(u_neg)
        v_neg_norm = np.linalg.norm(v_neg)
        
        m_pos = u_pos_norm * v_pos_norm
        m_neg = u_neg_norm * v_neg_norm
        
        if m_pos >= m_neg:
            W[:,j] = np.sqrt(S[j] * m_pos) * (u_pos / (u_pos_norm + 1e-10))
            H[j,:] = np.sqrt(S[j] * m_pos) * (v_pos / (v_pos_norm + 1e-10))
        else:
            W[:,j] = np.sqrt(S[j] * m_neg) * (u_neg / (u_neg_norm + 1e-10))
            H[j,:] = np.sqrt(S[j] * m_neg) * (v_neg / (v_neg_norm + 1e-10))
    
    W = torch.tensor(W, dtype=torch.float32, device=device, requires_grad=True)
    H = torch.tensor(H, dtype=torch.float32, device=device, requires_grad=True)
    
    return W, H
def deep_nmf_init_torch(X, rank1, rank2, device=None, requires_grad=True, dtype=torch.float32):
    """
    Initialisation NNDSVD pour Deep NMF : X ≈ W1 W2 H
    
    Paramètres
    ----------
    X : torch.Tensor (m x n), non-négatif
        Matrice de données.
    rank1 : int
        Rang de la première couche (W1).
    rank2 : int
        Rang de la deuxième couche (W2).
    device : torch.device ou str
        'cpu' ou 'cuda'. Si None -> même device que X.
    requires_grad : bool
        Si True, renvoie des tensors avec gradient activé.
    dtype : torch.dtype
        Type numérique (par défaut torch.float32).
    
    Retourne
    --------
    W1, W2, H : torch.Tensor
        Matrices initialisées pour Deep NMF.
    """
    if device is None:
        device = X.device

    # 1. Initialisation W1 * H1 ≈ X via NNDSVD classique
    W1, H1 = nndsvd_init_torch(X, rank=rank1, device=device, requires_grad=False, dtype=dtype)
    
    # 2. Factorisation de H1 ≈ W2 * H
    W2, H = nndsvd_init_torch(H1, rank=rank2, device=device, requires_grad=False, dtype=dtype)
    
    # 3. Activer require_grad sur toutes les matrices
    W1.requires_grad_(requires_grad)
    W2.requires_grad_(requires_grad)
    H.requires_grad_(requires_grad)
    
    return W1, W2, H

def top_cosmic_matches(H, COSMIC, cosmic_cols, device="cpu"):
    """
    Trouve les meilleures correspondances entre les lignes de H et les signatures COSMIC.
    Évite les doublons : si une signature COSMIC est retrouvée plusieurs fois,
    on ne garde que la meilleure correspondance.

    Args:
        H (torch.Tensor ou np.ndarray) : matrice (k x m)
        COSMIC (torch.Tensor ou np.ndarray) : matrice (86 x m)
        cosmic_cols (list of str) : noms des signatures COSMIC (longueur = nb colonnes de COSMIC)
        device (str) : "cpu" ou "cuda"

    Returns:
        matches (list of tuples) : [(ligne_H, nom_signature_COSMIC, score_cosine), ...]
        similarity_matrix (torch.Tensor) : matrice des cosines (k x 86)
    """
    # Conversion vers torch si nécessaire
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
    if not isinstance(COSMIC, torch.Tensor):
        COSMIC = torch.tensor(COSMIC, dtype=torch.float32)

    H = H.to(device)
    COSMIC = COSMIC.to(device)

    # Normalisation ligne par ligne
    H_norm = H / (H.norm(dim=1, keepdim=True) + 1e-8)
    C_norm = COSMIC / (COSMIC.norm(dim=1, keepdim=True) + 1e-8)

    # Similarités cosinus (k x 86)
    similarity_matrix = torch.mm(H_norm, C_norm.T)

    # Chercher pour chaque ligne de H sa meilleure correspondance
    best_indices = torch.argmax(similarity_matrix, dim=1)
    best_scores = torch.max(similarity_matrix, dim=1).values

    # Stockage des correspondances initiales
    all_matches = [
        (i, int(best_indices[i].cpu()), float(best_scores[i].cpu()))
        for i in range(H.shape[0])
    ]

    # Supprimer les doublons côté COSMIC → garder uniquement la meilleure correspondance
    best_per_cosmic = {}
    for h_idx, c_idx, score in all_matches:
        if c_idx not in best_per_cosmic or score > best_per_cosmic[c_idx][2]:
            best_per_cosmic[c_idx] = (h_idx, c_idx, score)

    # Récupérer les correspondances finales (sans doublons COSMIC)
    matches = [
        (h_idx, cosmic_cols[c_idx], score)  # ici on traduit l’indice → nom SBSk
        for h_idx, c_idx, score in best_per_cosmic.values()
    ]

    return matches

def normalize_H_scale_W_diag_torch(W, H):
    """
    Normalise les lignes de H (sommes = 1) avec matrice diagonale
    et ajuste W, en PyTorch GPU.
    """
    row_sums = H.sum(dim=1, keepdim=True)  # shape (r,1)
    row_sums[row_sums == 0] = 1.0

    D_inv = torch.diag(1.0 / row_sums.flatten())
    D = torch.diag(row_sums.flatten())

    H_new = D_inv @ H
    W_new = W @ D
    return W_new, H_new

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
        l1_lambda = 0.00
        loss = torch.norm(A_tensor - WH, p='fro')**2 + l1_lambda * torch.norm(W1, p = 1)

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
                # print("Lignes correspondantes W1 : " + str(lignes_correspondantes(W1, D1)))
                # print("Lignes correspondantes W2 : " + str(lignes_correspondantes(W2, D2)))
                # print("Lignes correspondantes H : " + str(lignes_correspondantes(H, HD)))

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


def plot_nmf_results(W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2,
                     image_shape=(3,64,64), X=None):

    # --- Conversion CPU + numpy si nécessaire
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # =====================
    # FIGURE 1 : Heatmaps
    # =====================
    plt.figure(figsize=(14, 6))

    # Heatmap W
    plt.subplot(1, 2, 1)
    plt.title("Heatmap - W")
    plt.imshow(W, aspect='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("Features")
    plt.ylabel("Components")

    # Heatmap H
    plt.subplot(1, 2, 2)
    plt.title("Heatmap - H")
    plt.imshow(H, aspect='auto', cmap='viridis')
    plt.colorbar(label="Intensity")
    plt.xlabel("Samples")
    plt.ylabel("Components")

    plt.tight_layout()

    # ==========================
    # FIGURE 2 : Courbes
    # ==========================
    plt.figure(figsize=(14, 8))

    titles = ["errorsGD", "rankGD", "nuclearrankGD", "SVGD1", "SVGD2"]
    data   = [errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2]

    for i, (title, y) in enumerate(zip(titles, data)):
        plt.subplot(3, 2, i+1)
        plt.plot(y)
        plt.title(title)
        plt.grid(True)

    plt.tight_layout()

    # ===========================================
    # FIGURE 3 : Images correspondant à H
    # ===========================================
    num_components = H.shape[0]

    n_cols = 8
    n_rows = int(np.ceil(num_components / n_cols))

    plt.figure(figsize=(2*n_cols, 2*n_rows))

    for i in range(num_components):
        comp = H[i]
        if comp.size != np.prod(image_shape):
            continue  # ignore si reshape impossible

        img_array = comp.reshape(image_shape).transpose(1,2,0)

        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img_array)
        plt.axis("off")
        plt.title(f"H[{i}]", fontsize=8)

    plt.suptitle("Reconstruction des lignes de H", fontsize=16)
    plt.tight_layout()

    # ==================================================
    # FIGURE 4 (optionnelle) : images du dataset X
    # ==================================================
    if X is not None:
        max_imgs = min(100, X.shape[0])
        n_cols = 8
        n_rows = int(np.ceil(max_imgs / n_cols))
        plt.figure(figsize=(2*n_cols, 2*n_rows))

        for i in range(100, 200):
            comp = X[i]
            if comp.size != np.prod(image_shape):
                continue

            img_array = comp.reshape(image_shape).transpose(1,2,0)

            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(img_array)
            plt.axis("off")
            plt.title(f"img {i}", fontsize=8)

        plt.suptitle("Images du dataset X", fontsize=16)
        plt.tight_layout()

    plt.show()

def safe_subplot(index, total, n_cols=8):
    """
    Calcule automatiquement le subplot sans jamais dépasser la grille.
    """
    n_rows = int(np.ceil(total / n_cols))
    return n_rows, n_cols, index+1


def plot_nmf_results(W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2,
                     image_shape=(3,64,64), X=None):

    # --- Convertir torch -> numpy si nécessaire
    if isinstance(W, torch.Tensor): W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor): H = H.detach().cpu().numpy()
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()

    import numpy as np
    import matplotlib.pyplot as plt

    # ==========================================================
    # FIGURE 1 : Heatmaps W et H
    # ==========================================================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Heatmap W")
    plt.imshow(W, aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Heatmap H")
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
        plt.subplot(3, 2, i+1)
        plt.plot(y)
        plt.title(t)
        plt.grid(True)

    plt.tight_layout()


    # ==========================================================
    # FIGURE 3 : Composantes NMF (= lignes de H) reconstruites en images
    # ==========================================================
    num_H = H.shape[0]
    n_cols = 8
    n_rows = int(np.ceil(num_H / n_cols))

    plt.figure(figsize=(2*n_cols, 2*n_rows))

    for i in range(num_H):
        comp = H[i]

        # reshape en image
        img = comp.reshape(image_shape).transpose(1, 2, 0)

        # clamp pour éviter le warning "Clipping input data"
        img = np.clip(img, 0, 1)

        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"H[{i}]", fontsize=7)

    plt.suptitle("Images reconstruites depuis H", fontsize=14)
    plt.tight_layout()


    # ==========================================================
    # FIGURE 4 : Images d'entrée X (optionnelles)
    # ==========================================================
    if X is not None:
        num_X = min(150, X.shape[0])  # éviter 1000 images en grille…
        n_cols = 8
        n_rows = int(np.ceil(num_X / n_cols))

        plt.figure(figsize=(2*n_cols, 2*n_rows))

        for i in range(num_X):
            comp = X[i]

            img = comp.reshape(image_shape).transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"X[{i}]", fontsize=7)

        plt.suptitle("Images originales X", fontsize=14)
        plt.tight_layout()

    plt.show()



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
                print(f"Époque {epoch}, erreur relative : {rel_error:.4f}, norme A : {torch.norm(Y_tensor, p='fro') ** 2:.4f}, norme WH : {torch.norm(WH_detached, p='fro') ** 2:.4f}")
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





from scipy.optimize import linear_sum_assignment
from numpy.linalg import norm

def cosine_sim_matrix(A, B, by="cols"):
    # A,B: même forme hors permutation. by="cols" (W), "rows" (H)
    if by == "cols":
        A_ = A / np.maximum(norm(A, axis=0, keepdims=True), 1e-12)
        B_ = B / np.maximum(norm(B, axis=0, keepdims=True), 1e-12)
        return A_.T @ B_
    else:
        A_ = A / np.maximum(norm(A, axis=1, keepdims=True), 1e-12)
        B_ = B / np.maximum(norm(B, axis=1, keepdims=True), 1e-12)
        return A_ @ B_.T


def align_to_reference(W_ref, W_new, by="cols"):
    # aligne W_new sur W_ref via Hungarian en maximisant la similarité cosinus
    S = cosine_sim_matrix(W_ref, W_new, by=by)       # (k x k)
    cost = 1 - S                                     # minimiser le coût = 1 - cos
    row_ind, col_ind = linear_sum_assignment(cost)
    if by == "cols":
        W_new_al = W_new[:, col_ind]
    else:
        W_new_al = W_new[row_ind, :]  # si by="rows", on permute les lignes
    return W_new_al
def mean_matrices(matrices):
    """
    matrices : list of np.ndarray
        Liste de matrices de même dimension.
        
    return : np.ndarray
        La matrice moyenne.
    """
    # Empile les matrices le long d'un nouvel axe, puis fait la moyenne
    return np.mean(np.stack(matrices, axis=0), axis=0)
def mean_per_epoch(loss_runs):
    """
    loss_runs : list of lists
        Chaque sous-liste correspond aux losses d'un run (1 valeur par epoch).
    
    return : list
        Moyenne des losses à chaque epoch.
    """
    max_len = max(len(run) for run in loss_runs)
    padded = np.full((len(loss_runs), max_len), np.nan)

    for i, run in enumerate(loss_runs):
        padded[i, :len(run)] = run

    mean_losses = np.nanmean(padded, axis=0)
    return mean_losses.tolist()
def plot_benchmark(Losses, DeepLosses, Ranks, DeepRanks, r1, r2, k):
    epochs_loss = range(1, len(Losses) + 1)
    epochs_rank = range(1, len(Ranks) + 1)

    # Figure pour les Losses
    plt.figure(figsize=(7,5))
    plt.plot(epochs_loss, Losses, label="Mean NMF MU Losses of rank " + str(r2) + " after " + str(k) + " different trainings")
    plt.plot(epochs_loss, DeepLosses, label="Mean Deep NMF Losses of rank "+ str((r1,r2)) + " after " + str(k) + " different trainings")
    # plt.plot(epochs_loss, MULosses, label="Mean NMF MU Losses of rank " + str(r2) + " after " + str(k) + " different trainings")
    # plt.plot(epochs_loss, DeepSLosses, label="Mean Deep NMF Losses of rank "+ str((r3,r2)) + " after " + str(k) + " different trainings")

    plt.xlabel("Epoch")
    plt.ylabel("Mean relative loss")
    plt.yscale("log")
    plt.title("Benchmark des Losses")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Figure pour les Ranks
    plt.figure(figsize=(7,5))
    plt.plot(epochs_rank, Ranks, label="NMF MU effective rank")
    plt.plot(epochs_rank, DeepRanks, label="Deep NMF effective ranks")
    # plt.plot(epochs_rank, MURanks, label="NMF MU effective ranks")
    # plt.plot(epochs_rank, DeepSRanks, label="Deep NMF Tuned effective ranks")

    plt.xlabel("Epoch")
    plt.ylabel("Rank")
    plt.title("Benchmark of Ranks")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

def compare_top5_per_method(matches_list_A, matches_list_B, method_names=("NMF MU", "Deep NMF"), top_k=8):
    """
    Compare les top_k meilleures correspondances COSMIC de deux méthodes,
    avec deux barplots côte à côte.
    
    Args:
        matches_list_A (list of list of tuples): résultats de top_cosmic_matches pour plusieurs runs de la méthode A.
        matches_list_B (list of list of tuples): idem pour méthode B.
        method_names (tuple): noms des méthodes.
        top_k (int): nombre de meilleures similarités à afficher par méthode.
    """

    # Aplatir les résultats
    flat_A = [match for run in matches_list_A for match in run]
    flat_B = [match for run in matches_list_B for match in run]

    # DataFrames
    df_A = pd.DataFrame(flat_A, columns=["H_idx", "COSMIC", "cosine"])
    df_A["method"] = method_names[0]
    df_B = pd.DataFrame(flat_B, columns=["H_idx", "COSMIC", "cosine"])
    df_B["method"] = method_names[1]

    # Garder le meilleur score par signature pour chaque méthode
    df_A = df_A.groupby("COSMIC", as_index=False)["cosine"].max().nlargest(top_k, "cosine")
    df_A["method"] = method_names[0]

    df_B = df_B.groupby("COSMIC", as_index=False)["cosine"].max().nlargest(top_k, "cosine")
    df_B["method"] = method_names[1]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(data=df_A, x="COSMIC", y="cosine", ax=axes[0], palette="Blues_d")
    axes[0].set_title(f"Top {top_k} {method_names[0]}")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_xlabel("Signatures COSMIC")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df_B, x="COSMIC", y="cosine", ax=axes[1], palette="Greens_d")
    axes[1].set_title(f"Top {top_k} {method_names[1]}")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Signatures COSMIC")
    axes[1].tick_params(axis='x', rotation=45)

    # Ajouter les valeurs au-dessus
    for ax, df in zip(axes, [df_A, df_B]):
        for i, row in df.iterrows():
            ax.text(i, row["cosine"] + 0.01, f"{row['cosine']:.2f}", 
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Comparaison des meilleures correspondances COSMIC entre méthodes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])





def compare_top5_three_methods(matches_list_A, matches_list_B, matches_list_C, 
                               method_names=("NMF PGD", "Deep NMF", "NMF MU"), 
                               top_k=8):
    """
    Compare les top_k meilleures correspondances COSMIC de trois méthodes,
    avec trois barplots côte à côte.
    
    Args:
        matches_list_A (list of list of tuples): résultats de top_cosmic_matches pour plusieurs runs méthode A.
        matches_list_B (list of list of tuples): idem pour méthode B.
        matches_list_C (list of list of tuples): idem pour méthode C.
        method_names (tuple): noms des méthodes (3 éléments).
        top_k (int): nombre de meilleures similarités à afficher par méthode.
    """

    def process_matches(matches_list, method_name):
        flat = [match for run in matches_list for match in run]
        df = pd.DataFrame(flat, columns=["H_idx", "COSMIC", "cosine"])
        df = df.groupby("COSMIC", as_index=False)["cosine"].max().nlargest(top_k, "cosine")
        df["method"] = method_name
        return df

    df_A = process_matches(matches_list_A, method_names[0])
    df_B = process_matches(matches_list_B, method_names[1])
    df_C = process_matches(matches_list_C, method_names[2])

    dfs = [df_A, df_B, df_C]
    palettes = ["Blues_d", "Greens_d", "Reds_d"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, df, method, palette in zip(axes, dfs, method_names, palettes):
        sns.barplot(data=df, x="COSMIC", y="cosine", ax=ax, palette=palette)
        ax.set_title(f"Top {top_k} {method}")
        ax.set_xlabel("Signatures COSMIC")
        ax.set_ylabel("Cosine similarity" if ax == axes[0] else "")
        ax.tick_params(axis='x', rotation=45)

        # Ajouter les valeurs au-dessus des barres
        for i, row in df.iterrows():
            ax.text(i, row["cosine"] + 0.01, f"{row['cosine']:.2f}", 
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Comparaison des meilleures correspondances COSMIC entre trois méthodes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def compare_top5_per_method2(matches_list_A, matches_list_B, matches_list_C, matches_list_D,
                            method_names=("NMF PGD", "Deep NMF", "NMF MU", "Autre"), top_k=8):
    """
    Compare les top_k meilleures correspondances COSMIC de quatre méthodes,
    avec quatre barplots côte à côte.
    
    Args:
        matches_list_X (list of list of tuples): résultats de top_cosmic_matches pour plusieurs runs.
        method_names (tuple): noms des méthodes.
        top_k (int): nombre de meilleures similarités à afficher par méthode.
    """

    # Fonction pour transformer les listes en DataFrame top_k
    def prepare_df(matches_list, method_name):
        flat = [match for run in matches_list for match in run]
        df = pd.DataFrame(flat, columns=["H_idx", "COSMIC", "cosine"])
        df = df.groupby("COSMIC", as_index=False)["cosine"].max().nlargest(top_k, "cosine")
        df["method"] = method_name
        return df

    dfs = [
        prepare_df(matches_list_A, method_names[0]),
        prepare_df(matches_list_B, method_names[1]),
        prepare_df(matches_list_C, method_names[2]),
        prepare_df(matches_list_D, method_names[3])
    ]

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    palettes = ["Blues_d", "Greens_d", "Oranges_d", "Purples_d"]

    for ax, df, palette, name in zip(axes, dfs, palettes, method_names):
        sns.barplot(data=df, x="COSMIC", y="cosine", ax=ax, palette=palette)
        ax.set_title(f"Top {top_k} {name}")
        ax.set_xlabel("Signatures COSMIC")
        ax.set_ylabel("Cosine similarity" if ax == axes[0] else "")
        ax.tick_params(axis='x', rotation=45)

        # Ajouter les valeurs au-dessus des barres
        for i, row in df.iterrows():
            ax.text(i, row["cosine"] + 0.01, f"{row['cosine']:.2f}", 
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Comparaison des meilleures correspondances COSMIC entre méthodes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

def average_cosine_heatmap_lines(H_list, H_cosmic, device='cpu', title=''):
    """
    H_list : liste de matrices (k_H x n) torch ou numpy
    H_cosmic : matrice de référence (k_cosmic x n) torch ou numpy
    """
    # Conversion en torch.Tensor si nécessaire
    if not isinstance(H_cosmic, torch.Tensor):
        H_cosmic = torch.tensor(H_cosmic, dtype=torch.float32)
    H_cosmic = H_cosmic.to(device)

    H_list_torch = []
    for H in H_list:
        if not isinstance(H, torch.Tensor):
            H = torch.tensor(H, dtype=torch.float32)
        H_list_torch.append(H.to(device))
    H_list = H_list_torch

    # Normalisation des lignes pour cosine similarity
    def row_normalize(X):
        return X / (X.norm(dim=1, keepdim=True) + 1e-8)

    H_cosmic_norm = row_normalize(H_cosmic)
    cos_sims_aligned = []

    for H in H_list:
        H_norm = row_normalize(H)
        # Cosine similarity matrix (k_H x k_cosmic)
        C = torch.mm(H_norm, H_cosmic_norm)
        C_np = C.cpu().numpy()

        # Hungarian Matching pour aligner les lignes de H aux COSMIC
        # On veut maximiser les correspondances : convertit en problème de minimisation
        C_cost = -C_np  # négatif pour maximiser
        row_ind, col_ind = linear_sum_assignment(C_cost)
        # Réordonne les lignes de H
        C_aligned = C_np[row_ind, :]
        cos_sims_aligned.append(torch.tensor(C_aligned, device=device))

    # Moyenne des matrices alignées
    avg_C = torch.stack(cos_sims_aligned).mean(dim=0).cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(avg_C, cmap='viridis', square=False, xticklabels=cosmic_cols)
    plt.xlabel('COSMIC Signatures')
    plt.ylabel('Recovered Signatures')
    plt.title(title)

    return avg_C

def Deep_NMF_2W_supervised_clustering(A, r1, r2, labels, init='random', end='all', epochs=5000, alpha = 0.1, beta=0.1):
    m, n = np.shape(A)
    lr = 1e-2
    num_classes = len(np.unique(labels))
    
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)  # str → int
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long).to(device)
    A_tensor = torch.tensor(A, dtype=torch.float32).to(device)

    # Initialisation
    if init == 'eye':
        W1 = torch.eye(m, r1, device=device, requires_grad=True)
        W2 = torch.eye(r1, r2, device=device, requires_grad=True)
        H  = torch.eye(r2, n, device=device, requires_grad=True)
    elif init == 'random':
        W1 = torch.rand((m, r1), device=device, requires_grad=True)
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        H  = torch.rand((r2, n), device=device, requires_grad=True)

    classifier = torch.nn.Linear(r2, num_classes).to(device)
    all_params = [W1, W2, H] + list(classifier.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Initialisation des listes pour les suivis
    errorsGD, nuclearrankGD, fullerrorsGD, rankGD = [], [], [], []
    SVGD1, SVGD2, SVGD3 = [], [], []
    coef1, coef2, coef3, coef4 = [], [], [], []
    accuracy_list = []
    SBS2 = []
    SBS7a = []
    SBS40a = []
    SBS40b = []
    SBS13 = []
    fro_Y = torch.norm(A_tensor, p='fro') ** 2 

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # Reconstruction
        W1_pos, W2_pos, H_pos = F.relu(W1), F.relu(W2), F.relu(H)
        WH = W1_pos @ W2_pos @ H_pos
        recon_loss = torch.norm(A_tensor - WH, p='fro')**2 / fro_Y

        # Supervised classification
        logits = classifier(W1@W2)  # (n, num_classes)
        classif_loss = F.cross_entropy(logits, labels_tensor)

        # Clustering loss (simple cohesion with mean)
        centroids = torch.mean(H_pos, dim=1, keepdim=True)
        cluster_loss = torch.norm(H_pos - centroids) / n

        # Regularization
        l1_lambda = 0.005
        l1_loss = l1_lambda * (torch.norm(W1_pos, p=1))

        # Total loss
        loss = recon_loss + alpha * classif_loss + beta * supervised_clustering_loss_vectorized(W1@W2, labels_tensor, num_classes) + l1_loss
        loss.backward()
        optimizer.step()

        # Clamp (relu peut parfois suffire)
        

        with torch.no_grad():
            W1.clamp_(min=0)
            W2.clamp_(min=0)
            H.clamp_(min=0)
            WH_detached = W1_pos @ W2_pos @ H_pos
            rel_error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2).item()
            errorsGD.append(rel_error)
            fullerrorsGD.append(error)
            rankGD.append(exp_effective_rank_torch(WH_detached))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH_detached))

            # Valeurs singulières
            singular_values = torch.linalg.svdvals(WH_detached).detach().cpu().numpy()
            SVGD1.append(singular_values[0])
            SVGD2.append(singular_values[1])
            SVGD3.append(singular_values[2])


            # Accuracy
            pred_labels = logits.argmax(dim=1)
            acc = (pred_labels == labels_tensor).float().mean().item()
            accuracy_list.append(acc)

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"[{epoch}] RelErr: {rel_error:.4f} | Acc: {acc:.3f} | ClsLoss: {classif_loss.item():.4f} | Recon: {recon_loss.item():.4f} | ClustLoss: {cluster_loss.item():.4f}")

    if end == 'lists':
        return H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H
    elif end == 'all':
        return W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, SBS2, SBS7a, SBS13, SBS40b, accuracy_list, classifier
def Deep_NMF_3W(A, r1, r2, r3, init='random', end='matrix', epochs=5000):
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
        W1 = torch.rand((m, r1), device=device, requires_grad=True)
        W2 = torch.rand((r1, r2), device=device, requires_grad=True)
        W3 = torch.rand((r2, r3), device=device, requires_grad=True)
        H  = torch.rand((r3, n), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([W1, W2, W3, H], lr=lr)

    errorsGD, nuclearrankGD, fullerrorsGD, rankGD = [], [], [], []
    SVGD1, SVGD2, SVGD3, SVGD4 = [], [], [], []
    coef1, coef2, coef3, coef4 = [], [], [], []
    SBS2 = []
    SBS7a = []
    SBS40a = []
    SBS40b = []
    fro_Y = torch.norm(A_tensor, p='fro') ** 2 

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Reconstruction via W1 * W2 * H
        WH = F.relu(W1) @ F.relu(W2) @ F.relu(W3) @ F.relu(H)
        l1_lambda = 0.03
        loss = torch.norm(A_tensor - WH, p='fro')**2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Clamp (optionnel, au cas où relu ne suffit pas)
            W1.clamp_(min=0)
            W2.clamp_(min=0)
            H.clamp_(min=0)

            WH_detached = F.relu(W1) @ F.relu(W2) @ F.relu(W3) @ F.relu(H)
            rel_error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2 / fro_Y).item()
            error = (torch.norm(A_tensor - WH_detached, p='fro') ** 2).item()

            errorsGD.append(rel_error)
            fullerrorsGD.append(error)
            rankGD.append(exp_effective_rank_torch(WH))
            nuclearrankGD.append(nuclear_over_operator_norm_torch(WH))

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"Époque {epoch}, erreur relative : {rel_error:.4f}")


    if end == 'lists':
        return W3, H, errorsGD, rankGD, SVGD1, SVGD2
    elif end == 'matrix':
        return W1, W2, H
    elif end == 'all':
        return W1@W2@W3, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, SBS2, SBS7a, SBS40a, SBS40b, SVGD3, SVGD4

print(matricecancer)
labels = ['Melanoma'] * 70 + ['Breast'] * 120 + ['Liver'] * 260 + ['Esofagus'] * 97
print(len(labels))
def pipelineNMF(k, r1, r2, r3):
    DeepLosses = []
    Losses = []
    MULosses = []
    DeepSLosses = []
    DeepRanks = []
    Ranks = []
    MURanks = []
    DeepSRanks = []
    ListeHD = []
    ListeH = []
    ListeHMU = []
    ListeHDS = []
    for i in range(k):
        matricecancer_train, matricecancer_test, labels_train, labels_test = split_matrix_and_labels_80_20(matricecancer, labels)
        A = matricecancer_train
        HD, errorD, rankD, SVGD1D, SVGD2D = Deep_NMF_2W(A, r1, r2, 'random', 'lists', 5)
        H, error, rank, SVGD1, SVGD2 = NMF_for_r_comparison(A, r2, 'random', 'lists', 5)
        HMU, errorMU, rankMU, SVGD1MU, SVGD2MU = NMF_for_r_comparison_MU(A, r2, 'random', 'lists', 5)
        HDS, errorDS, rankDS, SVGD1DS, SVGD2DS = Deep_NMF_2W(A, r3, r2, 'random', 'lists', 5)
        H_numpy = H.detach().cpu().numpy()
        HD_numpy = HD.detach().cpu().numpy()
        HMU_numpy = HMU.detach().cpu().numpy()
        HDS_numpy = HDS.detach().cpu().numpy()

        # if i > 0:
        #     HD_numpy = align_to_reference(ListeHD[0], HD_numpy, "rows")
        #     H_numpy = align_to_reference(ListeH[0], H_numpy, "rows")
        #     H_numpy = align_to_reference(ListeH[0], H_numpy, "rows")
        DeepLosses.append(errorD)
        Losses.append(error)
        MULosses.append(errorMU)
        DeepRanks.append(rankD)
        DeepSLosses.append(errorDS)

        Ranks.append(rank)
        MURanks.append(rankMU)
        DeepSRanks.append(rankDS)

        ListeHD.append(HD_numpy)
        ListeH.append(H_numpy)
        ListeHMU.append(HMU_numpy)
        ListeHDS.append(HDS_numpy)

        matches = top_cosmic_matches(H, cosmic_matrix, cosmic_cols)
        print("Correspondances retenues :")
        for h_idx, c_idx, score in matches:
            print(f"Signature H {h_idx} ↔ COSMIC {c_idx} (cosine={score:.4f})")
    MeanLosses = mean_per_epoch(Losses)
    MeanDeepLosses = mean_per_epoch(DeepLosses)
    MeanMULosses = mean_per_epoch(MULosses)
    MeanDeepSLosses = mean_per_epoch(DeepSLosses)
    MeanRanks = mean_per_epoch(Ranks)
    MeanDeepRanks = mean_per_epoch(DeepRanks)
    MeanMURanks = mean_per_epoch(MURanks)
    MeanDeepSRanks = mean_per_epoch(DeepSRanks)

    HDmean = mean_matrices(ListeHD)
    Hmean = mean_matrices(ListeH)
    matches_list_A = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHD]
    matches_list_B = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeH]
    matches_list_C = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHMU]
    matches_list_D = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHDS]

    compare_top5_per_method2(matches_list_A, matches_list_B, matches_list_C, matches_list_D, method_names=("Deep NMF", "NMF PGD", "NMF MU", "Deep NMF higher rank"))

    plot_benchmark(MeanLosses, MeanDeepLosses, MeanMULosses, MeanDeepSLosses, MeanRanks, MeanDeepRanks, MeanMURanks, MeanDeepSRanks, r1, r2, k)
    # average_cosine_heatmap_lines(ListeHD, cosmic_matrix.T, title = "Deep NMF COSMIC Signatures Cosine Similarities")
    # average_cosine_heatmap_lines(ListeH, cosmic_matrix.T, title = "Normal NMF COSMIC Signatures Cosine Similarities")

def compare_to_COSMIC(H, cosmic_file, title = "Cosine Similarities"):
    # Chargement COSMIC
    cosmic_df = pd.read_csv(cosmic_file, sep='\t', index_col=0).T
    
    #print("COSMIC shape:", cosmic_df.shape)  # (n_cosmic, 96)
    #print("H shape:", H.shape)               # (n_signatures, 96)

    # Normalisation L2 de COSMIC
    cosmic_values = cosmic_df.values
    cosmic_norm = cosmic_values / np.linalg.norm(cosmic_values, axis=1)[:, None]
    cosmic_norm = pd.DataFrame(cosmic_norm, index=cosmic_df.index, columns=cosmic_df.columns)

    # Normalisation L2 de H (ignore lignes trop petites)
    H = H if isinstance(H, np.ndarray) else H.detach().cpu().numpy()
    norms = np.linalg.norm(H, axis=1)
    threshold = 1e-6
    H_norm = H.copy()
    mask = norms > threshold
    H_norm[mask] = H[mask] / norms[mask, None]

    # Similarité cosinus
    similarity = cosine_similarity(H_norm, cosmic_norm.values)

    # Meilleurs matchs
    best_match_indices = np.argmax(similarity, axis=1)
    best_match_scores = np.max(similarity, axis=1)
    
    best_matches = []
    for i, (idx, score) in enumerate(zip(best_match_indices, best_match_scores)):
        sig = cosmic_norm.index[idx]
        print(f"Signature {i} best matches COSMIC signature {sig} with score {score:.3f}")
        best_matches.append((i, sig, score))

    # Heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(similarity, aspect='auto', cmap='viridis')
    plt.colorbar(label='Cosine similarity')
    plt.xlabel('COSMIC signatures')
    plt.ylabel('h signatures')
    plt.xticks(ticks=np.arange(len(cosmic_norm.index)), labels=cosmic_norm.index, rotation=90, fontsize=8)
    plt.title(title)
    plt.tight_layout()

    return similarity


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def reorder_matrix(base, mat):
    """
    Réordonne les lignes de mat par rapport à base
    en maximisant la similarité cosinus (algorithme hongrois).
    """
    # Matrice des similarités
    sim = cosine_similarity(base, mat)
    # Hungarian assignment (on minimise le coût -> on prend -sim)
    row_ind, col_ind = linear_sum_assignment(-sim)
    # Réordonner mat selon l’assignation trouvée
    mat_reordered = mat[col_ind]
    return mat_reordered


def compare_matrices(matrices):
    """
    matrices : liste de np.ndarray (toutes de même dimension n x d)
    Retourne une matrice (n x n) où (i,j) = moyenne des similarities
    entre ligne i de M0 et ligne j des autres matrices réordonnées.
    """
    base = matrices[0]
    n = base.shape[0]

    # Collecte des similarités
    sims = []
    for mat in matrices[1:]:
        mat_r = reorder_matrix(base, mat)
        sim = cosine_similarity(base, mat_r)
        sims.append(sim)

    # Moyenne sur toutes les matrices (sauf la base)
    mean_sim = np.mean(sims, axis=0)
    return mean_sim


def plot_heatmap(matrix, title="Heatmap des similarités"):
    plt.figure(figsize=(8,6))
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity moyenne")
    plt.title(title)
    plt.xlabel("Lignes des matrices comparées")
    plt.ylabel("Lignes de la matrice de base")


def plot_heatmapW2(matrix, title="Heatmap des similarités"):
    plt.figure(figsize=(8,6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="W2 strength")
    plt.title(title)
    plt.xlabel("W2 columns")
    plt.ylabel("W2 rows")

def pipelineNMFW2(k, r1, r2, r3, r4, r):
    
    DeepLosses = []
    Losses = []
    MULosses = []
    DeepSLosses = []
    DeepRanks = []
    Ranks = []
    MURanks = []
    DeepSRanks = []
    ListeHD = []
    ListeH = []
    ListeHMU = []
    ListeHDS = []
    ListeW21 = []
    ListeW22 = []
    ListeW23 = []
    for i in range(k):
        matricecancer_train, matricecancer_test, labels_train, labels_test = split_matrix_and_labels_80_20(matricecancer, labels)
        A = matricecancer_train
        W21, HD, errorD, rankD, SVGD1D, SVGD2D = Deep_NMF_2W(D1@D2@HD, r1, r, 'random', 'lists', 11500)
        # W23, H, error, rank, SVGD1, SVGD2 = Deep_NMF_2W(D1@D2@HD, r3, r, 'random', 'lists', 11500)
        HMU, errorMU, rankMU, SVGD1MU, SVGD2MU = NMF_for_r_comparison_MU(D1@D2@HD, r,'random', 'lists', 11500)
        # W22, HDS, errorDS, rankDS, SVGD1DS, SVGD2DS = Deep_NMF_2W(D1@D2@HD, r4, r, 'random', 'lists', 11500)
        # H_numpy = H.detach().cpu().numpy()
        HD_numpy = HD.detach().cpu().numpy()
        HMU_numpy = HMU.detach().cpu().numpy()
        # HDS_numpy = HDS.detach().cpu().numpy()
        W21_numpy = W21.detach().cpu().numpy()
        # W22_numpy = W22.detach().cpu().numpy()
        # W23_numpy = W23.detach().cpu().numpy()
        # print(H.shape)
        print(HD.shape)
        print(HMU.shape)
        # print(HDS.shape)
        # if i > 0:
        #     HD_numpy = align_to_reference(ListeHD[0], HD_numpy, "rows")
        #     H_numpy = align_to_reference(ListeH[0], H_numpy, "rows")
        #     H_numpy = align_to_reference(ListeH[0], H_numpy, "rows")
        DeepLosses.append(errorD)
        # Losses.append(error)
        MULosses.append(errorMU)
        DeepRanks.append(rankD)
        # DeepSLosses.append(errorDS)

        # Ranks.append(rank)
        MURanks.append(rankMU)
        # DeepSRanks.append(rankDS)

        ListeHD.append(HD_numpy)
        # ListeH.append(H_numpy)
        ListeHMU.append(HMU_numpy)
        # ListeHDS.append(HDS_numpy)


        ListeW21.append(W21_numpy)
        # ListeW22.append(W22_numpy)
        # ListeW23.append(W23_numpy)

        matches = top_cosmic_matches(H, cosmic_matrix, cosmic_cols)
        print("Correspondances retenues :")
        for h_idx, c_idx, score in matches:
            print(f"Signature H {h_idx} ↔ COSMIC {c_idx} (cosine={score:.4f})")
    MeanLosses = mean_per_epoch(Losses)
    MeanDeepLosses = mean_per_epoch(DeepLosses)
    MeanMULosses = mean_per_epoch(MULosses)
    MeanDeepSLosses = mean_per_epoch(DeepSLosses)
    MeanRanks = mean_per_epoch(Ranks)
    MeanDeepRanks = mean_per_epoch(DeepRanks)
    MeanMURanks = mean_per_epoch(MURanks)
    MeanDeepSRanks = mean_per_epoch(DeepSRanks)

    HDmean = mean_matrices(ListeHD)
    Hmean = mean_matrices(ListeH)
    # matches_list_A = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHD]
    matches_list_B = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeH]
    matches_list_C = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHMU]
    # matches_list_D = [top_cosmic_matches(H, cosmic_matrix, cosmic_cols) for H in ListeHDS]

    # compare_top5_per_method2(matches_list_A, matches_list_B, matches_list_C, matches_list_D, method_names=("Deep NMF " + str(r1), "Deep NMF " + str(r3), "NMF MU", "Deep NMF "+ str(r4)))
    compareW21 = compare_matrices(ListeW21)
    plot_heatmap(compareW21, "Cosine similarity between W2 for rank " + str(r1))
    # compareW22 = compare_matrices(ListeW22)
    # plot_heatmap(compareW22, "Cosine similarity between W2 for rank " + str(r4))
    # compareW23 = compare_matrices(ListeW23)
    # plot_heatmap(compareW23, "Cosine similarity between W2 for rank " + str(r3))

    plot_benchmark(MeanLosses, MeanDeepLosses, MeanMULosses, MeanDeepSLosses, MeanRanks, MeanDeepRanks, MeanMURanks, MeanDeepSRanks, r1, r2, r3, r4, k)
    plot_heatmapW2(ListeW21[0], title="W2 heatmap for rank " + str((r1)))
    # plot_heatmapW2(ListeW22[0], title="W2 heatmap for rank " + str(r4))
    # plot_heatmapW2(ListeW23[0], title="W2 heatmap for rank " + str(r3))
    compare_to_COSMIC(ListeHD[0], "C:/Users/thoma/Python/COSMIC/COSMIC_GRCh37.txt", "Cosine Similarities 3W")
    compare_to_COSMIC(ListeH[0], "C:/Users/thoma/Python/COSMIC/COSMIC_GRCh37.txt", "Cosine Similarities 2W")



def benchmark_H(H, H_t):
    """
    Compare deux matrices H et H_t (format r x n) en alignant H_t sur H
    via l'algorithme hongrois selon la similarité cosinus.
    
    Parameters
    ----------
    H : ndarray, shape (r, n)
        Matrice de référence ("ground truth").
    H_t : ndarray, shape (r, n)
        Matrice estimée (ex: issue de NMF).
    
    Returns
    -------
    similarities : list of float
        Similarités cosinus ligne par ligne (après alignement).
    permutation : list of int
        Indices de réordonnancement des lignes de H_t.
    mean_similarity : float
        Moyenne des similarités cosinus.
    """
    
    # Calcul de la matrice de similarité cosinus
    cos_sim = cosine_similarity(H, H_t)
    
    # Problème d'affectation : on veut maximiser la similarité
    # => on minimise (1 - cos_sim)
    cost_matrix = 1 - cos_sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Réordonnons H_t selon la permutation trouvée
    H_t_aligned = H_t[col_ind]
    
    # Calcul ligne par ligne
    similarities = [
        cosine_similarity(H[i].reshape(1, -1), H_t_aligned[i].reshape(1, -1))[0, 0]
        for i in range(H.shape[0])
    ]
    
    return similarities, col_ind, np.mean(similarities)

def plot_similarities_distributions(H, H_t_list, bins=100, title="NMF"):
    """
    Trace un histogramme des similarités cosinus pour chaque ligne de H,
    en utilisant une liste de matrices H_t issues de plusieurs runs (ex: NMF).
    
    Parameters
    ----------
    H : ndarray, shape (r, n)
        Matrice de référence ("ground truth").
    H_t_list : list of ndarray
        Liste de matrices H_t (chacune de shape (r, n)).
    bins : int
        Nombre de bins pour les histogrammes.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.optimize import linear_sum_assignment

    r = H.shape[0]
    all_similarities = [[] for _ in range(r)]
    
    # On benchmark chaque H_t de la liste
    for H_t in H_t_list:
        cos_sim = cosine_similarity(H, H_t)
        cost_matrix = 1 - cos_sim
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        H_t_aligned = H_t[col_ind]
        
        sims = [
            cosine_similarity(H[i].reshape(1, -1), H_t_aligned[i].reshape(1, -1))[0, 0]
            for i in range(r)
        ]
        for i, s in enumerate(sims):
            all_similarities[i].append(s)
    
    # Plot : un histogramme par ligne
    fig, axes = plt.subplots(r, 1, figsize=(6, 3*r), sharex=True)
    if r == 1:
        axes = [axes]
    for i, sims in enumerate(all_similarities):
        axes[i].hist(sims, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
        axes[i].set_title(f"Row {i} : cosines similarities between H ground truth and H " + title)
        axes[i].set_ylabel("Frequency")
    axes[-1].set_xlabel("Cosine Similarities")
    
    plt.tight_layout()
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_similarities_bar(H, H_t_list, title="NMF"):
    """
    Affiche les similarités cosinus pour chaque ligne de H
    avec uniquement les points + barres d'erreur IC95 (pas de barres pleines).
    """
    r = H.shape[0]
    all_similarities = [[] for _ in range(r)]
    
    # Benchmark chaque H_t
    for H_t in H_t_list:
        sims, _, _ = benchmark_H(H, H_t)
        for i, s in enumerate(sims):
            all_similarities[i].append(s)
    
    means = [np.mean(sims) for sims in all_similarities]
    cis = [stats.t.interval(0.95, len(sims)-1, loc=np.mean(sims), scale=stats.sem(sims))
           if len(sims) > 1 else (sims[0], sims[0]) 
           for sims in all_similarities]
    
    yerr = np.array([[m - ci[0], ci[1] - m] for m, ci in zip(means, cis)]).T
    
    # Plot uniquement les points et barres d'erreur
    x = np.arange(r)
    plt.figure(figsize=(8, 5))
    
    # Points individuels (avec petit jitter horizontal)
    for i, sims in enumerate(all_similarities):
        jitter = (np.random.rand(len(sims)) - 0.5) * 0.2
        plt.scatter([i + j for j in jitter], sims, color="steelblue", alpha=0.7, zorder=10)
    
    # Ajout de la moyenne + IC95 en ligne verticale
    plt.errorbar(x, means, yerr=yerr, fmt='o', color='red', capsize=5, markersize=8, label="Mean ± 95% CI")
    
    plt.xticks(x, [f"Ligne {i}" for i in range(r)])
    plt.ylabel("Cosine similarities")
    plt.title("Cosine Similarities between H from " + title + " and H ground truth(points + 95% CI)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
import numpy as np

def gaussian_noise_like(A, mean=0.0, std=1.0, seed=None):
    """
    Génère un bruit gaussien de la même taille que A.

    Parameters
    ----------
    A : np.ndarray
        Matrice d'entrée.
    mean : float
        Moyenne du bruit gaussien.
    std : float
        Écart-type du bruit gaussien.
    seed : int or None
        Pour reproduire les résultats.

    Returns
    -------
    noise : np.ndarray
        Bruit gaussien de même shape que A.
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.normal(loc=mean, scale=std, size=A.shape)

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)

def plot_similarities_distributions_W2(H, H_t_list, bins=100, title="NMF"):
    """
    Trace un histogramme des similarités cosinus pour chaque ligne de H,
    en utilisant une liste de matrices H_t issues de plusieurs runs (ex: NMF).
    Cosine similarities garanties dans [0,1].
    """
    def normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, 1e-12, None)

    # Normalisation
    H_norm = normalize_rows(H)
    H_t_list_norm = [normalize_rows(H_t) for H_t in H_t_list]

    r = H.shape[0]
    all_similarities = [[] for _ in range(r)]

    # Boucle sur chaque H_t
    for H_t_norm in H_t_list_norm:
        # Matrice de similarité cosinus r x r
        cos_sim_matrix = cosine_similarity(H_norm, H_t_norm)
        cost_matrix = 1 - cos_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        H_t_aligned = H_t_norm[col_ind]

        # Cosine similarities ligne par ligne, clip pour rester dans [0,1]
        sims = [
            np.clip(
                cosine_similarity(H_norm[i].reshape(1, -1),
                                  H_t_aligned[i].reshape(1, -1))[0, 0],
                0.0, 1.0
            )
            for i in range(r)
        ]
        for i, s in enumerate(sims):
            all_similarities[i].append(s)

    # Plot : histogrammes par ligne
    fig, axes = plt.subplots(r, 1, figsize=(6, 3*r), sharex=True)
    if r == 1:
        axes = [axes]
    for i, sims in enumerate(all_similarities):
        axes[i].hist(sims, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
        axes[i].set_title(f"Row {i} : cosine similarities between W2 ground truth and W2 {title}")
        axes[i].set_ylabel("Frequency")
    axes[-1].set_xlabel("Cosine Similarities between W2 rows")

    plt.tight_layout()
def align_and_compute_similarities(H, H_t_list):
    """
    Aligne chaque matrice de H_t_list sur H via l'algorithme hongrois
    (basé sur la similarité cosinus), puis calcule les similarités ligne par ligne.
    
    Parameters
    ----------
    H : ndarray, shape (r, n)
        Matrice de référence.
    H_t_list : list of ndarray
        Liste de matrices à comparer, chacune de shape (r, n).
    
    Returns
    -------
    all_similarities : list of ndarray
        Liste de vecteurs (taille r) contenant les similarités cosinus
        ligne par ligne entre H et chaque H_t aligné.
    permutations : list of ndarray
        Liste des permutations (indices de réordonnancement des lignes de H_t).
    """
    r = H.shape[0]
    all_similarities = []
    permutations = []
    
    for H_t in H_t_list:
        # Matrice des similarités cosinus r×r
        cos_sim_matrix = cosine_similarity(H, H_t)
        
        # Alignement par l'algorithme hongrois
        cost_matrix = 1 - cos_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Réordonner H_t
        H_t_aligned = H_t[col_ind]
        permutations.append(col_ind)
        
        # Similarités ligne par ligne (après alignement)
        sims = np.array([
            np.clip(cosine_similarity(H[i].reshape(1, -1),
                                      H_t_aligned[i].reshape(1, -1))[0, 0],
                    0.0, 1.0)
            for i in range(r)
        ])
        
        all_similarities.append(sims)
    
    return all_similarities


def best_fitting_distributions(data, top=3, distributions=None):
    """
    Trouve les distributions qui s'ajustent le mieux à une série de données.
    
    Parameters
    ----------
    data : array-like
        Liste ou array de données (ex: cosine similarities).
    top : int
        Nombre de distributions à retourner.
    distributions : list of str
        Liste des distributions scipy.stats à tester.
        Si None, on prend un set de distributions usuelles pour des données [0,1].
    
    Returns
    -------
    results : list of tuples
        [(nom_distribution, AIC, params), ...] trié du meilleur au pire.
    """
    data = np.array(data)
    
    if distributions is None:
        # Distributions adaptées à [0,1]
        distributions = ["beta", "uniform", "triang", "logistic", "norm"]
    
    results = []
    
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            # Estimation des paramètres
            params = dist.fit(data)
            
            # Log-vraisemblance
            loglik = np.sum(dist.logpdf(data, *params))
            
            # Nombre de paramètres
            k = len(params)
            
            # Critère d'info Akaike
            aic = 2 * k - 2 * loglik
            
            results.append((dist_name, aic, params))
        except Exception as e:
            # Certaines distributions peuvent planter sur les bornes
            continue
    
    # Classement par AIC (plus petit = meilleur)
    results.sort(key=lambda x: x[1])
    
    return results[:top]



def plot_similarities_bar_W2(H, H_t_list, title="NMF"):
    """
    Affiche les similarités cosinus pour chaque ligne de H
    avec uniquement les points + barres d'erreur IC95 (pas de barres pleines).
    """
    r = H.shape[0]
    all_similarities = [[] for _ in range(r)]
    
    # Benchmark chaque H_t
    for H_t in H_t_list:
        sims, _, _ = benchmark_H(H, H_t)
        for i, s in enumerate(sims):
            all_similarities[i].append(s)
    
    means = [np.mean(sims) for sims in all_similarities]
    cis = [stats.t.interval(0.95, len(sims)-1, loc=np.mean(sims), scale=stats.sem(sims))
           if len(sims) > 1 else (sims[0], sims[0]) 
           for sims in all_similarities]
    
    yerr = np.array([[m - ci[0], ci[1] - m] for m, ci in zip(means, cis)]).T
    
    # Plot uniquement les points et barres d'erreur
    x = np.arange(r)
    plt.figure(figsize=(8, 5))
    
    # Points individuels (avec petit jitter horizontal)
    for i, sims in enumerate(all_similarities):
        jitter = (np.random.rand(len(sims)) - 0.5) * 0.2
        plt.scatter([i + j for j in jitter], sims, color="steelblue", alpha=0.7, zorder=10)
    
    # Ajout de la moyenne + IC95 en ligne verticale
    plt.errorbar(x, means, yerr=yerr, fmt='o', color='red', capsize=5, markersize=8, label="Mean ± 95% CI")
    
    plt.xticks(x, [f"Ligne {i}" for i in range(r)])
    plt.ylabel("Cosine similarities between W2 rows")
    plt.title("Cosine Similarities between W2 from " + title + " and W2 ground truth(points + 95% CI)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
def plot_aligned_similarities(H, H_t_list, bins=50, title="Cosine Similarities"):
    """
    Aligne chaque matrice de H_t_list sur H (via l'algo hongrois avec cosine similarity),
    calcule les similarités ligne par ligne, puis trace les distributions.
    
    Parameters
    ----------
    H : ndarray, shape (r, n)
        Matrice de référence.
    H_t_list : list of ndarray
        Liste de matrices à comparer, chacune de shape (r, n).
    bins : int
        Nombre de bins pour les histogrammes.
    title : str
        Titre de la figure matplotlib.
    
    Returns
    -------
    all_similarities : ndarray, shape (len(H_t_list), r)
        Matrice des similarités cosinus : chaque ligne correspond à un H_t aligné,
        chaque colonne à une ligne de H.
    permutations : list of ndarray
        Liste des permutations appliquées à chaque H_t.
    """
    r = H.shape[0]
    all_similarities = []
    permutations = []

    for H_t in H_t_list:
        # Matrice r×r des similarités
        cos_sim_matrix = cosine_similarity(H, H_t)
        
        # Alignement hongrois
        cost_matrix = 1 - cos_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Réordonner H_t
        H_t_aligned = H_t[col_ind]
        permutations.append(col_ind)
        
        # Similarités ligne par ligne
        sims = np.array([
            np.clip(cosine_similarity(H[i].reshape(1, -1),
                                      H_t_aligned[i].reshape(1, -1))[0, 0],
                    0.0, 1.0)
            for i in range(r)
        ])
        all_similarities.append(sims)

    all_similarities = np.array(all_similarities)  # shape (len(H_t_list), r)

    # --- Plot ---
    fig, axes = plt.subplots(r, 1, figsize=(7, 3*r), sharex=True)
    if r == 1:
        axes = [axes]
    
    for i in range(r):
        sims = all_similarities[:, i]
        axes[i].hist(sims, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
        axes[i].axvline(np.mean(sims), color="red", linestyle="--", label="Mean")
        axes[i].set_title(f"Row {i} : cosine similarities")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
    
    axes[-1].set_xlabel("Cosine similarity")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return all_similarities
def plot_best_distributions_table(all_similarities, top=3, distributions=None):
    """
    Pour chaque ligne (colonne) de all_similarities, teste plusieurs distributions
    et affiche un tableau matplotlib avec les noms des meilleures distributions.

    Parameters
    ----------
    all_similarities : ndarray, shape (n_runs, n_rows)
        Matrice des similarités ligne par ligne (comme sortie de plot_aligned_similarities).
    top : int
        Nombre de distributions les plus probables à afficher.
    distributions : list of str
        Liste de distributions scipy.stats à tester. Si None, teste ['beta','uniform','triang','norm','logistic'].
    """
    n_rows = all_similarities.shape[1]

    if distributions is None:
        distributions = ['beta', 'uniform', 'triang', 'norm', 'logistic']

    best_dists_names = []

    for i in range(n_rows):
        data = all_similarities[:, i]
        results = []

        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(data)
                loglik = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2*k - 2*loglik
                results.append((dist_name, aic))
            except:
                continue

        results.sort(key=lambda x: x[1])
        top_names = [name for name, _ in results[:top]]
        best_dists_names.append(top_names)

    # --- Affichage sous forme de tableau matplotlib ---
    fig, ax = plt.subplots(figsize=(8, 0.6*n_rows + 1))
    ax.axis('off')

    # Créer un tableau : lignes = lignes de H, colonnes = top distributions
    table_data = []
    for i, names in enumerate(best_dists_names):
        row = [f"Row {i}"] + names
        table_data.append(row)

    col_labels = ["Row"] + [f"Best {i+1}" for i in range(top)]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title("Most probable distributions per row", fontsize=14)
    plt.tight_layout()

    return best_dists_names

def plot_aligned_similarities_with_fit(
    H, H_t_list, bins=50, title="Cosine Similarities",
    distributions=None
):
    """
    Aligne chaque matrice de H_t_list sur H, calcule les similarités, 
    puis trace histogrammes + points observés + meilleure distribution ajustée.
    
    Parameters
    ----------
    H : ndarray, shape (r, n)
        Matrice de référence.
    H_t_list : list of ndarray
        Liste de matrices à comparer, chacune de shape (r, n).
    bins : int
        Nombre de bins pour les histogrammes.
    title : str
        Titre de la figure matplotlib.
    distributions : list of str
        Liste de distributions scipy.stats à tester. 
        Par défaut : ['beta','uniform','triang','norm','logistic'].
    
    Returns
    -------
    all_similarities : ndarray, shape (len(H_t_list), r)
        Matrice des similarités cosinus.
    best_fits : list of tuple
        Pour chaque ligne, (nom_distribution, params).
    permutations : list of ndarray
        Liste des permutations appliquées à chaque H_t.
    """
    r = H.shape[0]
    all_similarities = []
    permutations = []

    # --- Étape 1 : alignement + similarités ---
    for H_t in H_t_list:
        cos_sim_matrix = cosine_similarity(H, H_t)
        cost_matrix = 1 - cos_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        H_t_aligned = H_t[col_ind]
        permutations.append(col_ind)

        sims = np.array([
            np.clip(cosine_similarity(H[i].reshape(1, -1),
                                      H_t_aligned[i].reshape(1, -1))[0, 0],
                    0.0, 1.0)
            for i in range(r)
        ])
        all_similarities.append(sims)

    all_similarities = np.array(all_similarities)  # shape (len(H_t_list), r)

    if distributions is None:
        distributions = ['beta', 'uniform', 'triang', 'norm', 'logistic']

    best_fits = []

    # --- Étape 2 : tracés ---
    fig, axes = plt.subplots(r, 1, figsize=(7, 3*r), sharex=True)
    if r == 1:
        axes = [axes]

    x = np.linspace(0, 1, 500)

    for i in range(r):
        data = all_similarities[:, i]

        # Histogramme
        axes[i].hist(data, bins=bins, alpha=0.6, color="steelblue",
                     edgecolor="black", density=True)

        # Rug plot = points individuels
        axes[i].plot(data, np.zeros_like(data)-0.01, '|', color="black")

        # Sélection de la meilleure distribution par AIC
        results = []
        for dist_name in distributions:
            dist = getattr(stats, dist_name)
            try:
                params = dist.fit(data)
                loglik = np.sum(dist.logpdf(data, *params))
                k = len(params)
                aic = 2*k - 2*loglik
                results.append((dist_name, aic, params))
            except Exception:
                continue

        results.sort(key=lambda x: x[1])
        best_name, _, best_params = results[0]
        best_fits.append((best_name, best_params))

        # Tracer la densité de la meilleure loi
        best_dist = getattr(stats, best_name)
        pdf_vals = best_dist.pdf(x, *best_params)
        axes[i].plot(x, pdf_vals, 'r-', lw=2, label=f"{best_name} fit")

        axes[i].set_title(f"Row {i} : cosine similarities")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    axes[-1].set_xlabel("Cosine similarity")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

matricecancer = X

def BenchmarkDNMF(epochs, r1, r2):
    DeepLosses = []
    Losses = []
    MULosses = []
    DeepSLosses = []
    DeepRanks = []
    Ranks = []
    MURanks = []
    DeepSRanks = []
    ListeHDeep = []
    ListeH = []
    ListeHMU = []
    ListeHDS = []
    ListeW21 = []
    ListeW22 = []
    ListeW23 = []
    for i in range(epochs):
        matricecancer_train, matricecancer_test, labels_train, labels_test = split_matrix_and_labels_80_20(matricecancer, labels, i)
        A = gaussian_noise_like(matricecancer, 20, 8, i)
        print(A)
        W21, HDeep, errorD, rankD, SVGD1D, SVGD2D = Deep_NMF_2W(matricecancer, r1, r2, 'random', 'lists', 3000)
        HMU, errorMU, rankMU, SVGD1MU, SVGD2MU = NMF_for_r_comparison_MU(matricecancer, r2,'random', 'lists', 3000)
        HDeep_numpy = HDeep.detach().cpu().numpy()
        HMU_numpy = HMU.detach().cpu().numpy()
        W21_numpy = W21.detach().cpu().numpy()

        print(HD.shape)
        print(HMU.shape)

        DeepLosses.append(errorD)
        MULosses.append(errorMU)
        
        DeepRanks.append(rankD)
        MURanks.append(rankMU)

        ListeHDeep.append(HDeep_numpy)
        ListeHMU.append(HMU_numpy)


        ListeW21.append(W21_numpy)
        # ListeW22.append(W22_numpy)
        # ListeW23.append(W23_numpy)


        # MeanLosses = mean_per_epoch(Losses)
        
    MeanDeepLosses = mean_per_epoch(DeepLosses)
    MeanMULosses = mean_per_epoch(MULosses)
    # MeanDeepSLosses = mean_per_epoch(DeepSLosses)
    # MeanRanks = mean_per_epoch(Ranks)
    MeanDeepRanks = mean_per_epoch(DeepRanks)
    MeanMURanks = mean_per_epoch(MURanks)
    # MeanDeepSRanks = mean_per_epoch(DeepSRanks)
    plot_benchmark(MeanMULosses,MeanDeepLosses, MeanDeepRanks, MeanMURanks, r1, r2, epochs)
    plot_similarities_distributions(ListeHDeep[0], ListeHDeep, 250, "Deep NMF")
    plot_similarities_distributions(ListeHMU[0], ListeHMU, 250, "NMF MU")
    plot_similarities_bar(ListeHDeep[0], ListeHDeep, "Deep NMF")
    plot_similarities_bar(ListeHMU[0], ListeHMU, "NMF MU")
    plot_similarities_distributions_W2(ListeW21[0], ListeW21, 250, "Deep NMF")
    plot_similarities_bar_W2(ListeW21[0], ListeW21, "Deep NMF")
    a = plot_aligned_similarities(ListeW21[0], ListeW21, 100, "Deep NMF")
    plot_best_distributions_table(a)
    plot_aligned_similarities_with_fit(ListeW21[0], ListeW21, 100, "Deep NMF")

def save_nmf_components(H, output_dir="nmf_components", image_shape=(3,64,64)):
    """
    Sauvegarde les composantes NMF comme images.
    
    Args:
        H (np.ndarray): matrice NMF (num_components, D)
        output_dir (str): dossier pour sauvegarder les images
        image_shape (tuple): forme originale des images (C,H,W)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_components = H.shape[0]
    
    for i, comp in enumerate(H):
        img_array = comp.reshape(image_shape).transpose(1,2,0)  # C,H,W -> H,W,C
        plt.figure()
        plt.imshow(img_array)
        plt.axis("off")
        path = os.path.join(output_dir, f"nmf_component_{i+1}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved NMF component: {path}")
    
    print(f"✅ {num_components} NMF component(s) saved to '{output_dir}'")


plt.show()
W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 = NMF_for_r_comparison(X[100:150, :], 10, 'random', 'all', 3500)

plot_nmf_results(W, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,64,64), X)
save_nmf_components(H)
# Transformation : juste redimensionnement pour visualisation
transform = transforms.Compose([
    transforms.Resize((64, 64))  # redimensionnement seulement
])

# Charger le dataset
dataset = datasets.Flowers102(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

# Afficher 5 images
for i in range(5):
    img, label = dataset[i]  # img est un PIL Image ici
    img_np = np.array(img)   # PIL -> numpy array HxWxC
    plt.figure()
    plt.imshow(img_np)
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()
