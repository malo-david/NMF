import os
import matplotlib.pyplot as plt
from datetime import datetime

# Imports depuis nos modules
from data_loader import load_mnist
from nmf_models import Deep_NMF_2W, NMF_for_r_comparison_MU
from visualizations import (
    plot_nmf_results, 
    plot_H_signatures, 
    plot_mnist_reconstruction,
    plot_mnist_reconstruction_nmf
)

# ------------------------- CHARGEMENT DES DONNÉES -------------------------
X, dataset = load_mnist(resize=(28, 28))
X_train = X[:1000, :]

# ------------------------- ENTRAÎNEMENT DES MODÈLES -------------------------

# Deep NMF
W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 = Deep_NMF_2W(
    X_train, 
    r1=20, 
    r2=10, 
    init='random', 
    end='all', 
    epochs=1000
)

# NMF avec Multiplicative Updates
# W_N, H_N, errorsGD_N, rankGD_N, nuclearrankGD_N, SVGD1_N, SVGD2_N = NMF_for_r_comparison_MU(
#     X_train, 
#     r=10, 
#     init='random', 
#     end='all', 
#     epochs=300
# )

# ------------------------- VISUALISATIONS -------------------------

# Résultats NMF
plot_nmf_results(W1@W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2, (3,28,28), X)
# plot_nmf_results(W_N, H_N, errorsGD_N, rankGD_N, nuclearrankGD_N, SVGD1_N, SVGD2_N, (3,28,28), X)

# Signatures H
plot_H_signatures(H, "deep", image_shape=(28, 28), n_show=9)
# plot_H_signatures(H_N, "MU", image_shape=(28, 28), n_show=9)

# Reconstructions
plot_mnist_reconstruction(X, W1, W2, H, index=0, image_shape=(28, 28))
# plot_mnist_reconstruction_nmf(X, W_N, H_N, index=0, image_shape=(28, 28))

# ------------------------- SAUVEGARDE -------------------------

# Dossier de base (à adapter selon ton système)
base_dir = r"./data"

# Dossier unique par exécution
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(base_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

# Récupère toutes les figures ouvertes et sauvegarde automatiquement
for i, fig_num in enumerate(plt.get_fignums(), 1):
    fig = plt.figure(fig_num)
    path = os.path.join(save_dir, f"figure_{i}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")

# Affiche tout à la fin
#plt.show()
