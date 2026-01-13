import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_nmf_results(W, H, errorsGD, rankGD, nuclearrankGD,
                     SVGD1, SVGD2, image_shape=(64, 64), X=None, epochs_metrics=None):
    """
    Affiche les heatmaps W et H + courbes de suivi
    """
    # Convertir torch -> numpy si nécessaire
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    if X is not None and isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # FIGURE 1 : Heatmaps W et H
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

    # FIGURE 2 : Courbes de suivi
    plt.figure(figsize=(12, 8))

    plots = [errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2]
    titles = ["errorsGD", "rankGD", "nuclearrankGD", "SVGD1", "SVGD2"]

    for i, (t, y) in enumerate(zip(titles, plots)):
        plt.subplot(3, 2, i + 1)
        if epochs_metrics is None:
            if t == "errorsGD":
                plt.semilogy(y)             # log scale pour l'erreur
                plt.xlim(right=30000)

            else:    
                plt.plot(y)                    # fallback (pas exact)
                plt.xlim(right=30000)

        else:
            # x = epochs où les métriques ont été calculées
            x = np.asarray(epochs_metrics)

            # ne garder que les points après 20000 epochs
            mask = x <= 30000

            if t == "errorsGD":
                plt.semilogy(x[mask], np.asarray(y)[mask])  # log scale pour l'erreur
            else:
                plt.plot(x[mask], np.asarray(y)[mask])
            plt.xlim(right=30000)

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


def plot_H_signatures(H, title, image_shape=(28, 28), n_show=5):
    """
    Affiche n_show signatures (lignes) de H sous forme d'images MNIST
    H : (r2, D)
    """
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

    plt.suptitle("Signatures NMF (lignes de H)" + title, fontsize=14)
    plt.tight_layout()


def plot_mnist_reconstruction(A, W1, W2, H, index=0, image_shape=(28, 28)):
    """
    Compare une image MNIST originale et sa reconstruction NMF (Deep NMF)
    """
    # Conversion numpy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(W1, torch.Tensor):
        W1 = W1.detach().cpu().numpy()
    if isinstance(W2, torch.Tensor):
        W2 = W2.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    # Original
    x_true = A[index]

    # Reconstruction NMF
    W_eff = W1 @ W2
    coeffs = W_eff[index]
    x_rec = coeffs @ H

    # Reshape
    img_true = x_true.reshape(image_shape)
    img_rec = x_rec.reshape(image_shape)
    img_err = np.abs(img_true - img_rec)

    # Normalisation sécurité
    img_rec = np.clip(img_rec, 0, 1)
    img_err = img_err / img_err.max() if img_err.max() > 0 else img_err

    # Plot
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

    plt.suptitle(f"MNIST – reconstruction NMF (index={index})", fontsize=14)
    plt.tight_layout()


def plot_mnist_reconstruction_nmf(A, W, H, index=0, image_shape=(28, 28)):
    """
    Compare une image MNIST originale et sa reconstruction avec une NMF simple
    A : (N, D) images originales
    W : (N, r) coefficients NMF
    H : (r, D) signatures NMF
    """
    # Conversion numpy si nécessaire
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    # Image originale
    x_true = A[index]

    # Reconstruction NMF simple
    x_rec = W[index] @ H

    # Reshape
    img_true = x_true.reshape(image_shape)
    img_rec = np.clip(x_rec.reshape(image_shape), 0, 1)
    img_err = np.abs(img_true - img_rec)
    img_err = img_err / img_err.max() if img_err.max() > 0 else img_err

    # Plot
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

    plt.suptitle(f"MNIST – reconstruction NMF simple (index={index})", fontsize=14)
    plt.tight_layout()
