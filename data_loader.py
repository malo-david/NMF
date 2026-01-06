import numpy as np
from torchvision import datasets, transforms

def load_mnist(resize=(28, 28)):
    """
    Charge MNIST et retourne la matrice X pour NMF
    """
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    X_list = []
    for img, _ in dataset:
        arr = img.numpy().reshape(-1)
        X_list.append(arr)
    
    X = np.stack(X_list)
    return X, dataset


def load_flowers102(resize=(64, 64)):
    """
    Charge Flowers102 et retourne la matrice X pour NMF
    """
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    
    dataset = datasets.Flowers102(
        root="./data",
        split="train",
        download=True,
        transform=transform
    )
    
    X_list = []
    for img, _ in dataset:
        arr = img.numpy().reshape(-1)
        X_list.append(arr)
    
    X = np.stack(X_list)
    return X, dataset
