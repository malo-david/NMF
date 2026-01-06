# Projet NMF - Advanced Machine Learning

## Description du projet

Ce projet implémente plusieurs variantes de NMF (Non-negative Matrix Factorization) pour l'analyse et la reconstruction d'images (MNIST, Flowers102). Un des objectifs est d'étudier le **low rank bias** dans ce contexte - un phénomène où les modèles apprennent d'abord les caractéristiques simples/basses fréquences avant les détails complexes

### Objectifs scientifiques
- Implémentation de NMF avec descente de gradient et multiplicative updates
- Deep NMF (factorisation multi-couches)
- Étude du low rank bias via l'effective rank
- Reconstruction et génération d'images
- Early stopping basé sur le plateau de l'effective rank

---

## Structure du projet

```
nmf_project/
├── data_loader.py       # Chargement des datasets (MNIST, Flowers102)
├── metrics.py           # Calcul des métriques (effective rank, nuclear norm, etc.)
├── nmf_models.py        # Implémentations des modèles NMF
├── visualizations.py    # Fonctions de visualisation et plots
├── experiments.py       # Script principal d'exécution
└── readme_nmf.md        # Ce fichier
```

### Description des fichiers

#### `data_loader.py`
Gère le chargement et la préparation des datasets pour NMF.
- **`load_mnist(resize=(28, 28))`** : Charge MNIST et retourne la matrice X aplatie
- **`load_flowers102(resize=(64, 64))`** : Charge Flowers102 (dataset de fleurs)

#### `metrics.py`
Fonctions de calcul des métriques pour analyser les factorisation.
- **`exp_effective_rank_torch(A)`** : Calcule l'effective rank exponentiel (mesure de diversité des valeurs singulières)
- **`nuclear_over_operator_norm_torch(A)`** : Ratio norme nucléaire / norme opérateur
- **`cosine_separation_loss(H)`** : Pénalité pour encourager l'orthogonalité des lignes de H

#### `nmf_models.py`
Implémentations des différentes variantes de NMF.
- **`Deep_NMF_2W(A, r1, r2, init, end, epochs)`** : Deep NMF avec 2 couches (A ≈ W1 @ W2 @ H)
- **`NMF_for_r_comparison(A, r, init, end, epochs)`** : NMF classique avec descente de gradient (Adam)
- **`NMF_for_r_comparison_MU(A, r, init, end, epochs)`** : NMF avec Multiplicative Updates (Lee & Seung)

#### `visualizations.py`
Fonctions de visualisation des résultats.
- **`plot_nmf_results(W, H, ...)`** : Heatmaps des matrices + courbes de suivi (erreur, rank, etc.)
- **`plot_H_signatures(H, title, ...)`** : Affiche les signatures (lignes de H) comme images
- **`plot_mnist_reconstruction(A, W1, W2, H, ...)`** : Compare image originale vs reconstruction (Deep NMF)
- **`plot_mnist_reconstruction_nmf(A, W, H, ...)`** : Compare image originale vs reconstruction (NMF simple)

#### `experiments.py`
Script principal qui orchestre l'entraînement et les visualisations.

---

## Installation et utilisation

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn tqdm scikit-learn scipy torch torchvision
```

### Lancement rapide

1. **Cloner ou copier tous les fichiers dans le même dossier**

2. **Lancer le script principal :**
```bash
python experiments.py
```

3. **Les résultats seront sauvegardés** dans un dossier horodaté :
```
C:\Users\thoma\Desktop\COde\NMF Graphiques\Advanced ML\20250106_143025\
```

---

## Configuration et personnalisation

### Changer le dataset

Dans `experiments.py`, remplacer :
```python
X, dataset = load_mnist(resize=(28, 28))
```

Par :
```python
X, dataset = load_flowers102(resize=(64, 64))
```

### Modifier les hyperparamètres

```python
# Deep NMF
W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 = Deep_NMF_2W(
    X[100:350, :],   # Sous-ensemble des données
    r1=20,           # Rang intermédiaire
    r2=10,           # Rang final
    init='random',   # 'random', 'eye', ou 'ssvd'
    end='all',       # 'all', 'matrix', ou 'lists'
    epochs=10000     # Nombre d'époques
)
```

### Choisir l'initialisation

- **`'random'`** : Initialisation aléatoire (par défaut)
- **`'eye'`** : Initialisation avec matrices identité
- **`'ssvd'`** : Initialisation avec SVD non-négative (nécessite implémentation)

### Choisir le type de retour

- **`'matrix'`** : Retourne seulement W, H (ou W1, W2, H)
- **`'lists'`** : Retourne les métriques de suivi
- **`'all'`** : Retourne matrices + métriques (recommandé)

---

## Métriques suivies pendant l'entraînement

### Erreur de reconstruction
- **`errorsGD`** : Erreur relative ||A - WH||²_F / ||A||²_F
- **`fullerrorsGD`** : Erreur absolue ||A - WH||²_F

### Analyse du rang
- **`rankGD`** : Effective rank (mesure la "diversité" des composantes)
- **`nuclearrankGD`** : Ratio norme nucléaire / norme opérateur

### Valeurs singulières
- **`SVGD1`** : Plus grande valeur singulière de WH
- **`SVGD2`** : Deuxième plus grande valeur singulière

### Interpréter les courbes

- **Effective rank qui plateau** → Moment idéal pour early stopping (low rank bias)
- **Erreur qui stagne** → Convergence atteinte
- **Valeurs singulières décroissantes** → Compression de l'information

---

## Étudier le low rank bias

### Concept

Le **low rank bias** est l'observation empirique que les réseaux de neurones (et NMF) apprennent d'abord les caractéristiques simples/basses fréquences avant les détails fins.

### Comment l'observer ?

1. **Suivre l'effective rank pendant l'entraînement**
2. **Repérer le plateau** de l'effective rank
3. **Faire un early stopping à ce moment**
4. **Visualiser les signatures H** → elles devraient capturer les structures globales/simples

---

## Visualisations disponibles

### Heatmaps des matrices
- Visualise la structure de W et H
- Permet de voir les patterns appris

### Courbes de suivi
- Erreur, effective rank, nuclear rank
- Valeurs singulières au cours du temps

### Signatures NMF
- Affiche les lignes de H comme images
- Révèle les "composantes" apprises

### Reconstructions
- Compare image originale vs reconstruction
- Carte d'erreur pour voir les zones mal reconstruites

---

## Contributeurs

- Thomas Lambelin
- Malo David
- Maxime Chansat