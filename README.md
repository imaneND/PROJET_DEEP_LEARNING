# 🧠 Projet Deep Learning —  4IAD G2

> **Conception, implémentation, comparaison et analyse critique de modèles de deep learning**  
> Module : Deep Learning | Filière : 4ème année — Intelligence Artificielle et Data  
> Année universitaire : 2025–2026 | Encadrant : Mosaab Batal

👩‍💻 **Réalisé par:**

Imane Nadif




---

##  Idée Générale du Projet

Ce projet couvre trois grandes familles d'architectures de deep learning, chacune adaptée à un type de données spécifique :

| Partie | Architecture | Type de données | Dataset |
|--------|-------------|-----------------|---------|
| I | MLP | Tabulaires | Wine Quality (178 exemples, 13 features, 3 classes) |
| II | CNN + Dérivés (LeNet, AlexNet, VGG) | Images | MNIST (70 000 images 28×28) |
| III | RNN / LSTM / GRU / Transformer | Séquences | Tatoeba fra-eng (5 000 paires) |

Le fil conducteur est l'**inductive bias** : comment chaque architecture exploite la structure géométrique, locale ou temporelle des données, en s'appuyant sur un même principe fondateur — descente de gradient et rétropropagation.

---

##  Résultats Clés

| Modèle | Dataset | Métrique | Score |
|--------|---------|----------|-------|
| MLP 2 — [64, 32] | Wine Quality | Test Accuracy | **100%** |
| LeNet | MNIST | Test Accuracy | 98.89% |
| AlexNet simplifié | MNIST | Test Accuracy | **99.30%** |
| VGG simplifié | MNIST | Test Accuracy | 99.28% |
| Transformer | Tatoeba | Perplexité | **6.20** |
| GRU | Tatoeba | Perplexité | 7.71 |

---

## 📁 Structure du Repository

```
PROJET_DEEP_LEARNING/
│
├──  Partie_I.ipynb            # MLP sur Wine Quality
├──  Partie_II.ipynb           # CNN (LeNet, AlexNet, VGG) sur MNIST
├──  Partie_III.ipynb          # RNN / LSTM / GRU / Transformer sur Tatoeba
│
├── 📂 data/MNIST/raw/           # Dataset MNIST brut (images et labels)
│   ├── train-images-idx3-ubyte(.gz)
│   ├── train-labels-idx1-ubyte(.gz)
│   ├── t10k-images-idx3-ubyte(.gz)
│   └── t10k-labels-idx1-ubyte(.gz)
│
├──  LeNet.pth                 # Poids sauvegardés — LeNet
├──  AlexNet_simplifié.pth     # Poids sauvegardés — AlexNet
├──  VGG_simplifié.pth         # Poids sauvegardés — VGG
│
├──  fra-eng.zip               # Dataset Tatoeba français-anglais (compressé)
├──  fra.txt                   # Corpus texte brut français-anglais
├──  _about.txt                # Description du dataset Tatoeba
│
├──  Synthése_Projet.pdf       # Rapport de synthèse complet (exporté depuis Overleaf)
└──  README.md                 # Ce fichier
```

---

##  Détail des Parties

### Partie I — MLP et Ingénierie PyTorch

- Implémentation de 5 architectures MLP sur **Wine Quality**
- Comparaison de 3 stratégies d'initialisation : Gaussienne, Constante, Xavier
- **Best Model : MLP 2 — [64, 32]** avec 3 075 paramètres → Test Accuracy = **100%**
- Techniques : `nn.Module`, `state_dict`, Dropout, DataLoader, StandardScaler

### Partie II — CNN et Architectures Dérivées

- Implémentation de **LeNet**, **AlexNet simplifié**, **VGG simplifié** sur **MNIST**
- Calculs manuels de corrélation croisée 2D et tailles de feature maps
- Visualisation des feature maps (conv1 et conv2)
- Comparaison MLP vs CNN : AlexNet atteint **99.30%** avec moins de paramètres que le MLP

### Partie III — RNN / LSTM / GRU / Transformer

- Pipeline complet de traduction **français → anglais** sur Tatoeba
- Tokenisation, vocabulaires, tokens spéciaux (`<pad>`, `<bos>`, `<eos>`, `<unk>`)
- Architecture **Seq2Seq** (encodeur GRU + décodeur GRU) avec teacher forcing
- Décodage glouton vs **Beam Search** (k=3)
- **Transformer** : meilleure perplexité = **6.20**
- Évaluation par Perplexité et score **BLEU**

---

##  Installation et Exécution

### Prérequis

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### Lancer les notebooks

```bash
jupyter notebook Partie_I.ipynb    # MLP
jupyter notebook Partie_II.ipynb   # CNN
jupyter notebook Partie_III.ipynb  # RNN/Transformer
```

---

##  Technologies Utilisées

- **Python 3.10**
- **PyTorch** — Implémentation de tous les modèles
- **Jupyter Notebook** — Expérimentation et visualisation
- **Matplotlib / Seaborn** — Graphiques et courbes d'apprentissage
- **Scikit-learn** — Prétraitement et métriques
- **Overleaf / LaTeX** — Rédaction du rapport de synthèse