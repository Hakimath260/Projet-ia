# 🌤️ Projet IA Embarquée — Partie IA Hors Carte

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![ONNX](https://img.shields.io/badge/Export-ONNX-lightgrey?logo=onnx)
![Status](https://img.shields.io/badge/Status-En%20cours-yellow)

---

## 📋 Sommaire

- [Contexte](#-contexte)
- [Objectifs](#-objectifs)
- [Fonctionnalités implémentées](#-fonctionnalités-implémentées)
- [Arborescence du projet](#-arborescence-du-projet)
- [Pipeline complet](#-pipeline-complet)
- [Choix de modélisation](#-choix-de-modélisation)
- [Résultats obtenus](#-résultats-obtenus)
- [Export et interopérabilité](#-export-et-interopérabilité)
- [Installation & Exécution](#-installation--exécution)
- [État du projet](#-état-du-projet)
- [Guide de démonstration](#-guide-de-démonstration)

---

## 🧭 Contexte

Ce dépôt contient l'avancement de la **partie IA hors carte** de notre projet d'IA embarquée.  
L'objectif est de construire une chaîne complète de traitement **avant intégration sur STM32** :

1. Récupération de données météo historiques avec **Meteostat**
2. Préparation d'un dataset exploitable pour la classification
3. Entraînement de plusieurs modèles **TensorFlow/Keras**
4. Comparaison des architectures selon le compromis **accuracy / taille / pertinence embarquée**
5. Export du meilleur modèle au format **ONNX**
6. Validation d'une inférence locale en **TensorFlow** et en **ONNX Runtime**

Cette partie constitue la base logicielle qui sera ensuite raccordée au **Cloud** (ThingSpeak / MATLAB), puis à la **carte STM32** pour l'inférence embarquée.

---

## 🎯 Objectifs

- Créer un dataset météo réaliste à partir de données ouvertes
- Définir des **classes météo** cohérentes
- Entraîner un premier modèle simple mais robuste
- Comparer plusieurs variantes pour justifier le choix final
- Préparer l'interopérabilité **Cloud / Edge** via **ONNX**
- Disposer d'une brique IA démontrable même sans carte réseau/capteurs finalisés

---

## ✅ Fonctionnalités implémentées

| # | Fonctionnalité | Description |
|---|----------------|-------------|
| 1 | **Téléchargement Meteostat** | Récupération de données météo horaires depuis une station proche de Chambéry |
| 2 | **Préparation du dataset** | Extraction, nettoyage et transformation des colonnes utiles |
| 3 | **Construction des labels météo** | Classification des observations en classes météo cohérentes |
| 4 | **Entraînement d'un baseline** | Premier modèle dense simple sous TensorFlow/Keras |
| 5 | **Comparaison d'architectures** | Étude du compromis performance / taille entre plusieurs modèles |
| 6 | **Export ONNX** | Export du modèle final pour réutilisation cloud et embarquée |
| 7 | **Validation de l'inférence** | Test croisé TensorFlow ↔ ONNX Runtime — même classe prédite ✔️ |

---

## 📁 Arborescence du projet

```text
.
├── README.md
├── data/
│   ├── raw/
│   │   └── meteostat_chambery_hourly.csv
│   ├── processed/
│   │   └── weather_4classes.csv
│   └── external/
├── docs/
│   └── ia_pipeline.md
├── ml/
│   ├── models/
│   │   ├── weather_model_3classes.keras
│   │   ├── weather_model_final.keras
│   │   ├── scaler_mean.npy
│   │   └── scaler_scale.npy
│   ├── notebooks/
│   ├── onnx/
│   │   └── weather_model_final.onnx
│   └── scripts/
│       ├── 01_download_meteostat.py
│       ├── 02_prepare_dataset.py
│       ├── 03_train_baseline.py
│       ├── 04_compare_models.py
│       ├── 05_export_onnx.py
│       └── 06_inference_test.py
└── results/
    ├── confusion_matrices/
    │   └── baseline_3classes_cm.npy
    ├── figures/
    │   ├── class_distribution.png
    │   ├── baseline_3classes_accuracy.png
    │   └── baseline_3classes_loss.png
    └── metrics/
        ├── baseline_3classes_accuracy.txt
        ├── baseline_3classes_report.json
        ├── model_comparison.csv
        └── best_model.json
```

### Rôle des dossiers

| Dossier | Rôle |
|---------|------|
| `data/raw/` | Données brutes Meteostat, sans transformation |
| `data/processed/` | Données nettoyées, prêtes pour l'apprentissage |
| `data/external/` | Jeux de données complémentaires éventuels |
| `docs/` | Pipeline IA, architecture, notes de conception, préparation soutenance |
| `ml/scripts/` | Scripts Python exécutables du pipeline |
| `ml/models/` | Modèles entraînés et paramètres de normalisation |
| `ml/onnx/` | Exports ONNX du modèle final |
| `ml/notebooks/` | Expérimentations interactives et visualisations |
| `results/figures/` | Graphiques : répartition des classes, courbes accuracy/loss |
| `results/metrics/` | Métriques chiffrées : accuracy, rapport de classification, comparaison |
| `results/confusion_matrices/` | Matrices de confusion pour l'analyse des erreurs |

---

## 🔄 Pipeline complet

```
[1] Téléchargement des données brutes (Meteostat)
        ↓
[2] Nettoyage et préparation
        ↓
[3] Création des classes météo
        ↓
[4] Entraînement d'un baseline
        ↓
[5] Comparaison de plusieurs modèles
        ↓
[6] Choix d'un modèle final
        ↓
[7] Export ONNX
        ↓
[8] Validation de l'inférence TensorFlow / ONNX ✅
```

### Rôle de chaque script

| Script | Rôle |
|--------|------|
| `01_download_meteostat.py` | Identifie une station, vérifie l'inventaire, récupère et sauvegarde les données horaires |
| `02_prepare_dataset.py` | Sélectionne les colonnes, nettoie les valeurs manquantes, construit les labels, génère les graphiques |
| `03_train_baseline.py` | Charge le dataset, enlève la classe `fog`, normalise, entraîne et sauvegarde le modèle baseline |
| `04_compare_models.py` | Entraîne plusieurs variantes, compare accuracy et taille, sélectionne le meilleur modèle |
| `05_export_onnx.py` | Exporte le modèle final en `.onnx` pour une utilisation cloud et embarquée |
| `06_inference_test.py` | Teste une inférence croisée TensorFlow ↔ ONNX Runtime sur un exemple simulé |

---

## 🧠 Choix de modélisation

### Features utilisées

| Feature | Description |
|---------|-------------|
| `temp` | Température |
| `rhum` | Humidité relative |
| `pres` | Pression atmosphérique |
| `wspd` | Vitesse du vent |
| `prcp` | Précipitations |

Ces variables sont directement interprétables, pertinentes pour une classification météo simple et proches des mesures exploitées par le système final.

### Classes météo

Le dataset contient initialement **4 classes** :

| Label | Classe |
|-------|--------|
| `0` | ☀️ clear |
| `1` | ☁️ cloudy |
| `2` | 🌧️ rain |
| `3` | 🌫️ fog |

> ⚠️ La classe `fog` étant très fortement sous-représentée, elle a été exclue pour le baseline afin d'obtenir un premier modèle robuste sur **3 classes**.

---

## 📊 Résultats obtenus

### Baseline 3 classes

| Métrique | Valeur |
|----------|--------|
| Accuracy test | **~78.6 %** |
| Meilleure classe | `cloudy` |
| Classe la plus difficile | `clear` (souvent confondue avec `cloudy`) |

### Comparaison des architectures

| Modèle | Accuracy | Taille | Usage recommandé |
|--------|----------|--------|-----------------|
| `small_relu` | ✔️ compétitif | 🟢 très léger | **Edge / STM32** |
| `baseline_relu` | ✔️ bon | 🟡 moyen | Référence |
| `bigger_relu` | 🏆 meilleure | 🔴 plus lourd | **Cloud** |
| `tanh_model` | ✔️ correct | 🟡 moyen | Alternative |
| `baseline_relu_rmsprop` | ✔️ correct | 🟡 moyen | Alternative |

> 💡 **Conclusion** : `bigger_relu` offre la meilleure accuracy brute (candidat cloud), tandis que `small_relu` offre le meilleur compromis taille/performance (candidat edge).

---

## 📦 Export et interopérabilité

Le modèle final est disponible en deux formats :

| Format | Fichier | Usage |
|--------|---------|-------|
| Keras | `weather_model_final.keras` | Entraînement / fine-tuning Python |
| ONNX | `weather_model_final.onnx` | Cloud, MATLAB, embarqué |

L'export ONNX permet une interopérabilité complète entre Python/TensorFlow, ONNX Runtime, MATLAB et les outils de conversion pour l'embarqué.

**Résultat de validation :**
```
Même classe TensorFlow / ONNX : True ✅
```

---

## 🚀 Installation & Exécution

### 1. Installation des dépendances

```bash
pip install meteostat pandas numpy matplotlib scikit-learn tensorflow tf2onnx onnx onnxruntime
```

### 2. Exécution du pipeline complet

```bash
python3 ml/scripts/01_download_meteostat.py
python3 ml/scripts/02_prepare_dataset.py
python3 ml/scripts/03_train_baseline.py
python3 ml/scripts/04_compare_models.py
python3 ml/scripts/05_export_onnx.py
python3 ml/scripts/06_inference_test.py
```

---

## 📌 État du projet

### ✅ Validé

- [x] Récupération des données Meteostat
- [x] Préparation du dataset
- [x] Construction des labels météo
- [x] Entraînement d'un baseline
- [x] Comparaison d'architectures
- [x] Choix d'un modèle final
- [x] Export ONNX
- [x] Test d'inférence locale TensorFlow / ONNX

### 🔜 Reste à intégrer

- [ ] Lecture des données réelles depuis la carte STM32
- [ ] Transmission réseau / cloud depuis la carte
- [ ] Raccordement entre acquisition embarquée et modèle final
- [ ] Démonstration temps réel complète : capteurs → cloud → inférence → retour carte

---

## 🎤 Guide de démonstration (Soutenance)

> Objectif : montrer un maximum de choses en **5 minutes**, avec une logique simple, visuelle et fluide.

### ⚡ Version rapide

```bash
python3 ml/scripts/02_prepare_dataset.py
python3 ml/scripts/04_compare_models.py
python3 ml/scripts/06_inference_test.py
```

> *"Dataset météo réel Meteostat → comparaison de modèles → export ONNX validé."*

---

## 🏁 Conclusion

Cette partie du projet démontre que la **chaîne IA fonctionne de manière cohérente hors carte**.  
Elle fournit une base solide pour la suite :

- 🌐 Intégration cloud
- 🔌 Intégration sur STM32
- ⚡ Inférence embarquée
- 🎯 Démonstration finale complète
