# 🤖 Projet IA Embarquée — Classification Météo sur STM32

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![ONNX](https://img.shields.io/badge/Export-ONNX-lightgrey?logo=onnx)
![ThingSpeak](https://img.shields.io/badge/Cloud-ThingSpeak-blue)
![MATLAB](https://img.shields.io/badge/Analyse-MATLAB-orange?logo=mathworks)
![STM32](https://img.shields.io/badge/Embarqué-STM32-brightgreen)
![Status](https://img.shields.io/badge/Status-En%20cours-yellow)

---

## 📋 Sommaire

- [Présentation générale](#-présentation-générale)
- [Schéma global du projet](#-schéma-global-du-projet)
- [État actuel du projet](#-état-actuel-du-projet)
- [Ce qui fonctionne / ce qui reste](#-ce-qui-fonctionne--ce-qui-reste)
- [Répartition du travail](#-répartition-du-travail)
- [Arborescence du projet](#-arborescence-du-projet)
- [Pipeline de fonctionnement](#-pipeline-de-fonctionnement)
- [Démonstration soutenance](#-démonstration-possible-à-la-soutenance)
- [Perspectives](#-perspectives)

---

## 🧭 Présentation générale

Ce dépôt regroupe le travail réalisé dans le cadre de notre **projet d'IA embarquée**.  
L'objectif global est de mettre en place une chaîne complète permettant de :

1. **Récupérer des données météo** (réelles ou simulées)
2. **Les préparer et les exploiter** dans une pipeline IA
3. **Entraîner un modèle de classification météo**
4. **Exporter ce modèle** vers un format exploitable pour l'embarqué
5. **Mettre en place une partie cloud** avec **ThingSpeak / MATLAB**
6. **Intégrer le système final sur carte STM32**

À ce stade, le projet est déjà bien avancé sur les parties :
- ✅ **IA hors carte**
- ✅ **Cloud hors carte**
- ✅ **Organisation du dépôt et documentation**
- 🔜 **Intégration embarquée finale sur STM32**

---

## 🏗️ Schéma global du projet

```
Meteostat / Capteurs STM32
          │
          ▼
  Prétraitement Python
          │
          ▼
   Modèle TensorFlow
          │
          ▼
      Export ONNX
          │
          ▼
Cloud ThingSpeak / MATLAB
          │
          ▼
 Résultat de classification
          │
          ▼
  Intégration STM32 (à venir)
```

---

## 📊 État actuel du projet

### 🤖 Partie IA

- [x] Récupération de données météo via Meteostat
- [x] Constitution d'un dataset exploitable
- [x] Préparation et nettoyage des données
- [x] Définition d'une tâche de classification météo
- [x] Entraînement d'un premier modèle de base
- [x] Comparaison de plusieurs architectures
- [x] Sélection d'un meilleur modèle final
- [x] Export du modèle au format ONNX
- [x] Test d'inférence et vérification de cohérence TensorFlow ↔ ONNX

### ☁️ Partie Cloud

- [x] Création et configuration des channels ThingSpeak
- [x] Envoi de données simulées vers un channel Raw Data
- [x] Lecture des dernières entrées du channel depuis Python
- [x] Exploitation côté MATLAB Analysis
- [x] Publication du résultat de classification dans un channel Results

### 📁 Partie Documentation / Dépôt

- [x] Structuration propre du dépôt Git
- [x] Séparation claire entre données, scripts ML, cloud, résultats et documentation
- [x] Documentation progressive pour la soutenance et la reprise du projet

### 🔌 Partie Embarquée

- [x] Chaîne Edge AI préparée conceptuellement
- [ ] Intégration complète sur STM32 (en cours)
- [ ] Lecture réelle des capteurs
- [ ] Connectivité réseau depuis la carte

---

## ✅ Ce qui fonctionne / ce qui reste

### ✅ Ce qui fonctionne déjà

**1) Pipeline IA hors carte**
- Téléchargement des données Meteostat
- Préparation du dataset
- Visualisation de la distribution des classes
- Entraînement d'un modèle baseline
- Comparaison de plusieurs modèles
- Choix d'un meilleur modèle
- Export au format `.onnx`
- Test d'inférence validé ✔️

**2) Pipeline Cloud hors carte**
- Channel ThingSpeak Raw opérationnel
- Envoi de données météo simulées via Python
- Lecture des dernières données du channel
- Traitement via MATLAB Analysis
- Publication d'un résultat sur le channel ThingSpeak Results ✔️

**3) Résultats exploitables pour la soutenance**
- Courbes d'entraînement disponibles
- Matrice de confusion disponible
- Métriques sauvegardées
- Modèle final sauvegardé
- Format ONNX généré
- Démonstration logicielle faisable sans dépendre de la carte ✔️

### 🔜 Ce qui reste à finaliser

**1) Partie embarquée / carte STM32**
- Lecture effective des capteurs sur la carte
- Intégration complète du modèle dans la chaîne embarquée
- Exécution locale de l'inférence sur la carte
- Validation des sorties embarquées

**2) Partie réseau embarqué**
- Obtention fiable d'une adresse IP via le DHCP de l'université
- Stabilisation de la communication réseau depuis la carte
- Envoi direct des données de la carte vers le cloud

**3) Intégration bout-en-bout finale**
- Capteur → STM32 → réseau → cloud → résultat
- Démonstration complète avec données réelles issues du matériel

---

## 👥 Répartition du travail

Le projet est réalisé par trois étudiants issus de **deux formations complémentaires**, ce qui reflète directement la séparation naturelle des responsabilités dans le projet :

| Membre | Formation | Domaine de responsabilité |
|--------|-----------|--------------------------|
| **Ait Hamou Hakim** | L3 TRI | Infrastructure réseau, Cloud & Pipeline IA |
| **Benmansour Omar** | L3 TRI | Infrastructure réseau, Cloud & Pipeline IA |
| **Chaize Quentin** | L3 ESET | Électronique & Systèmes embarqués (STM32) |

---

### 🌐 Ait Hamou Hakim — L3 TRI

**Responsabilités principales :** pipeline IA, architecture logicielle et intégration cloud.

- Conception de l'architecture globale du projet
- Création et organisation de l'arborescence Git
- Récupération et préparation des données Meteostat
- Écriture des scripts ML (téléchargement, préparation, entraînement)
- Travail sur la connectivité réseau depuis la STM32 (obtention IP, protocole HTTP)
- Développement et test des scripts Python d'envoi vers ThingSpeak
- Comparaison des architectures et sélection du modèle final
- Export ONNX et test d'inférence croisée TensorFlow ↔ ONNX
- Mise en place du code MATLAB Analysis côté cloud
- Production de la documentation technique et préparation de la soutenance

---

### 🌐 Benmansour Omar — L3 TRI

**Responsabilités principales :** infrastructure réseau, scripts cloud et communication carte ↔ cloud.

- Participation à la conception de l'architecture réseau du projet
- Développement et test des scripts Python d'envoi vers ThingSpeak
- Mise en place du script de lecture et de vérification du channel Raw Data
- Préparation des échantillons de démonstration (`03_prepare_demo_samples.py`)
- Configuration réseau et gestion des clés API ThingSpeak
- Contribution à la stabilisation de la communication réseau embarquée
- Appui sur la future intégration STM32 → réseau → cloud

---

### 🔌 Chaize Quentin — L3 ESET

**Responsabilités principales :** électronique, systèmes embarqués et intégration STM32.

- Prise en charge de la partie matérielle du projet (carte STM32)
- Sélection et câblage des capteurs météo (température, humidité, pression…)
- Développement du firmware embarqué pour la lecture des capteurs
- Préparation de l'intégration du modèle ONNX sur la carte
- Coordination avec l'équipe TRI pour le raccordement STM32 → cloud
- Tests matériels et validation des données issues des capteurs
- Contribution à la démonstration finale du système complet

---

## 📁 Arborescence du projet

```text
Projet-ia/
├── cloud/
│   ├── docs/
│   │   └── cloud_pipeline.md
│   └── thingspeak/
│       ├── 01_send_simulated_data.py
│       ├── 02_read_channel.py
│       ├── 03_prepare_demo_samples.py
│       ├── config_example.py
│       ├── config.py
│       └── matlab_analysis_code.md
│
├── data/
│   ├── external/
│   ├── processed/
│   │   └── weather_4classes.csv
│   └── raw/
│       └── meteostat_chambery_hourly.csv
│
├── docs/
│   └── ia_pipeline.md
│
├── ml/
│   ├── models/
│   │   ├── scaler_mean.npy
│   │   ├── scaler_scale.npy
│   │   ├── weather_model_3classes.keras
│   │   └── weather_model_final.keras
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
│
├── results/
│   ├── confusion_matrices/
│   │   └── baseline_3classes_cm.npy
│   ├── figures/
│   │   ├── baseline_3classes_accuracy.png
│   │   ├── baseline_3classes_loss.png
│   │   └── class_distribution.png
│   └── metrics/
│       ├── baseline_3classes_accuracy.txt
│       ├── baseline_3classes_report.json
│       ├── best_model.json
│       └── model_comparison.csv
│
└── README.md
```

### Description des dossiers

| Dossier / Fichier | Rôle |
|-------------------|------|
| `cloud/docs/` | Documentation de la chaîne cloud |
| `cloud/thingspeak/` | Scripts Python et code MATLAB pour ThingSpeak |
| `data/raw/` | Données météo brutes Meteostat |
| `data/processed/` | Dataset nettoyé et structuré pour l'apprentissage |
| `data/external/` | Données complémentaires éventuelles |
| `docs/` | Documentation générale du projet |
| `ml/models/` | Modèles Keras entraînés et paramètres de normalisation |
| `ml/onnx/` | Modèle exporté au format ONNX |
| `ml/scripts/` | Scripts Python de la pipeline IA |
| `ml/notebooks/` | Notebooks Jupyter pour expérimentations |
| `results/figures/` | Courbes d'entraînement et répartition des classes |
| `results/metrics/` | Métriques chiffrées et comparaison des modèles |
| `results/confusion_matrices/` | Matrices de confusion sauvegardées |

---

## 🔄 Pipeline de fonctionnement

```
[1] Téléchargement des données météo (Meteostat)       ← L3 TRI
        ↓
[2] Préparation / nettoyage des données (Python)       ← L3 TRI
        ↓
[3] Entraînement et comparaison des modèles (Keras)    ← L3 TRI
        ↓
[4] Export du modèle final (ONNX)                      ← L3 TRI
        ↓
[5] Envoi de données vers ThingSpeak (Python)          ← L3 TRI
        ↓
[6] Lecture et traitement des données (MATLAB)         ← L3 TRI
        ↓
[7] Publication du résultat (ThingSpeak Results)       ← L3 TRI
        ↓
[8] Intégration sur STM32 + lecture capteurs           ← L3 ESET
        ↓
[9] Démo complète : capteur → carte → cloud → résultat ← Équipe complète
```

---

## 🎤 Démonstration possible à la soutenance

Même sans carte complètement finalisée, une démonstration cohérente et stable est déjà possible.

### Démo

1. Montrer le dépôt Git et l'arborescence
2. Exécuter (ou montrer les résultats de) la pipeline IA
3. Montrer les graphiques et métriques
4. Envoyer des données simulées vers ThingSpeak
5. Montrer la lecture du channel Raw Data
6. Montrer le résultat calculé côté MATLAB / channel Results
7. Conclure sur la future intégration STM32

### Commandes clés

```bash
# Pipeline IA
python3 ml/scripts/02_prepare_dataset.py
python3 ml/scripts/04_compare_models.py
python3 ml/scripts/06_inference_test.py

# Pipeline Cloud
cd cloud/thingspeak
python3 03_prepare_demo_samples.py
python3 02_read_channel.py
```

---

## 🚀 Perspectives

- 🔌 Connexion stable de la carte STM32 au réseau universitaire
- 📡 Lecture réelle des capteurs météo depuis la carte
- ⚡ Exécution du modèle ONNX directement sur la STM32
- 🎯 Démonstration complète : **capteur → carte → cloud → classification**

---

## 🏁 Conclusion

Le projet dispose déjà :

- ✅ D'une **pipeline IA fonctionnelle** (données → modèle → ONNX → inférence)
- ✅ D'une **pipeline cloud fonctionnelle** (ThingSpeak → MATLAB → résultats)
- ✅ D'une **organisation Git propre** et d'une documentation complète
- ✅ D'une **base solide pour la soutenance**, démontrable sans dépendre de la carte

La partie restant à finaliser concerne l'**intégration embarquée réelle sur STM32**, portée par Quentin Chaize (L3 ESET), qui viendra compléter la chaîne déjà bien avancée côté logiciel et cloud par l'équipe TRI.
