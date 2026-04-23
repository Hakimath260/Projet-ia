# ☁️ Projet IA Embarquée — Partie Cloud

![ThingSpeak](https://img.shields.io/badge/Cloud-ThingSpeak-blue)
![MATLAB](https://img.shields.io/badge/Analyse-MATLAB-orange?logo=mathworks)
![Python](https://img.shields.io/badge/Scripts-Python-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Validé-brightgreen)

---

## 📋 Sommaire

- [Objectif](#-objectif)
- [Architecture mise en place](#-architecture-mise-en-place)
- [Channels ThingSpeak](#-channels-thingspeak-créés)
- [Scripts Python](#-scripts-python-réalisés)
- [Analyse MATLAB](#-analyse-matlab-réalisée)
- [Résultats obtenus](#-résultats-obtenus)
- [Bilan](#-bilan-de-la-partie-cloud)
- [Guide de démonstration](#-guide-de-démonstration-soutenance)

---

## 🎯 Objectif

L'objectif de cette partie était de mettre en place une **chaîne cloud fonctionnelle sans utiliser la carte STM32**, afin de valider l'architecture logicielle du projet avant l'intégration matérielle finale.

Cette étape s'inscrit directement dans la logique du projet :

- Collecte et affichage de données dans **ThingSpeak**
- Exploitation de ces données via **MATLAB Analysis**
- Préparation d'une logique cloud pouvant ensuite être raccordée à la carte et au modèle IA

La carte STM32 a été **temporairement remplacée par des scripts Python** exécutés sur PC / Codespaces, afin de :

- Simuler l'envoi de données
- Vérifier le stockage cloud
- Lire les données depuis le cloud
- Effectuer une analyse côté MATLAB
- Écrire un résultat dans un second canal

---

## 🏗️ Architecture mise en place

```
Python / Codespaces
        │
        ▼
ThingSpeak — Raw Data (Channel 1)
        │
        ▼
MATLAB Analysis (traitement & classification)
        │
        ▼
ThingSpeak — Results (Channel 2)
```

| Étape | Rôle |
|-------|------|
| **Python / Codespaces** | Simule l'envoi de données météo vers le cloud |
| **ThingSpeak Raw Data** | Stocke les données d'entrée du système |
| **MATLAB Analysis** | Lit, traite et classifie les données |
| **ThingSpeak Results** | Stocke le résultat de classification et les statistiques |

> 💡 Cette architecture permet de valider toute la logique cloud même en l'absence temporaire d'adresse IP sur la carte, de capteurs reliés, ou de communication réelle STM32 → réseau.

---

## 📡 Channels ThingSpeak créés

### Channel 1 — Raw Data

Ce canal stocke les **données d'entrée du système** (rôle futur de la STM32).

| Champ | Variable |
|-------|----------|
| Field 1 | `Temperature` |
| Field 2 | `Humidity` |
| Field 3 | `Pressure` |
| Field 4 | `WindSpeed` |
| Field 5 | `Precipitation` |
| Field 6 | `Source` |

### Channel 2 — Results

Ce canal stocke les **résultats de traitement** produits par MATLAB Analysis.

| Champ | Variable |
|-------|----------|
| Field 1 | `PredictedClass` |
| Field 2 | `Confidence` |
| Field 3 | `TempMean` |
| Field 4 | `HumidityMean` |
| Field 5 | `PressureMean` |

---

## 🐍 Scripts Python réalisés

### Test préliminaire — Requête `curl`

Avant d'automatiser l'envoi, un premier test a été réalisé avec une requête HTTP POST manuelle vers l'API ThingSpeak afin de vérifier :

- La validité de la **Write API Key**
- L'accessibilité du canal
- Le bon affichage des données dans ThingSpeak

✅ Test concluant : une première entrée a bien été créée dans le canal Raw Data.

---

### `01_send_simulated_data.py` — Envoi de données simulées

Ce script envoie automatiquement plusieurs valeurs simulées vers ThingSpeak.

- Génère des valeurs réalistes de `temp`, `humidity`, `pressure`, `windspeed`, `precipitation`
- Envoie ces valeurs via l'**API REST ThingSpeak**
- Respecte la fréquence d'écriture du service (délai entre chaque envoi)

---

### `02_read_channel.py` — Lecture du canal Raw Data

Ce script relit les dernières valeurs du canal Raw Data.

- Interroge l'**API REST ThingSpeak**
- Récupère les derniers points du canal
- Affiche les données sous forme tabulaire

✅ Valide que les données sont non seulement bien écrites, mais aussi **relisibles par un client externe**.

---

### `03_prepare_demo_samples.py` — Injection de données Meteostat

Ce script remplace les données aléatoires par de **vraies lignes issues du dataset Meteostat** préparé dans la partie IA.

Il injecte plusieurs lignes du CSV dans le canal Raw Data, rendant la démonstration bien plus crédible :

- Les données ne sont plus artificielles
- Elles proviennent du pipeline de préparation IA
- Elles préparent concrètement le lien entre partie IA et partie cloud

---

## 📊 Analyse MATLAB réalisée

### Script 1 — Calcul des moyennes

Le premier script MATLAB lit les dernières valeurs du canal Raw Data et calcule :

- Moyenne de `température`
- Moyenne d'`humidité`
- Moyenne de `pression`

✅ Valide que le canal est accessible depuis MATLAB Analysis et que des traitements peuvent être exécutés côté cloud.

---

### Script 2 — Classification et écriture dans Results

Le second script reprend la lecture du canal Raw Data, calcule les moyennes, puis applique une **règle de décision** :

| Condition | Classe prédite |
|-----------|---------------|
| Temp. élevée **ET** humidité faible | ☀️ `clear` |
| Humidité très élevée | ☁️ `cloudy` |
| Autre cas | 🌧️ `rain` |

Il calcule ensuite les champs de sortie et les écrit dans le **canal Results** :

```
predictedClass  →  Channel 2 / Field 1
confidence      →  Channel 2 / Field 2
tempMean        →  Channel 2 / Field 3
humMean         →  Channel 2 / Field 4
presMean        →  Channel 2 / Field 5
```

---

## 📈 Résultats obtenus

### Côté Raw Data

Les valeurs envoyées depuis Python sont bien apparues dans le canal Raw Data, sous forme de courbes sur les champs numériques. Les données simulées puis les données Meteostat ont été stockées correctement.

### Côté lecture Python

Le script de lecture a bien affiché les dernières lignes du canal, confirmant que la **communication REST avec ThingSpeak fonctionne correctement**.

### Côté MATLAB Analysis

Lorsque les dernières données Meteostat ont été prises en compte, le résultat obtenu était :

| Champ | Valeur |
|-------|--------|
| `PredictedClass` | `1` (cloudy) |
| `Confidence` | `0.75` |
| `TempMean` | `0.48` |
| `HumidityMean` | `99.4` |
| `PressureMean` | `≈ 1030.8` |

> ✅ Résultats cohérents avec les données injectées : température basse, humidité très élevée, pression forte.

### Côté Results

Le canal ThingSpeak Results a bien reçu la nouvelle entrée produite par MATLAB Analysis.  
**La boucle cloud complète est validée.**

---

## 📌 Bilan de la partie cloud

### ✅ Validé

- [x] Création et configuration de deux canaux ThingSpeak
- [x] Écriture manuelle puis automatique de données dans Raw Data
- [x] Lecture du canal depuis Python
- [x] Analyse des données avec MATLAB Analysis
- [x] Écriture d'un résultat de classification dans Results
- [x] Utilisation de données Meteostat pour une démo réaliste

### 🔜 Reste à intégrer

- [ ] Raccorder la carte STM32 à cette chaîne cloud
- [ ] Envoyer les vraies mesures de capteurs depuis la carte
- [ ] Remplacer la règle MATLAB simple par la vraie logique IA (modèle ONNX)
- [ ] Ajouter TalkBack pour le retour cloud → carte

---

## 🎤 Guide de démonstration (Soutenance)

### ⚡ Version courte

```bash
python3 03_prepare_demo_samples.py
python3 02_read_channel.py
```

Puis lancer **MATLAB Analysis** et montrer le canal **Results**.

> *"Nous avons validé toute la chaîne cloud sans carte : envoi des données, lecture, analyse MATLAB et écriture du résultat."*

---

## 🏁 Conclusion

La partie cloud a été **validée avec succès sans dépendre de la carte STM32**.  
Même sans IP sur la carte et sans capteurs encore opérationnels, il a été possible de démontrer que :

- 📤 Les données peuvent être **envoyées** au cloud
- 📥 Elles peuvent être **relues**
- ⚙️ Elles peuvent être **traitées** par MATLAB
- 📊 Un résultat peut être **calculé et stocké** dans un canal dédié

Cette partie constitue une base solide pour la soutenance et pour l'**intégration finale du système embarqué**.
