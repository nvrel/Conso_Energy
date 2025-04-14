# Projet de Prédiction de Consommation d'Énergie

Ce projet vise à prédire la consommation d'énergie pour le chauffage et la climatisation en utilisant des techniques d'apprentissage automatique et d'IA agentique.

## Structure du Projet

Le projet est organisé en modules indépendants :
1. `data_downloader.py` : Module de téléchargement et décompression des données
2. (À venir) Module de transformation des données
3. (À venir) Module de traitement des données manquantes et outliers
4. (À venir) Module d'analyse des données

## Installation

1. Créez un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation du Module de Téléchargement

Le module `data_downloader.py` permet de télécharger et décompresser automatiquement les données depuis l'UCI Machine Learning Repository.

Exemple d'utilisation :
```python
from data_downloader import DataDownloader

# Créer une instance du téléchargeur
downloader = DataDownloader(
    url="https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    output_dir="data"
)

# Télécharger et extraire les données
data_dir = downloader.download_and_extract()
```

Les données seront téléchargées dans le répertoire `data/` par défaut.

## Prochaines Étapes

1. Implémentation du module de transformation des données
2. Implémentation du module de traitement des données manquantes
3. Implémentation du module d'analyse des données
4. Développement du modèle de prédiction
5. Intégration de l'IA agentique pour le contrôle intelligent 