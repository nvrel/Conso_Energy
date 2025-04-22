# %% [markdown]
# # Analyse des données énergétiques
# 
# Ce notebook contient l'analyse des données énergétiques normalisées et transformées.

# %% [markdown]
# ## 1. Configuration et imports

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des chemins
DATA_DIR = os.path.join('..', 'data')
NORMALIZED_DIR = os.path.join(DATA_DIR, 'normalized_csv')
TRANSFORMED_DIR = os.path.join(DATA_DIR, 'transformed_data')

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn')
sns.set_palette('husl')

# %% [markdown]
# ## 2. Fonctions utilitaires

# %%
def load_data(directory):
    """
    Charge tous les fichiers CSV d'un répertoire dans un dictionnaire de DataFrames.
    
    Args:
        directory (str): Chemin du répertoire contenant les fichiers CSV
        
    Returns:
        dict: Dictionnaire avec les noms de fichiers comme clés et les DataFrames comme valeurs
    """
    data = {}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            data[file] = pd.read_csv(file_path)
    return data

def analyze_missing_values(df):
    """
    Analyse les valeurs manquantes dans un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        
    Returns:
        dict: Dictionnaire contenant les statistiques des valeurs manquantes
    """
    # Calcul du nombre total de lignes
    total_rows = len(df)
    
    # Calcul des lignes avec au moins une valeur manquante
    rows_with_missing = df.isnull().any(axis=1).sum()
    
    # Calcul des statistiques par colonne
    missing_stats = df.isnull().sum()
    missing_percentages = (missing_stats / total_rows * 100).round(2)
    
    # Colonnes avec au moins une valeur manquante
    columns_with_missing = missing_stats[missing_stats > 0].index.tolist()
    
    return {
        'total_rows': total_rows,
        'rows_with_missing': rows_with_missing,
        'percentage_rows_with_missing': (rows_with_missing / total_rows * 100).round(2),
        'columns_with_missing': columns_with_missing,
        'missing_stats': pd.DataFrame({
            'missing_count': missing_stats,
            'missing_percentage': missing_percentages
        }).loc[columns_with_missing]
    }

# %% [markdown]
# ## 3. Chargement des données

# %%
# Chargement des données normalisées
normalized_data = load_data(NORMALIZED_DIR)
print(f"Fichiers normalisés chargés: {list(normalized_data.keys())}")

# Chargement des données transformées
transformed_data = load_data(TRANSFORMED_DIR)
print(f"Fichiers transformés chargés: {list(transformed_data.keys())}")

# %% [markdown]
# ## 4. Analyse des valeurs manquantes

# %%
def display_missing_values_analysis(data_dict, title):
    """
    Affiche l'analyse des valeurs manquantes pour un ensemble de DataFrames.
    
    Args:
        data_dict (dict): Dictionnaire de DataFrames
        title (str): Titre de l'analyse
    """
    print(f"\n=== {title} ===\n")
    
    for file_name, df in data_dict.items():
        print(f"\nAnalyse pour {file_name}:")
        missing_analysis = analyze_missing_values(df)
        
        print(f"Nombre total de lignes: {missing_analysis['total_rows']}")
        print(f"Lignes avec valeurs manquantes: {missing_analysis['rows_with_missing']} ({missing_analysis['percentage_rows_with_missing']}%)")
        
        if missing_analysis['columns_with_missing']:
            print("\nColonnes avec valeurs manquantes:")
            print(missing_analysis['missing_stats'])
        else:
            print("\nAucune valeur manquante dans ce fichier.")
        
        print("\n" + "="*50 + "\n")

# %%
# Analyse des données normalisées
display_missing_values_analysis(normalized_data, "Données normalisées")

# Analyse des données transformées
display_missing_values_analysis(transformed_data, "Données transformées") 