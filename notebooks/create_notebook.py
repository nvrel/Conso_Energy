import nbformat as nbf
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openai
from IPython.display import display, Markdown
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer

# Charger les variables d'environnement depuis .env
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)  # Remonter d'un niveau
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

# Créer un nouveau notebook
nb = nbf.v4.new_notebook()

# Cellule 1: Titre
cells = [
    nbf.v4.new_markdown_cell("# Analyse des données énergétiques\n\nCe notebook contient l'analyse des données énergétiques normalisées et transformées.")
]

# Cellule 2: Configuration et imports
cells.append(nbf.v4.new_markdown_cell("## 1. Configuration et imports"))
cells.append(nbf.v4.new_code_cell('''import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openai
import json
from IPython.display import display, Markdown

# Configuration des chemins
DATA_DIR = os.path.join('..', 'data')
NORMALIZED_DIR = os.path.join(DATA_DIR, 'normalized_csv')
TRANSFORMED_DIR = os.path.join(DATA_DIR, 'transformed_data')

# Afficher les chemins
print("DATA_DIR:", os.path.abspath(DATA_DIR))
print("NORMALIZED_DIR:", os.path.abspath(NORMALIZED_DIR))
print("TRANSFORMED_DIR:", os.path.abspath(TRANSFORMED_DIR))

# Vérifier l'existence des répertoires
print("Vérification des répertoires:")
print("DATA_DIR existe:", os.path.exists(DATA_DIR))
print("NORMALIZED_DIR existe:", os.path.exists(NORMALIZED_DIR))
print("TRANSFORMED_DIR existe:", os.path.exists(TRANSFORMED_DIR))

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('default')  # Style par défaut de matplotlib
sns.set_theme()  # Style par défaut de seaborn'''))

# Cellule 3: Fonctions utilitaires
cells.append(nbf.v4.new_markdown_cell("## 2. Fonctions utilitaires"))
cells.append(nbf.v4.new_code_cell('''def load_data(directory):
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
            # Utiliser sep=';' et decimal=',' pour le format français
            data[file] = pd.read_csv(file_path, sep=';', decimal=',')
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
    }'''))

# Cellule 4: Chargement des données
cells.append(nbf.v4.new_markdown_cell("## 3. Chargement des données"))
cells.append(nbf.v4.new_code_cell('''# Charger les données avec les bons paramètres
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
            # Utiliser sep=';' et decimal=',' pour le format français
            data[file] = pd.read_csv(file_path, sep=';', decimal=',')
    return data

# Charger les données
print("Chargement des données :")
normalized_data = load_data(NORMALIZED_DIR)
transformed_data = load_data(TRANSFORMED_DIR)

# Afficher les premières lignes des données chargées
for file_name, df in normalized_data.items():
    print(f"Données brutes de {file_name} :")
    display(df.head())
    print("Types des colonnes :")
    display(df.dtypes)'''))

# Cellule 5: Analyse des valeurs manquantes
cells.append(nbf.v4.new_markdown_cell("## 4. Analyse des valeurs manquantes"))
cells.append(nbf.v4.new_code_cell('''def display_missing_values_analysis(data_dict, title):
    """
    Affiche l'analyse des valeurs manquantes pour un ensemble de DataFrames.
    
    Args:
        data_dict (dict): Dictionnaire de DataFrames
        title (str): Titre de l'analyse
    """
    print(f"=== {title} === \\n")
    
    for file_name, df in data_dict.items():
        print(f"Analyse pour {file_name}:")
        missing_analysis = analyze_missing_values(df)
        
        print(f"Nombre total de lignes: {missing_analysis['total_rows']}")
        print(f"Lignes avec valeurs manquantes: {missing_analysis['rows_with_missing']} ({missing_analysis['percentage_rows_with_missing']}%)")
        
        if missing_analysis['columns_with_missing']:
            print("Colonnes avec valeurs manquantes:")
            print(missing_analysis['missing_stats'])
        else:
            print("Aucune valeur manquante dans ce fichier.")
        
        print("\\n" + "="*50 + "\\n")'''))
cells.append(nbf.v4.new_code_cell('''# Analyse des données normalisées
display_missing_values_analysis(normalized_data, "Données normalisées")

# Analyse des données transformées
display_missing_values_analysis(transformed_data, "Données transformées")'''))

# Cellule 6: Configuration des colonnes
cells.append(nbf.v4.new_markdown_cell("## 5. Configuration des colonnes"))
cells.append(nbf.v4.new_code_cell('''# Configuration des colonnes
column_config = {
    'X1': {'name': 'relative_compactness', 'type': 'numeric', 'prefix': 'f_'},
    'X2': {'name': 'surface_area', 'type': 'numeric', 'prefix': 'f_'},
    'X3': {'name': 'wall_area', 'type': 'numeric', 'prefix': 'f_'},
    'X4': {'name': 'roof_area', 'type': 'numeric', 'prefix': 'f_'},
    'X5': {'name': 'overall_height', 'type': 'numeric', 'prefix': 'f_'},
    'X6': {'name': 'orientation', 'type': 'categorical', 'prefix': 'f_'},
    'X7': {'name': 'glazing_area', 'type': 'numeric', 'prefix': 'f_'},
    'X8': {'name': 'glazing_distribution', 'type': 'categorical', 'prefix': 'f_'},
    'Y1': {'name': 'heating_load', 'type': 'numeric', 'prefix': 'l_'},
    'Y2': {'name': 'cooling_load', 'type': 'numeric', 'prefix': 'l_'}
}

# Créer un DataFrame pour l'affichage
config_df = pd.DataFrame([
    {
        'Code': code,
        'Nom': config['name'],
        'Nom avec préfixe': f"{config['prefix']}{config['name']}",
        'Type': config['type']
    }
    for code, config in column_config.items()
])

# Afficher le tableau
print("Configuration des colonnes :")
display(config_df.style
    .set_properties(**{'text-align': 'left'})
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ])
    .hide(axis='index')
)'''))

# Cellule 7: Conversion des colonnes
cells.append(nbf.v4.new_markdown_cell("## 6. Conversion des colonnes"))
cells.append(nbf.v4.new_code_cell('''# Fonction pour convertir les colonnes selon leur type
def convert_columns(df, column_config):
    """
    Convertit les colonnes selon leur type et applique les préfixes appropriés.
    
    Args:
        df (pd.DataFrame): DataFrame à convertir
        column_config (dict): Configuration des colonnes
        
    Returns:
        pd.DataFrame: DataFrame converti
    """
    # Créer une copie du DataFrame
    df_converted = df.copy()
    
    # Dictionnaire pour mapper les noms originaux aux noms avec préfixes
    column_mapping = {}
    
    # Parcourir la configuration et convertir les colonnes
    for code, config in column_config.items():
        original_name = config['name']
        new_name = f"{config['prefix']}{original_name}"
        
        if original_name in df_converted.columns:
            # Convertir selon le type
            if config['type'] == 'numeric':
                df_converted[original_name] = pd.to_numeric(df_converted[original_name], errors='coerce')
            elif config['type'] == 'categorical':
                df_converted[original_name] = df_converted[original_name].astype('category')
            
            # Enregistrer le mapping
            column_mapping[original_name] = new_name
    
    # Renommer les colonnes avec les préfixes
    df_converted = df_converted.rename(columns=column_mapping)
    
    return df_converted

# Convertir les données normalisées
print("Conversion des données normalisées :")
normalized_data_converted = {}
for file_name, df in normalized_data.items():
    print(f"Traitement du fichier {file_name}:")
    df_converted = convert_columns(df, column_config)
    normalized_data_converted[file_name] = df_converted
    print("Colonnes converties :")
    display(df_converted.head())
    print("Types des colonnes après conversion :")
    display(df_converted.dtypes)'''))

# Cellule 8: Analyse des corrélations
cells.append(nbf.v4.new_markdown_cell("## 7. Analyse des corrélations"))
cells.append(nbf.v4.new_code_cell('''# Fonction pour créer la matrice de corrélation
def create_correlation_matrix(data_dict):
    """
    Crée une matrice de corrélation pour chaque DataFrame dans le dictionnaire.
    
    Args:
        data_dict (dict): Dictionnaire de DataFrames
        
    Returns:
        dict: Dictionnaire contenant les matrices de corrélation
    """
    correlation_matrices = {}
    
    for file_name, df in data_dict.items():
        print(f"Analyse du fichier {file_name}:")
        
        # Vérifier si le DataFrame est vide
        if df.empty:
            print(f"  Attention: Le DataFrame {file_name} est vide!")
            continue
            
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        # Vérifier s'il y a des colonnes numériques
        if len(numeric_cols) == 0:
            print(f"  Attention: Aucune colonne numérique trouvée dans {file_name}!")
            print("  Types de colonnes présents:", df.dtypes.unique())
            continue
            
        print(f"  Colonnes numériques trouvées: {len(numeric_cols)}")
        
        # Créer la matrice de corrélation
        corr_matrix = df[numeric_cols].corr()
        
        correlation_matrices[file_name] = corr_matrix
    
    return correlation_matrices

# Créer les matrices de corrélation pour les données normalisées converties
correlation_matrices = create_correlation_matrix(normalized_data_converted)

# Afficher les matrices de corrélation
for file_name, corr_matrix in correlation_matrices.items():
    print(f"Matrice de corrélation pour {file_name}:")
    print(corr_matrix)
    
    # Créer un heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title(f'Matrice de corrélation - {file_name}')
    plt.tight_layout()
    plt.show()'''))

# Cellule 9: Test d'indépendance des variables catégorielles
cells.append(nbf.v4.new_markdown_cell("""## 8. Test d'indépendance des variables catégorielles

### Démarche d'analyse

Pour tester l'indépendance entre les features catégorielles (f_orientation et f_glazing_distribution), nous utilisons :

- **Test exact de Fisher** :
  * H0 : Les variables sont indépendantes
  * H1 : Il existe une dépendance entre les variables
  * Interprétation : Si p-value < 0.05, on rejette H0 et on conclut à une dépendance entre les variables
"""))

cells.append(nbf.v4.new_code_cell('''# Test d'indépendance entre variables catégorielles
from scipy.stats import fisher_exact
import pandas as pd

def test_categorical_independence(df, var1, var2):
    """
    Teste l'indépendance entre deux variables catégorielles avec le test exact de Fisher.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        var1 (str): Nom de la première variable catégorielle
        var2 (str): Nom de la deuxième variable catégorielle
    """
    print(f"Test d'indépendance entre {var1} et {var2}")
    print("-" * 60)
    
    # Créer la table de contingence
    contingency = pd.crosstab(df[var1], df[var2])
    print("Table de contingence :")
    display(contingency)
    
    # Test de Fisher
    odds_ratio, p_value = fisher_exact(contingency)
    
    # Affichage des résultats
    print(f"Test exact de Fisher :")
    print(f"p-value: {p_value:.4f}")
    
    # Interprétation
    print("Interprétation:")
    if p_value < 0.05:
        print("→ p-value < 0.05 : On rejette l'hypothèse d'indépendance")
        print("→ Les variables ne sont pas indépendantes (elles sont liées)")
    else:
        print("→ p-value ≥ 0.05 : On ne peut pas rejeter l'hypothèse d'indépendance")
        print("→ Pas de preuve statistique d'une dépendance entre les variables")

# Analyser l'indépendance des variables catégorielles
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Analyse d'indépendance des variables catégorielles pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les features catégorielles (préfixe 'f_')
    categorical_features = [col for col in df.columns if df[col].dtype.name == 'category' and col.startswith('f_')]
    
    # Tester l'indépendance entre chaque paire de features catégorielles
    for i in range(len(categorical_features)):
        for j in range(i+1, len(categorical_features)):
            test_categorical_independence(df, categorical_features[i], categorical_features[j])
'''))

# Cellule 10: Sélection des features
cells.append(nbf.v4.new_markdown_cell("""## 10. Sélection des features

### Démarche d'analyse
Nous allons appliquer l'algorithme de sélection des features en trois étapes :

1. **Sélection initiale des features numériques** :
   - Calcul de la somme des corrélations absolues avec tous les labels pour chaque feature
   - Tri des features par ordre décroissant de cette somme (les plus corrélées d'abord)
   - Sélection itérative en évitant la multicolinéarité :
     * On prend la première feature (la plus corrélée avec les labels)
     * Pour chaque feature restante, on la garde uniquement si sa corrélation avec toutes les features déjà sélectionnées est ≤ 0.85
     * Cela permet d'éviter la redondance d'information tout en gardant les features les plus pertinentes

2. **Sélection des features catégorielles** :
   - Calcul d'un score de "corrélation" pour chaque feature catégorielle :
     * Pour chaque label, on effectue un test statistique (Wilcoxon ou Kruskal-Wallis selon le nombre de classes)
     * Le score est la somme des -log(p-value) pour tous les labels
     * Plus le score est élevé, plus la feature est liée aux labels
   - Tri des features par score décroissant
   - Élimination des redondances :
     * Test de Fisher entre chaque paire de features catégorielles
     * Si dépendance détectée (p-value < 0.05), on garde celle avec le meilleur score

3. **Validation finale** :
   - Pour chaque paire (feature numérique, feature catégorielle) :
     * Test statistique pour vérifier leur indépendance
     * Si dépendance détectée (p-value < 0.05), on élimine la feature catégorielle
     * Cela permet d'éviter la redondance entre types de features différents
"""))

cells.append(nbf.v4.new_code_cell('''# Initialiser les ensembles pour stocker les features
features_to_keep = set()
features_to_eliminate = set()

for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Sélection des features pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les features numériques et catégorielles
    numeric_features = [col for col in df.columns if col.startswith('f_') and df[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in df.columns if col.startswith('f_') and df[col].dtype.name == 'category']
    labels = [col for col in df.columns if col.startswith('l_')]
    
    print("Features numériques initiales:", numeric_features)
    print("Features catégorielles initiales:", categorical_features)
    print("Labels:", labels)
    
    # 1. Sélection initiale des features numériques
    print("1. Sélection des features numériques")
    print("-" * 40)
    
    # Calculer la somme des corrélations absolues avec les labels
    corr_sums = {}
    for feature in numeric_features:
        corr_sum = sum(abs(df[feature].corr(df[label])) for label in labels)
        corr_sums[feature] = corr_sum
    
    # Trier les features par corrélation décroissante
    sorted_features = sorted(corr_sums.items(), key=lambda x: x[1], reverse=True)
    
    # Sélectionner les features en évitant les corrélations élevées
    selected_numeric = []
    for feature, _ in sorted_features:
        if not selected_numeric:
            selected_numeric.append(feature)
        else:
            # Vérifier les corrélations avec les features déjà sélectionnées
            correlations = [abs(df[feature].corr(df[f])) for f in selected_numeric]
            if all(corr <= 0.85 for corr in correlations):
                selected_numeric.append(feature)
    
    print("Features numériques sélectionnées:", selected_numeric)
    
    # 2. Sélection des features catégorielles
    print("2. Sélection des features catégorielles")
    print("-" * 40)
    
    # Calculer les scores pour chaque feature catégorielle
    categorical_scores = {}
    for feature in categorical_features:
        total_score = 0
        for label in labels:
            groups = [group[label].values for name, group in df.groupby(feature, observed=False)]
            if len(groups) == 2:
                # Test de Wilcoxon
                _, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                # Test de Kruskal-Wallis
                _, p_value = stats.kruskal(*groups)
            total_score += -np.log(p_value) if p_value > 0 else 1000
        categorical_scores[feature] = total_score
    
    # Trier les features catégorielles par score
    sorted_categorical = sorted(categorical_scores.items(), key=lambda x: x[1], reverse=True)
    selected_categorical = [f[0] for f in sorted_categorical]
    
    # Tester l'indépendance entre paires de features catégorielles
    to_eliminate = set()
    for i in range(len(selected_categorical)):
        if selected_categorical[i] in to_eliminate:
            continue
        for j in range(i + 1, len(selected_categorical)):
            if selected_categorical[j] in to_eliminate:
                continue
            
            # Test de Fisher
            contingency = pd.crosstab(df[selected_categorical[i]], df[selected_categorical[j]])
            _, p_value = stats.fisher_exact(contingency)
            
            if p_value < 0.05:  # Dépendance détectée
                # Garder celle avec le score le plus élevé
                if categorical_scores[selected_categorical[i]] >= categorical_scores[selected_categorical[j]]:
                    to_eliminate.add(selected_categorical[j])
                else:
                    to_eliminate.add(selected_categorical[i])
                    break
    
    selected_categorical = [f for f in selected_categorical if f not in to_eliminate]
    print("Features catégorielles sélectionnées:", selected_categorical)
    
    # 3. Dernière passe avec tests statistiques
    print("3. Dernière passe avec tests statistiques")
    print("-" * 40)
    
    categorical_to_eliminate = set()
    for cat_feature in selected_categorical:
        for num_feature in selected_numeric:
            groups = [group[num_feature].values for name, group in df.groupby(cat_feature, observed=False)]
            
            if len(groups) == 2:
                # Test de Wilcoxon
                _, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                # Test de Kruskal-Wallis
                _, p_value = stats.kruskal(*groups)
            
            if p_value < 0.05:
                categorical_to_eliminate.add(cat_feature)
                break
    
    final_categorical = [f for f in selected_categorical if f not in categorical_to_eliminate]
    
    # Mettre à jour les ensembles de features
    features_to_keep.update(selected_numeric + final_categorical)
    features_to_eliminate.update(
        [f for f in numeric_features if f not in selected_numeric] +
        [f for f in categorical_features if f not in final_categorical]
    )
    
    print("Résultats finaux:")
    print("Features conservées:", sorted(features_to_keep))
    print("Features éliminées:", sorted(features_to_eliminate))
    
    # Créer un tableau récapitulatif
    print("Tableau récapitulatif des features :")
    summary_data = []
    
    # Features numériques
    for feature in numeric_features:
        status = "Conservée" if feature in features_to_keep else "Éliminée"
        reason = "Corrélation suffisante avec les labels" if feature in features_to_keep else "Corrélation trop faible ou redondante"
        summary_data.append({
            'Feature': feature,
            'Type': 'Numérique',
            'Status': status,
            'Raison': reason
        })
    
    # Features catégorielles
    for feature in categorical_features:
        status = "Conservée" if feature in features_to_keep else "Éliminée"
        if feature in features_to_keep:
            reason = "Score élevé et indépendance validée"
        elif feature in to_eliminate:
            reason = "Dépendance avec une autre feature catégorielle"
        elif feature in categorical_to_eliminate:
            reason = "Dépendance avec une feature numérique"
        else:
            reason = "Score insuffisant"
        summary_data.append({
            'Feature': feature,
            'Type': 'Catégorielle',
            'Status': status,
            'Raison': reason
        })
    
    # Créer et afficher le DataFrame
    summary_df = pd.DataFrame(summary_data)
    display(summary_df.style
        .set_properties(**{'text-align': 'left'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ])
        .hide(axis='index')
    )
'''))

# Cellule 11: Analyse des distributions des labels
cells.append(nbf.v4.new_markdown_cell("""## 11. Analyse des distributions des labels

### Objectif
Analyser la distribution des variables cibles (labels) pour identifier des transformations potentielles qui pourraient améliorer leur adéquation avec un modèle linéaire.

Pour chaque label, nous allons :
1. Visualiser sa distribution avec un histogramme et une courbe de densité
2. Calculer les statistiques descriptives (moyenne, médiane, écart-type, skewness, kurtosis)
3. Tester différentes transformations (log, racine carrée, carré, cube)
4. Visualiser les distributions transformées
5. Calculer les statistiques des distributions transformées
6. Recommander les transformations les plus pertinentes
"""))

cells.append(nbf.v4.new_code_cell('''# Fonction pour analyser la distribution d'une variable
def analyze_distribution(df, column):
    """
    Analyse la distribution d'une variable et teste différentes transformations.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        column (str): Nom de la colonne à analyser
    """
    print(f"Analyse de la distribution de {column}")
    print("-" * 60)
    
    # Statistiques descriptives
    stats = {
        'Moyenne': df[column].mean(),
        'Médiane': df[column].median(),
        'Écart-type': df[column].std(),
        'Skewness': df[column].skew(),
        'Kurtosis': df[column].kurtosis()
    }
    
    print("Statistiques descriptives :")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Analyse de la distribution de {column}', fontsize=16)
    
    # Distribution originale
    sns.histplot(data=df, x=column, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution originale')
    
    # Transformation log
    if (df[column] > 0).all():
        log_col = np.log(df[column])
        sns.histplot(data=log_col, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Transformation log')
        print("Statistiques après transformation log :")
        print(f"Skewness: {log_col.skew():.2f}")
        print(f"Kurtosis: {log_col.kurtosis():.2f}")
    
    # Transformation racine carrée
    if (df[column] >= 0).all():
        sqrt_col = np.sqrt(df[column])
        sns.histplot(data=sqrt_col, kde=True, ax=axes[0, 2])
        axes[0, 2].set_title('Transformation racine carrée')
        print("Statistiques après transformation racine carrée :")
        print(f"Skewness: {sqrt_col.skew():.2f}")
        print(f"Kurtosis: {sqrt_col.kurtosis():.2f}")
    
    # Transformation carré
    square_col = df[column] ** 2
    sns.histplot(data=square_col, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Transformation carré')
    print("Statistiques après transformation carré :")
    print(f"Skewness: {square_col.skew():.2f}")
    print(f"Kurtosis: {square_col.kurtosis():.2f}")
    
    # Transformation cube
    cube_col = df[column] ** 3
    sns.histplot(data=cube_col, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Transformation cube')
    print("Statistiques après transformation cube :")
    print(f"Skewness: {cube_col.skew():.2f}")
    print(f"Kurtosis: {cube_col.kurtosis():.2f}")
    
    # Transformation puissance 4
    power4_col = df[column] ** 4
    sns.histplot(data=power4_col, kde=True, ax=axes[1, 2])
    axes[1, 2].set_title('Transformation puissance 4')
    print("Statistiques après transformation puissance 4 :")
    print(f"Skewness: {power4_col.skew():.2f}")
    print(f"Kurtosis: {power4_col.kurtosis():.2f}")
    
    # Transformation racine cubique
    if (df[column] >= 0).all():
        cbrt_col = np.cbrt(df[column])
        sns.histplot(data=cbrt_col, kde=True, ax=axes[2, 0])
        axes[2, 0].set_title('Transformation racine cubique')
        print("Statistiques après transformation racine cubique :")
        print(f"Skewness: {cbrt_col.skew():.2f}")
        print(f"Kurtosis: {cbrt_col.kurtosis():.2f}")
    
    # Transformation Box-Cox
    if (df[column] > 0).all():
        from scipy import stats
        boxcox_col, _ = stats.boxcox(df[column])
        sns.histplot(data=boxcox_col, kde=True, ax=axes[2, 1])
        axes[2, 1].set_title('Transformation Box-Cox')
        print("Statistiques après transformation Box-Cox :")
        print(f"Skewness: {pd.Series(boxcox_col).skew():.2f}")
        print(f"Kurtosis: {pd.Series(boxcox_col).kurtosis():.2f}")
    
    # QQ-plot
    probplot(df[column], dist="norm", plot=axes[2, 2])
    axes[2, 2].set_title('QQ-plot')
    
    plt.tight_layout()
    plt.show()

# Analyser les distributions des labels
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Analyse des distributions des labels pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les labels (préfixe 'l_')
    labels = [col for col in df.columns if col.startswith('l_')]
    
    for label in labels:
        analyze_distribution(df, label)
'''))

# Cellule 12: Analyse GPT
cells.append(nbf.v4.new_markdown_cell("""## 12. Analyse GPT des distributions

### Objectif
Utiliser un modèle GPT pour analyser les résultats des transformations et recommander la transformation la plus appropriée pour chaque label dans le cadre d'un modèle linéaire.
"""))

cells.append(nbf.v4.new_code_cell('''# Stocker les résultats de l'analyse GPT et les features à éliminer
gpt_results = {}
features_to_eliminate = set()  # Ensemble pour stocker les features à éliminer

# Collecter les statistiques pour chaque label
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Analyse GPT des distributions pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les labels (préfixe 'l_')
    labels = [col for col in df.columns if col.startswith('l_')]
    
    # Collecter les statistiques pour chaque label
    label_stats = {}
    for label in labels:
        # Distribution originale
        original_stats = {
            'skewness': df[label].skew(),
            'kurtosis': df[label].kurtosis()
        }
        
        # Transformations
        transformations = {}
        
        # Log
        if (df[label] > 0).all():
            log_col = np.log(df[label])
            transformations['log'] = {
                'skewness': log_col.skew(),
                'kurtosis': log_col.kurtosis()
            }
        
        # Racine carrée
        if (df[label] >= 0).all():
            sqrt_col = np.sqrt(df[label])
            transformations['sqrt'] = {
                'skewness': sqrt_col.skew(),
                'kurtosis': sqrt_col.kurtosis()
            }
        
        # Carré
        square_col = df[label] ** 2
        transformations['square'] = {
            'skewness': square_col.skew(),
            'kurtosis': square_col.kurtosis()
        }
        
        # Cube
        cube_col = df[label] ** 3
        transformations['cube'] = {
            'skewness': cube_col.skew(),
            'kurtosis': cube_col.kurtosis()
        }
        
        # Puissance 4
        power4_col = df[label] ** 4
        transformations['power4'] = {
            'skewness': power4_col.skew(),
            'kurtosis': power4_col.kurtosis()
        }
        
        # Racine cubique
        if (df[label] >= 0).all():
            cbrt_col = np.cbrt(df[label])
            transformations['cbrt'] = {
                'skewness': cbrt_col.skew(),
                'kurtosis': cbrt_col.kurtosis()
            }
        
        # Box-Cox
        if (df[label] > 0).all():
            boxcox_col, _ = stats.boxcox(df[label])
            transformations['boxcox'] = {
                'skewness': pd.Series(boxcox_col).skew(),
                'kurtosis': pd.Series(boxcox_col).kurtosis()
            }
        
        label_stats[label] = {
            'original': original_stats,
            'transformations': transformations
        }
    
    # Analyser avec GPT et stocker les résultats
    gpt_results[file_name] = analyze_distributions_with_gpt(file_name, label_stats)
    
    # Stocker les features à éliminer (ajoutez ici la logique pour identifier les features à éliminer)
    # Par exemple, si vous avez une fonction qui identifie les features à éliminer :
    # features_to_eliminate.update(identify_features_to_eliminate(df))'''))

# Cellule 13: Transformation des labels et visualisation
cells.append(nbf.v4.new_markdown_cell("""## 13. Transformation des labels et visualisation

### Objectif
Transformer les labels selon les recommandations de GPT et visualiser l'impact des transformations sur les relations avec les features conservées uniquement.
"""))

cells.append(nbf.v4.new_code_cell('''# Transformer les labels selon les recommandations de GPT
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Transformation des labels pour {file_name}")
    print(f"{'='*80}")
    
    # Récupérer les transformations recommandées depuis les résultats stockés
    transformations = gpt_results[file_name]
    
    if transformations:
        # Créer une copie du DataFrame pour les transformations
        df_transformed = df.copy()
        
        # Appliquer les transformations
        for item in transformations:
            label = item['label']
            transformation = item['transformation']
            
            if transformation and transformation.lower() != 'none':
                print(f"Transformation de {label} avec {transformation}:")
                
                # Appliquer la transformation appropriée
                if transformation.lower() == 'log':
                    df_transformed[label] = np.log(df[label])
                elif transformation.lower() == 'sqrt':
                    df_transformed[label] = np.sqrt(df[label])
                elif transformation.lower() == 'square':
                    df_transformed[label] = df[label] ** 2
                elif transformation.lower() == 'cube':
                    df_transformed[label] = df[label] ** 3
                elif transformation.lower() == 'power4':
                    df_transformed[label] = df[label] ** 4
                elif transformation.lower() == 'cbrt':
                    df_transformed[label] = np.cbrt(df[label])
                elif transformation.lower() == 'boxcox':
                    df_transformed[label], _ = stats.boxcox(df[label])
                
                # Afficher les statistiques avant/après
                print(f"Avant transformation - Skewness: {df[label].skew():.2f}, Kurtosis: {df[label].kurtosis():.2f}")
                print(f"Après transformation - Skewness: {df_transformed[label].skew():.2f}, Kurtosis: {df_transformed[label].kurtosis():.2f}")
        
        # Ne considérer que les features conservées
        features = [col for col in df.columns if col.startswith('f_') and col in features_to_keep]
        labels = [col for col in df.columns if col.startswith('l_')]
        
        for label in labels:
            print(f"Graphiques pour {label}:")
            n_features = len(features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten()
            
            for i, feature in enumerate(features):
                # Graphique avant transformation
                axes[i].scatter(df[feature], df[label], alpha=0.5, label='Original')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{feature} vs {label}')
                axes[i].legend()
                
                # Graphique après transformation
                if label in df_transformed.columns:
                    axes[i].scatter(df[feature], df_transformed[label], alpha=0.5, color='red', label='Transformé')
                    axes[i].legend()
            
            # Cacher les axes inutilisés
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # Afficher les matrices de corrélation avant/après
            print("Matrices de corrélation:")
            
            # Avant transformation
            corr_before = df[features + [label]].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_before, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Matrice de corrélation avant transformation - {label}')
            plt.tight_layout()
            plt.show()
            
            # Après transformation
            if label in df_transformed.columns:
                corr_after = df_transformed[features + [label]].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_after, annot=True, cmap='coolwarm', center=0)
                plt.title(f'Matrice de corrélation après transformation - {label}')
                plt.tight_layout()
                plt.show()'''))

# Cellule 13.5: Création de features polynomiales
cells.append(nbf.v4.new_markdown_cell("""## 13.5. Création de features polynomiales pour les relations non-linéaires

### Objectif
Pour chaque feature qui présente une relation non-linéaire avec un des labels, créer des transformations polynomiales :
- log
- racine carrée
- racine cubique
- carré
- cube
- puissance 4

Ces nouvelles features seront utilisées dans la sélection de modèles qui suit.

### Critère de non-linéarité
Une feature est considérée comme ayant une relation non-linéaire potentielle avec le label si sa corrélation est modérée, c'est-à-dire entre 0.33 et 0.66 (en valeur absolue).
"""))

cells.append(nbf.v4.new_code_cell('''import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def create_polynomial_features(df, features_to_transform):
    """
    Crée des features polynomiales pour les colonnes spécifiées.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        features_to_transform (list): Liste des noms de colonnes à transformer
        
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features polynomiales
    """
    # Créer une copie du DataFrame pour ne pas modifier l'original
    df_poly = df.copy()
    
    # Définir les transformations
    transformations = {
        'log': lambda x: np.log1p(x),  # log1p pour gérer les valeurs nulles ou négatives
        'sqrt': lambda x: np.sqrt(x),
        'cbrt': lambda x: np.cbrt(x),
        'square': lambda x: np.square(x),
        'cube': lambda x: np.power(x, 3),
        'power4': lambda x: np.power(x, 4)
    }
    
    # Appliquer les transformations pour chaque feature
    for feature in features_to_transform:
        for trans_name, trans_func in transformations.items():
            # Créer le nom de la nouvelle colonne
            new_col_name = f"{feature}_{trans_name}"
            
            # Appliquer la transformation
            df_poly[new_col_name] = trans_func(df[feature])
    
    return df_poly

def get_polynomial_features(df, target_col):
    """
    Identifie les features qui pourraient bénéficier de transformations polynomiales
    en analysant leur relation avec la variable cible.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        target_col (str): Nom de la colonne cible
        
    Returns:
        list: Liste des noms de colonnes à transformer
    """
    # Calculer le coefficient de corrélation pour chaque feature
    correlations = df.corr()[target_col].abs()
    
    # Sélectionner les features avec une corrélation modérée (entre 0.33 et 0.66)
    features_to_transform = correlations[
        (correlations > 0.33) & (correlations < 0.66)
    ].index.tolist()
    
    # Exclure la colonne cible si elle est dans la liste
    if target_col in features_to_transform:
        features_to_transform.remove(target_col)
    
    return features_to_transform

# Dictionnaire pour stocker les résultats de l'analyse
polynomial_analysis_results = {}

# Appliquer les transformations polynomiales pour chaque fichier
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Création des features polynomiales pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les labels
    labels = [col for col in df.columns if col.startswith('l_')]
    
    # Pour chaque label, identifier les features à transformer
    for label in labels:
        print(f"\\nAnalyse pour le label {label}:")
        
        # Identifier les features à transformer
        features_to_transform = get_polynomial_features(df, label)
        
        if features_to_transform:
            print(f"Features sélectionnées pour transformation: {features_to_transform}")
            
            # Créer les features polynomiales
            df_poly = create_polynomial_features(df, features_to_transform)
            
            # Afficher les nouvelles colonnes créées
            new_columns = [col for col in df_poly.columns if col not in df.columns]
            print(f"Nouvelles colonnes créées: {len(new_columns)}")
            print("Exemple de nouvelles colonnes:")
            display(df_poly[new_columns[:5]].head())
            
            # Stocker les résultats de l'analyse
            polynomial_analysis_results[(file_name, label)] = {
                'features': features_to_transform,
                'new_columns': new_columns,
                'correlations': {f: df[f].corr(df[label]) for f in features_to_transform}
            }
            
            # Mettre à jour le DataFrame dans normalized_data_converted
            normalized_data_converted[file_name] = df_poly
        else:
            print("Aucune feature sélectionnée pour transformation.")

# Créer un tableau de synthèse
print("\\nTableau de synthèse des features polynomiales créées:")
summary_data = []
for (file_name, label), results in polynomial_analysis_results.items():
    for feature in results['features']:
        correlation = results['correlations'][feature]
        summary_data.append({
            'Fichier': file_name,
            'Label': label,
            'Feature originale': feature,
            'Corrélation originale': f"{correlation:.3f}",
            'Critère de non-linéarité': "Corrélation modérée (0.33 < |r| < 0.66) suggérant une relation non-linéaire potentielle",
            'Transformations appliquées': "log, sqrt, cbrt, square, cube, power4"
        })

summary_df = pd.DataFrame(summary_data)
display(summary_df.style
    .set_properties(**{'text-align': 'left'})
    .set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ])
    .hide(axis='index')
)'''))

# Cellule 14: Séparation des données et préparation des features
cells.append(nbf.v4.new_markdown_cell("## 14. Séparation des données et préparation des features"))
cells.append(nbf.v4.new_code_cell('''# Prendre le premier fichier comme exemple
file_name = next(iter(normalized_data_converted))
df = normalized_data_converted[file_name]

# Identifier les features et labels
numeric_features = [col for col in df.columns if col.startswith('f_') and df[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in df.columns if col.startswith('f_') and df[col].dtype.name == 'category']
all_features = numeric_features + categorical_features
labels = [col for col in df.columns if col.startswith('l_')]

# Prendre le premier label comme exemple
label = labels[0]

# Créer une copie pour les données transformées
df_transformed = df.copy()

print("Séparation des données et préparation des features")
print("-" * 40)

# Séparer les données en train/test
X = df[all_features]  # Utiliser les données avec features polynomiales
y = df_transformed[label]  # Utiliser les données transformées pour y
y_orig = df[label]  # Garder les valeurs originales pour le RMSE

# Séparation train/test
X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
    X, y, y_orig, test_size=0.2, random_state=42
)

# Créer les transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Créer le ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Créer le pipeline complet
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Ajuster le pipeline sur les données d'entraînement
pipeline.fit(X_train)

# Transformer les données
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Obtenir les noms des colonnes après transformation
feature_names = (
    numeric_features +  # Features numériques standardisées
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()  # Features catégorielles encodées
)

# Créer les DataFrames transformés
X_train_transformed_df = pd.DataFrame(
    X_train_transformed,
    columns=feature_names,
    index=X_train.index
)
X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

# Afficher les dimensions
print(f"Dimensions des données d'entraînement: {X_train_transformed_df.shape}")
print(f"Dimensions des données de test: {X_test_transformed_df.shape}")
print("-" * 40)

# Vérifier les features disponibles
print("Features disponibles dans normalized_data_converted[file_name]:")
print(normalized_data_converted[file_name].columns.tolist())

print("Features demandées (all_features):")
print(all_features)'''))

# Cellule 15: Sélection de modèles linéaires optimaux avec features centrées-réduites
cells.append(nbf.v4.new_markdown_cell(r"""## 15. Sélection de modèles linéaires optimaux avec features centrées-réduites

### Objectif
Reproduire l'analyse précédente en :
1. Encodant les variables catégorielles en one-hot
2. Centrant et réduisant toutes les features (numériques, polynomiales et catégorielles encodées)
3. Sélectionnant les meilleurs modèles

Cette approche permet de :
- Mettre toutes les features sur la même échelle
- Faciliter l'interprétation des coefficients
- Améliorer la stabilité numérique des calculs
- Traiter correctement les variables catégorielles
- Capturer les relations non-linéaires via les features polynomiales créées dans la cellule 13.5
"""))

cells.append(nbf.v4.new_code_cell(r"""from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dictionnaire pour stocker les meilleures variables sélectionnées
best_features_dict = {}

# Appliquer les trois méthodes pour chaque label avec features centrées-réduites
for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Sélection de modèles avec features centrées-réduites pour {file_name}")
    print(f"{'='*80}")

    # Identifier les features polynomiales créées dans la cellule 13.5
    polynomial_features = []
    for (fname, label), results in polynomial_analysis_results.items():
        if fname == file_name:
            polynomial_features.extend(results['new_columns'])
    
    # Identifier les features originales
    original_features = [col for col in df.columns if col.startswith('f_') and col not in polynomial_features]
    
    # Combiner toutes les features en évitant les doublons
    all_features = list(set(original_features + polynomial_features))
    
    # Identifier les labels
    labels = [col for col in df.columns if col.startswith('l_')]

    # Identifier les features numériques et catégorielles parmi les features originales
    numeric_features = [f for f in original_features if df[f].dtype in ['int64', 'float64']]
    categorical_features = [f for f in original_features if df[f].dtype.name == 'category']

    # Créer un DataFrame pour les labels transformés
    df_transformed = df.copy()

    # Appliquer les transformations recommandées par GPT
    transformations = gpt_results[file_name]
    if transformations:
        for item in transformations:
            label = item['label']
            transformation = item['transformation']

            if transformation and transformation.lower() != 'none':
                if transformation.lower() == 'log':
                    df_transformed[label] = np.log(df[label])
                elif transformation.lower() == 'sqrt':
                    df_transformed[label] = np.sqrt(df[label])
                elif transformation.lower() == 'square':
                    df_transformed[label] = df[label] ** 2
                elif transformation.lower() == 'cube':
                    df_transformed[label] = df[label] ** 3
                elif transformation.lower() == 'power4':
                    df_transformed[label] = df[label] ** 4
                elif transformation.lower() == 'cbrt':
                    df_transformed[label] = np.cbrt(df[label])
                elif transformation.lower() == 'boxcox':
                    df_transformed[label], _ = stats.boxcox(df[label])

    for label in labels:
        print(f"Modélisation de {label}")
        print("-" * 40)

        # Vérifier que toutes les features existent dans le DataFrame
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            print(f"Attention: Les features suivantes sont manquantes dans le DataFrame:")
            for f in missing_features:
                print(f"- {f}")
            print("Ces features seront ignorées dans l'analyse.")
            all_features = [f for f in all_features if f in df.columns]

        # Séparer les données en train/test
        X = df[all_features]
        y = df_transformed[label]
        y_orig = df[label]  # Valeurs originales pour le RMSE

        X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
            X, y, y_orig, test_size=0.3, random_state=42
        )

        # Créer le préprocesseur pour le one-hot encoding et la standardisation
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features + polynomial_features),  # Standardiser les features numériques et polynomiales
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ],
            remainder='passthrough'
        )

        # Appliquer le préprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Récupérer les noms des features après one-hot encoding
        feature_names = (
            numeric_features + polynomial_features +  # Features numériques et polynomiales standardisées
            preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()  # Features catégorielles encodées
        )

        # Convertir en DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_processed,
            columns=feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index
        )

        # Récupérer la transformation appliquée pour ce label
        transformation = next((item['transformation'] for item in transformations if item['label'] == label), None)

        print("1. Forward Selection")
        forward_features = forward_selection(X_train_scaled, y_train, X_test_scaled, y_test)
        forward_metrics = calculate_metrics(
            X_train_scaled[forward_features], y_train,
            X_test_scaled[forward_features], y_test,
            y_orig_train, y_orig_test,
            transformation
        )

        print("2. Backward Elimination")
        backward_features = backward_elimination(X_train_scaled, y_train, X_test_scaled, y_test)
        backward_metrics = calculate_metrics(
            X_train_scaled[backward_features], y_train,
            X_test_scaled[backward_features], y_test,
            y_orig_train, y_orig_test,
            transformation
        )

        print("3. Stepwise Selection")
        stepwise_features = stepwise_selection(X_train_scaled, y_train, X_test_scaled, y_test)
        stepwise_metrics = calculate_metrics(
            X_train_scaled[stepwise_features], y_train,
            X_test_scaled[stepwise_features], y_test,
            y_orig_train, y_orig_test,
            transformation
        )

        # Créer un tableau comparatif des résultats
        results_data = {
            'Méthode': ['Forward', 'Backward', 'Stepwise'],
            'Features': [forward_features, backward_features, stepwise_features],
            'Nombre de features': [len(forward_features), len(backward_features), len(stepwise_features)],
            'R² ajusté (train)': [forward_metrics['r2_adj'], backward_metrics['r2_adj'], stepwise_metrics['r2_adj']],
            'R² (test)': [forward_metrics['test_r2'], backward_metrics['test_r2'], stepwise_metrics['test_r2']],
            'MSE (test)': [forward_metrics['test_mse'], backward_metrics['test_mse'], stepwise_metrics['test_mse']],
            'RMSE (test)': [forward_metrics['test_rmse'], backward_metrics['test_rmse'], stepwise_metrics['test_rmse']],
            'AIC': [forward_metrics['aic'], backward_metrics['aic'], stepwise_metrics['aic']],
            'BIC': [forward_metrics['bic'], backward_metrics['bic'], stepwise_metrics['bic']]
        }

        results_df = pd.DataFrame(results_data)
        print(f"Comparaison des modèles pour {label} (après transformation et centrage-réduction):")
        display(results_df.style
            .format({
                'R² ajusté (train)': '{:.4f}',
                'R² (test)': '{:.4f}',
                'MSE (test)': '{:.4f}',
                'RMSE (test)': '{:.4f}',
                'AIC': '{:.2f}',
                'BIC': '{:.2f}'
            })
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'td', 'props': [('text-align', 'left')]}
            ])
            .hide(axis='index')
        )

        # Afficher les coefficients du meilleur modèle (basé sur le RMSE test)
        best_method_idx = results_df['RMSE (test)'].argmin()  # Changé de R² à RMSE
        best_method = results_df.iloc[best_method_idx]
        best_features = best_method['Features']

        # Stocker les meilleures variables pour ce label
        best_features_dict[(file_name, label)] = {
            'features': best_features,
            'transformation': transformation,
            'method': best_method['Méthode'],
            'metrics': {
                'r2_test': best_method['R² (test)'],
                'rmse_test': best_method['RMSE (test)'],
                'mse_test': best_method['MSE (test)']
            }
        }

        print(f"Meilleur modèle ({best_method['Méthode']}) pour {label} (après transformation et centrage-réduction):")
        best_model = calculate_metrics(
            X_train_scaled[best_features], y_train,
            X_test_scaled[best_features], y_test,
            y_orig_train, y_orig_test,
            transformation
        )['model']
        print(best_model.summary())

        # Afficher la transformation appliquée et les performances
        print(f"Transformation appliquée : {transformation}")
        print("Performances sur le jeu de test:")
        print(f"R² test: {best_method['R² (test)']:.4f}")
        print(f"MSE test: {best_method['MSE (test)']:.4f}")
        print(f"RMSE test: {best_method['RMSE (test)']:.4f}")

        # Afficher les coefficients standardisés avec leur interprétation
        print("Coefficients standardisés:")
        coef_df = pd.DataFrame({
            'Feature': best_features,
            'Coefficient': best_model.params[1:],  # Ignorer l'intercept
            'Std Error': best_model.bse[1:],
            't-value': best_model.tvalues[1:],
            'p-value': best_model.pvalues[1:]
        })

        # Ajouter une colonne d'interprétation
        coef_df['Interprétation'] = coef_df.apply(lambda row: 
            f"Une augmentation d'un écart-type de {row['Feature']} est associée à une {'augmentation' if row['Coefficient'] > 0 else 'diminution'} de {abs(row['Coefficient']):.4f} unités de {label}"
            if row['Feature'] in numeric_features + polynomial_features
            else f"La catégorie {row['Feature']} est associée à une {'augmentation' if row['Coefficient'] > 0 else 'diminution'} de {abs(row['Coefficient']):.4f} unités de {label} par rapport à la catégorie de référence",
            axis=1
        )

        display(coef_df.style
            .format({
                'Coefficient': '{:.4f}',
                'Std Error': '{:.4f}',
                't-value': '{:.4f}',
                'p-value': '{:.4f}'
            })
            .set_properties(**{'text-align': 'left'})
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]},
                {'selector': 'td', 'props': [('text-align', 'left')]}
            ])
            .hide(axis='index')
        )
        
        # Afficher les features polynomiales sélectionnées
        polynomial_features_selected = [f for f in best_features if f in polynomial_features]
        if polynomial_features_selected:
            print("\\nFeatures polynomiales sélectionnées:")
            for feature in polynomial_features_selected:
                print(f"- {feature}")
                
            # Afficher les informations sur les features polynomiales sélectionnées
            print("\\nInformations sur les features polynomiales sélectionnées:")
            for (fname, lbl), results in polynomial_analysis_results.items():
                if fname == file_name:
                    for orig_feature in results['features']:
                        for poly_feature in polynomial_features_selected:
                            if poly_feature.startswith(orig_feature):
                                print(f"- {poly_feature} (transformée de {orig_feature})")
                                print(f"  Corrélation originale: {results['correlations'][orig_feature]:.4f}")"""))

# Cellule 16: Entraînement avec scikit-learn
cells.append(nbf.v4.new_markdown_cell('''## 16. Entraînement avec scikit-learn

### Objectif
Entraîner un modèle linéaire avec scikit-learn en utilisant :
1. Les meilleures variables sélectionnées dans la cellule 10 (features originales)
2. Les features polynomiales créées dans la cellule 13.5
3. Cross-validation pour une meilleure estimation des performances
4. Pipeline complet avec preprocessing (StandardScaler et OneHotEncoder)
5. Comparaison avec le meilleur modèle précédent
'''))

cells.append(nbf.v4.new_code_cell('''from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def inverse_transform(y, transformation):
    """
    Applique la transformation inverse aux prédictions.
    """
    if not transformation or transformation.lower() == 'none':
        return y
    elif transformation.lower() == 'log':
        return np.exp(y)
    elif transformation.lower() == 'sqrt':
        return np.square(y)
    elif transformation.lower() == 'square':
        return np.sqrt(y)
    elif transformation.lower() == 'cube':
        return np.cbrt(y)
    elif transformation.lower() == 'power4':
        return np.power(y, 1/4)
    elif transformation.lower() == 'cbrt':
        return np.power(y, 3)
    elif transformation.lower() == 'boxcox':
        # Pour BoxCox, on ne peut pas faire la transformation inverse exacte
        # car on n'a pas gardé le paramètre lambda. On retourne les valeurs telles quelles
        return y
    return y

def remap_features(features, onehot_features):
    # Convertir la liste des features en set pour plus d'efficacité
    features_set = set(features)

    # Supprimer les anciennes features
    for replaced_features in onehot_features.values():
        features_set -= set(replaced_features)

    # Ajouter les nouvelles features (les clés) si elles ne sont pas déjà présentes
    features_set.update(onehot_features.keys())

    # Retourner une liste triée (facultatif)
    return list(features_set)

# Dictionnaire pour stocker les résultats des modèles scikit-learn
sklearn_results = {}

for (file_name, label), best_model_info in best_features_dict.items():
    print(f"{'='*80}")
    print(f"Entraînement scikit-learn pour {file_name} - {label}")
    print(f"{'='*80}")
    
    # Récupérer les données
    df = normalized_data_converted[file_name]
    features = best_model_info['features']  # Features sélectionnées dans la cellule 10
    transformation = best_model_info['transformation']
    
    # Identifier les features polynomiales créées dans la cellule 13.5
    polynomial_features = []
    for (fname, lbl), results in polynomial_analysis_results.items():
        if fname == file_name:
            polynomial_features.extend(results['new_columns'])
    
    # Identifier les features polynomiales sélectionnées parmi les features de la cellule 10
    polynomial_features_selected = [f for f in features if f in polynomial_features]
    original_features = [f for f in features if f not in polynomial_features]
    
    print(f"Features originales sélectionnées (cellule 10): {len(original_features)}")
    print(f"Features polynomiales sélectionnées (cellule 13.5): {len(polynomial_features_selected)}")
    
    # Identifier les features numériques et catégorielles parmi les features originales
    numeric_features = []
    categorical_features = []
    onehot_features = {}  # Pour regrouper les features one-hot par variable d'origine
    
    for f in original_features:
        if f in df.columns:  # Feature directe du DataFrame
            if df[f].dtype in ['int64', 'float64']:
                numeric_features.append(f)
            elif df[f].dtype.name == 'category':
                categorical_features.append(f)
        else:  # Feature issue du one-hot encoding
            # Extraire le nom de la variable catégorielle d'origine
            base_feature = '_'.join(f.split('_')[:-1])
            if base_feature not in onehot_features:
                onehot_features[base_feature] = []
            onehot_features[base_feature].append(f)
    
    # Ajouter les variables catégorielles d'origine aux features catégorielles
    categorical_features.extend(onehot_features.keys())
    
    # Préparer les données
    new_features = remap_features(original_features, onehot_features)
    X = df[new_features + polynomial_features_selected]  # Combiner features originales et polynomiales
    y_orig = df[label]  # Valeurs originales
    y = y_orig.copy()   # Valeurs à transformer
    
    # Appliquer la transformation au label si nécessaire
    if transformation and transformation.lower() != 'none':
        if transformation.lower() == 'log':
            y = np.log(y)
        elif transformation.lower() == 'sqrt':
            y = np.sqrt(y)
        elif transformation.lower() == 'square':
            y = y ** 2
        elif transformation.lower() == 'cube':
            y = y ** 3
        elif transformation.lower() == 'power4':
            y = y ** 4
        elif transformation.lower() == 'cbrt':
            y = np.cbrt(y)
        elif transformation.lower() == 'boxcox':
            y, _ = stats.boxcox(y)
    
    # Créer le préprocesseur
    # 1. on repère d'abord les vraies variables catégorielles directes
    direct_cat = [
        f for f in original_features 
        if (f in df.columns and df[f].dtype.name == 'category')
           and f not in onehot_features
    ]
    
    # 2. on construit la liste des transformeurs
    transformers = []
    
    # – scale des numériques
    transformers.append(('num', StandardScaler(), numeric_features))
    
    # – scale des features polynomiales
    if polynomial_features_selected:
        transformers.append(('poly', StandardScaler(), polynomial_features_selected))
    
    # – OHE "classique" des catégorielles directes
    for cat in direct_cat:
        transformers.append((
            f'ohe_{cat}',
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
            [cat]
        ))
    
    # – OHE "sur mesure" pour chacune des variables one‑hot
    for base_feature, dummy_list in onehot_features.items():
        cats = [col.split('_')[-1] for col in dummy_list]
        transformers.append((
            f'ohe_sel_{base_feature}',
            OneHotEncoder(categories=[cats],
                          drop=None,
                          sparse_output=False,
                          handle_unknown='ignore'),
            [base_feature]
        ))
    
    # 3. on remonte tout dans un ColumnTransformer qui DROPE le reste
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )    
    # Créer le pipeline complet
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Configuration de la cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation pour le R²
    cv_r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
    cv_rmse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    
    # Entraîner le modèle sur toutes les données
    pipeline.fit(X, y)
    
    # Prédictions
    y_pred_transformed = pipeline.predict(X)
    
    # Appliquer la transformation inverse aux prédictions
    y_pred_orig = inverse_transform(y_pred_transformed, transformation)
    
    # Calculer les métriques sur les valeurs originales
    r2 = r2_score(y_orig, y_pred_orig)
    mse = mean_squared_error(y_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    
    # Stocker les résultats
    sklearn_results[(file_name, label)] = {
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_rmse_mean': -cv_rmse_scores.mean(),  # Convertir en positif
        'cv_rmse_std': cv_rmse_scores.std(),
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'model': pipeline,
        'features': features,
        'polynomial_features': polynomial_features_selected,
        'original_features': original_features
    }
    
    # Comparaison avec le modèle précédent
    print("Comparaison des performances:")
    print("-" * 40)
    print("Modèle précédent (statsmodels):")
    print(f"R² test: {best_model_info['metrics']['r2_test']:.4f}")
    print(f"RMSE test: {best_model_info['metrics']['rmse_test']:.4f}")
    print(f"MSE test: {best_model_info['metrics']['mse_test']:.4f}")
    
    print("Modèle scikit-learn:")
    print(f"R² (cross-validation): {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"RMSE (cross-validation): {-cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
    print(f"R² (train): {r2:.4f}")
    print(f"RMSE (train): {rmse:.4f}")
    
    # Afficher les coefficients
    print("Coefficients du modèle:")
    
    # Obtenir directement les noms des colonnes post-prétraitement
    feature_names = pipeline \
       .named_steps['preprocessor'] \
       .get_feature_names_out()
    
    coef_df = pd.DataFrame({
       'Feature':     feature_names,
       'Coefficient': pipeline.named_steps['regressor'].coef_
    })
    
    display(coef_df.style
        .format({'Coefficient': '{:.4f}'})
        .set_properties(**{'text-align': 'left'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ])
        .hide(axis='index')
    )
    
    # Afficher les features polynomiales utilisées
    if polynomial_features_selected:
        print("\\nFeatures polynomiales utilisées:")
        for feature in polynomial_features_selected:
            print(f"- {feature}")
            
        # Afficher les informations sur les features polynomiales sélectionnées
        print("\\nInformations sur les features polynomiales sélectionnées:")
        for (fname, lbl), results in polynomial_analysis_results.items():
            if fname == file_name:
                for orig_feature in results['features']:
                    for poly_feature in polynomial_features_selected:
                        if poly_feature.startswith(orig_feature):
                            print(f"- {poly_feature} (transformée de {orig_feature})")
                            print(f"  Corrélation originale: {results['correlations'][orig_feature]:.4f}")'''))

# Cellule 17: Analyse du meilleur modèle
cells.append(nbf.v4.new_markdown_cell('''## 17. Analyse du meilleur modèle

### Objectif
Analyser en détail le meilleur modèle identifié dans les cellules précédentes, en se concentrant sur :
1. L'importance des features (originales et polynomiales)
2. L'interprétation des coefficients
3. La visualisation des relations non-linéaires
4. Les prédictions et les résidus

Le modèle analysé est celui qui a obtenu les meilleures performances parmi tous les modèles testés dans les cellules précédentes.
'''))

cells.append(nbf.v4.new_code_cell('''import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_feature_importance(model, feature_names):
    """
    Affiche l'importance des features basée sur les coefficients standardisés.
    """
    coefs = model.named_steps['regressor'].coef_
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefs)
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Importance des features (coefficients standardisés)')
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, label):
    """
    Affiche les graphiques de diagnostic des résidus.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analyse des résidus - {label}')
    
    # Histogramme des résidus
    sns.histplot(residuals, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution des résidus')
    axes[0, 0].axvline(x=0, color='r', linestyle='--')
    
    # QQ-plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('QQ-plot des résidus')
    
    # Résidus vs Prédictions
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Résidus vs Prédictions')
    axes[1, 0].set_xlabel('Prédictions')
    axes[1, 0].set_ylabel('Résidus')
    
    # Résidus vs Observations
    axes[1, 1].scatter(y_true, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Résidus vs Observations')
    axes[1, 1].set_xlabel('Observations')
    axes[1, 1].set_ylabel('Résidus')
    
    plt.tight_layout()
    plt.show()

def plot_nonlinear_relationships(df, features, label, model):
    """
    Affiche les relations non-linéaires entre les features polynomiales et le label.
    """
    for feature in features:
        if any(trans in feature for trans in ['_log', '_sqrt', '_cbrt', '_square', '_cube', '_power4']):
            plt.figure(figsize=(10, 6))
            plt.scatter(df[feature], df[label], alpha=0.5, label='Données')
            
            # Trier les valeurs pour une meilleure visualisation
            sorted_idx = np.argsort(df[feature])
            plt.plot(df[feature].iloc[sorted_idx], 
                    model.predict(df)[sorted_idx], 
                    color='red', 
                    label='Prédictions')
            
            plt.title(f'Relation entre {feature} et {label}')
            plt.xlabel(feature)
            plt.ylabel(label)
            plt.legend()
            plt.tight_layout()
            plt.show()

# Analyser chaque meilleur modèle
for (file_name, label), model_info in sklearn_results.items():
    print(f"{'='*80}")
    print(f"Analyse du meilleur modèle pour {file_name} - {label}")
    print(f"{'='*80}")
    
    # Récupérer les données
    df = normalized_data_converted[file_name]
    model = model_info['model']
    features = model_info['features']  # Toutes les features du meilleur modèle
    polynomial_features = model_info['polynomial_features']
    original_features = model_info['original_features']
    
    print(f"Nombre total de features utilisées: {len(features)}")
    print(f"- Features originales: {len(original_features)}")
    print(f"- Features polynomiales: {len(polynomial_features)}")
    
    # Obtenir les noms des features après preprocessing
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    # 1. Importance des features
    print("1. Importance des features:")
    plot_feature_importance(model, feature_names)
    
    # 2. Analyse des résidus
    print("2. Analyse des résidus:")
    y_pred = model.predict(df[features])
    y_true = df[label]
    plot_residuals(y_true, y_pred, label)
    
    # 3. Relations non-linéaires
    if polynomial_features:
        print("3. Relations non-linéaires:")
        plot_nonlinear_relationships(df, polynomial_features, label, model)
    
    # 4. Statistiques des résidus
    residuals = y_true - y_pred
    print("4. Statistiques des résidus:")
    print(f"Moyenne des résidus: {residuals.mean():.4f}")
    print(f"Écart-type des résidus: {residuals.std():.4f}")
    print(f"Skewness des résidus: {stats.skew(residuals):.4f}")
    print(f"Kurtosis des résidus: {stats.kurtosis(residuals):.4f}")
    
    # 5. Test de normalité des résidus
    _, p_value = stats.normaltest(residuals)
    print(f"Test de normalité des résidus (p-value): {p_value:.4f}")
    if p_value < 0.05:
        print("→ Les résidus ne suivent pas une distribution normale")
    else:
        print("→ Les résidus suivent une distribution normale")
        
    # 6. Afficher les coefficients avec leur interprétation
    print("6. Coefficients du modèle:")
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.named_steps['regressor'].coef_,
        'Type': ['Polynomiale' if any(trans in f for trans in ['_log', '_sqrt', '_cbrt', '_square', '_cube', '_power4']) else 'Originale' for f in feature_names]
    })
    
    display(coef_df.style
        .format({'Coefficient': '{:.4f}'})
        .set_properties(**{'text-align': 'left'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ])
        .hide(axis='index')
    )'''))

# Ajouter les cellules au notebook
nb['cells'] = cells

# Sauvegarder le notebook
with open('notebooks/data_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 