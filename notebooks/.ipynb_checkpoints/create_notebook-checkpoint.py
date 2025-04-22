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

# Cellule 10: Test d'indépendance entre features catégorielles et numériques
cells.append(nbf.v4.new_markdown_cell("""## 9. Test d'indépendance entre features catégorielles et numériques

### Démarche d'analyse

Pour tester l'indépendance entre une feature catégorielle et une feature numérique, nous utilisons :

- **Test de Wilcoxon** (si la variable catégorielle a exactement 2 classes) :
  * Test non-paramétrique pour comparer deux échantillons indépendants
  * H0 : Les deux distributions sont identiques
  * H1 : Les distributions sont différentes
  * Interprétation : Si p-value < 0.05, on rejette H0 et on conclut à une différence significative

- **Test de Kruskal-Wallis** (si la variable catégorielle a plus de 2 classes) :
  * Test non-paramétrique pour comparer plus de deux échantillons indépendants
  * H0 : Les distributions sont identiques pour tous les groupes
  * H1 : Au moins un groupe a une distribution différente
  * Interprétation : Si p-value < 0.05, on rejette H0 et on conclut à une différence significative
"""))

cells.append(nbf.v4.new_code_cell('''# Test d'indépendance entre features catégorielles et numériques
from scipy.stats import kruskal, mannwhitneyu, probplot

def test_cat_num_independence(df, cat_var, num_var):
    """
    Teste l'indépendance entre une feature catégorielle et une feature numérique.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        cat_var (str): Nom de la feature catégorielle
        num_var (str): Nom de la feature numérique
    
    Returns:
        bool: True si une dépendance est détectée (p-value < 0.05)
    """
    print(f"Test d'indépendance entre {cat_var} et {num_var}")
    print("-" * 60)
    
    # Obtenir les catégories uniques
    categories = df[cat_var].unique()
    n_categories = len(categories)
    
    # Créer les groupes de données numériques pour chaque catégorie
    groups = [df[df[cat_var] == cat][num_var].values for cat in categories]
    
    # Choisir et appliquer le test approprié
    if n_categories == 2:
        # Test de Wilcoxon (Mann-Whitney U)
        stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        test_name = "Test de Wilcoxon"
    else:
        # Test de Kruskal-Wallis
        stat, p_value = kruskal(*groups)
        test_name = "Test de Kruskal-Wallis"
    
    # Affichage des résultats
    print(f"{test_name} :")
    print(f"p-value: {p_value:.4f}")
    
    # Interprétation
    print("Interprétation:")
    if p_value < 0.05:
        print("→ p-value < 0.05 : On rejette l'hypothèse d'indépendance")
        print("→ Il existe une dépendance significative entre les variables")
        return True
    else:
        print("→ p-value ≥ 0.05 : On ne peut pas rejeter l'hypothèse d'indépendance")
        print("→ Pas de preuve statistique d'une dépendance entre les variables")
        return False

# Analyser l'indépendance entre features catégorielles et numériques
dependencies_found = []

for file_name, df in normalized_data_converted.items():
    print(f"{'='*80}")
    print(f"Analyse d'indépendance entre features catégorielles et numériques pour {file_name}")
    print(f"{'='*80}")
    
    # Identifier les features catégorielles et numériques (préfixe 'f_')
    categorical_features = [col for col in df.columns if df[col].dtype.name == 'category' and col.startswith('f_')]
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col.startswith('f_')]
    
    # Tester l'indépendance pour chaque paire de features
    file_dependencies = []
    for cat_var in categorical_features:
        for num_var in numeric_features:
            if test_cat_num_independence(df, cat_var, num_var):
                file_dependencies.append((cat_var, num_var))
    
    if file_dependencies:
        dependencies_found.append((file_name, file_dependencies))

# Synthèse des résultats
print(f"{'='*80}")
print("Synthèse des dépendances détectées")
print(f"{'='*80}")

if not dependencies_found:
    print("Aucune dépendance significative n'a été détectée entre les features catégorielles et numériques.")
else:
    print("Les dépendances significatives suivantes ont été détectées :")
    for file_name, dependencies in dependencies_found:
        print(f"Dans le fichier {file_name} :")
        for cat_var, num_var in dependencies:
            print(f"  - {cat_var} est lié à {num_var}")
'''))

# Cellule 11: Analyse des distributions
cells.append(nbf.v4.new_markdown_cell("""## 10. Analyse des distributions des labels

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
cells.append(nbf.v4.new_markdown_cell("""## 11. Analyse GPT des distributions

### Objectif
Utiliser un modèle GPT pour analyser les résultats des transformations et recommander la transformation la plus appropriée pour chaque label dans le cadre d'un modèle linéaire.
"""))

cells.append(nbf.v4.new_code_cell('''def analyze_distributions_with_gpt(file_name, label_stats):
    """
    Analyse les distributions avec GPT et recommande les transformations appropriées.
    
    Args:
        file_name (str): Nom du fichier analysé
        label_stats (dict): Statistiques des distributions pour chaque label
    """
    # Configuration du prompt
    prompt = f"""En tant qu'expert en data science, analysez les distributions suivantes pour le fichier {file_name} :
    
    {json.dumps(label_stats, indent=2)}
    
    Pour chaque label, indiquez :
    1. Si une transformation serait bénéfique pour un modèle linéaire
    2. Si oui, quelle transformation serait la plus appropriée parmi :
       - Log
       - Racine carrée
       - Carré
       - Cube
       - Puissance 4
       - Racine cubique
       - Box-Cox
    3. Justifiez votre choix en vous basant sur les statistiques fournies
    
    Répondez de manière concise et structurée."""
    
    try:
        # Configuration de l'API OpenAI
        client = openai.OpenAI()
        
        # Appel à l'API
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Modèle le plus adapté pour l'analyse data science
            messages=[
                {"role": "system", "content": "Vous êtes un expert en data science spécialisé dans l'analyse des distributions et la préparation des données pour les modèles linéaires."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Température basse pour des réponses plus déterministes
        )
        
        # Affichage de la réponse
        display(Markdown(f"### Analyse pour {file_name}"))
        display(Markdown(response.choices[0].message.content))
        
    except Exception as e:
        print(f"Erreur lors de l'analyse GPT : {str(e)}")

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
    
    # Analyser avec GPT
    analyze_distributions_with_gpt(file_name, label_stats)'''))

# Ajouter les cellules au notebook
nb['cells'] = cells

# Sauvegarder le notebook
with open('notebooks/data_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 