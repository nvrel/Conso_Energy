import os
import sys
import pandas as pd

def load_data():
    """
    Charge tous les fichiers CSV d'un répertoire dans un dictionnaire de DataFrames,
    puis applique les types de colonnes tels que définis dans column_mapping.

    Args:
        directory (str): Chemin du répertoire contenant les fichiers CSV

    Returns:
        dict: Dictionnaire avec les noms de fichiers comme clés et les DataFrames typés comme valeurs
    """
    # Charger les données avec les bons paramètres
    config_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "config"))
    if config_dir not in sys.path:
        sys.path.append(config_dir)
    from column_config import column_mapping  # column_mapping défini dans config/column_config.py
    directory = os.path.join('..', 'data/csv')
    data = None
    # 1) Récupérer tous les fichiers .csv dans le répertoire
    all_files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    # 2) Les trier par date de modification (mtime), du plus récent au plus ancien
    all_files_sorted = sorted(
        all_files,
        key=lambda f: os.path.getmtime(os.path.join(directory, f)),
        reverse=True
    )

    for file in os.listdir(directory):
        if not file.endswith('.csv'):
            continue

        file_path = os.path.join(directory, file)
        # Lecture basique du CSV (ajustez sep=';' et decimal=',' si votre CSV utilise ce format)
        df = pd.read_csv(file_path)

        # Conversion des types de colonnes en fonction de column_mapping
        #   - column_mapping a pour clé le code d'origine (ex: "X1"), et pour valeur :
        #       { "normalized_name": "f_relative_compactness", "data_type": "numeric" }
        for code, info in column_mapping.items():
            col_name = info["normalized_name"]
            desired_type = info["data_type"]

            # Ne traiter la colonne que si elle existe dans le DataFrame
            if col_name not in df.columns:
                continue

            if desired_type == "numeric":
                # Forcer la colonne en numérique (float). Erreur si une valeur n'est pas convertible.
                df[col_name] = pd.to_numeric(df[col_name], errors="raise")

            elif desired_type.startswith("categorical"):
                # Convertir en type 'category'. On ne connaît pas ici les catégories ordonnées,
                # donc on laisse pandas déterminer l'ordre s'il y en a un.
                df[col_name] = df[col_name].astype("category")

            # autres cas (par ex. "date") si besoin :
            # elif desired_type == "date":
            #     df[col_name] = pd.to_datetime(df[col_name], errors="raise")

        # Stocker le DataFrame typé. On fait l'hypothèse qu'il n'y a qu'un csv dans le répertoire
        data = df
        break

    return data
