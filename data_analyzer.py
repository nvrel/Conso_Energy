import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging_config import setup_logger
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import base64
from io import BytesIO
import json
from openai import OpenAI

class DataAnalyzer:
    def __init__(self, input_dir: str = "data/normalized_csv"):
        self.input_dir = Path(input_dir)
        self.logger = setup_logger(__name__)
        # Set style for all plots
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        self.client = OpenAI()
        
    def analyze_missing_values(self, file_name: str) -> Dict:
        """Analyse les valeurs manquantes dans un fichier CSV"""
        try:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Le fichier {file_name} n'existe pas")
                
            # Lecture du fichier CSV
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
            
            # Calcul des statistiques globales
            total_rows = len(df)
            rows_with_missing = df.isnull().any(axis=1).sum()
            percentage_rows_with_missing = round((rows_with_missing / total_rows) * 100, 2)
            
            # Analyse par colonne
            columns_with_missing = []
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    missing_percentage = round((missing_count / total_rows) * 100, 2)
                    columns_with_missing.append({
                        'name': column,
                        'missing_count': int(missing_count),
                        'missing_percentage': float(missing_percentage)
                    })
            
            return {
                'total_rows': int(total_rows),
                'rows_with_missing': int(rows_with_missing),
                'percentage_missing': float(percentage_rows_with_missing),
                'columns_with_missing': columns_with_missing
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des valeurs manquantes: {str(e)}")
            raise
            
    def handle_missing_values(self, file_name: str, method: str, action: str = 'global', 
                            row_index: Optional[int] = None, custom_value: Optional[float] = None) -> bool:
        """Gère les valeurs manquantes selon la méthode spécifiée"""
        try:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Le fichier {file_name} n'existe pas")
                
            # Lecture du fichier CSV
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
            
            if action == 'global':
                if method == 'mean':
                    df = df.fillna(df.mean())
                elif method == 'median':
                    df = df.fillna(df.median())
                elif method == 'drop_row':
                    df = df.dropna(axis=0)
                elif method == 'drop_column':
                    df = df.dropna(axis=1)
                elif method == 'custom' and custom_value is not None:
                    df = df.fillna(custom_value)
                else:
                    raise ValueError(f"Méthode invalide: {method}")
                    
            elif action == 'row':
                if row_index is None:
                    raise ValueError("L'index de la ligne est requis pour l'action 'row'")
                    
                if method == 'mean':
                    df.iloc[row_index] = df.iloc[row_index].fillna(df.mean())
                elif method == 'median':
                    df.iloc[row_index] = df.iloc[row_index].fillna(df.median())
                elif method == 'drop_row':
                    df = df.drop(row_index)
                elif method == 'drop_column':
                    missing_cols = df.iloc[row_index][df.iloc[row_index].isnull()].index
                    df = df.drop(columns=missing_cols)
                elif method == 'custom':
                    if custom_value is None:
                        raise ValueError("Une valeur personnalisée est requise pour la méthode 'custom'")
                    df.iloc[row_index] = df.iloc[row_index].fillna(custom_value)
                else:
                    raise ValueError(f"Méthode invalide: {method}")
                    
            else:
                raise ValueError(f"Action invalide: {action}")
                
            # Sauvegarde du fichier modifié
            df.to_csv(file_path, sep=';', decimal=',', encoding='utf-8', index=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des valeurs manquantes: {str(e)}")
            raise 

    def generate_distribution_plots(self, file_name: str) -> Dict[str, List[Dict[str, str]]]:
        """Génère des visualisations de distribution pour chaque colonne"""
        try:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Le fichier {file_name} n'existe pas")

            # Charger la configuration des colonnes si elle existe
            config_path = Path('config') / f'column_types_{file_name}.json'
            column_types = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    column_types = json.load(f)

            # Lecture du fichier CSV
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
            
            distributions = {}
            
            for column in df.columns:
                plots = []
                col_type = column_types.get(column, {}).get('type', 'numeric')
                
                if col_type in ['numeric', 'datetime']:
                    # Convertir en numérique si possible
                    try:
                        series = pd.to_numeric(df[column], errors='coerce')
                        
                        # 1. Histogramme avec KDE
                        plt.figure(figsize=(10, 6))
                        sns.histplot(data=series.dropna(), kde=True, stat='density')
                        plt.title(f'Distribution de {column}')
                        plt.xlabel(column)
                        plt.ylabel('Densité')
                        plt.grid(True, alpha=0.3)
                        plots.append(self._save_plot_to_base64())

                        # 2. Box plot
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x=series.dropna())
                        plt.title(f'Boîte à moustaches de {column}')
                        plt.xlabel(column)
                        plt.grid(True, alpha=0.3)
                        plots.append(self._save_plot_to_base64())

                        # 3. Q-Q plot
                        plt.figure(figsize=(10, 6))
                        stats.probplot(series.dropna(), dist="norm", plot=plt)
                        plt.title(f'Q-Q Plot de {column}')
                        plt.grid(True, alpha=0.3)
                        plots.append(self._save_plot_to_base64())

                    except Exception as e:
                        self.logger.warning(f"Impossible de générer les plots pour {column}: {str(e)}")
                        continue

                elif col_type == 'categorical':
                    # Pour les variables catégorielles
                    value_counts = df[column].value_counts()
                    
                    if len(value_counts) <= 20:  # Limiter aux catégories avec moins de 20 valeurs uniques
                        plt.figure(figsize=(12, 6))
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        plt.title(f'Distribution des catégories de {column}')
                        plt.xticks(rotation=45, ha='right')
                        plt.grid(True, alpha=0.3)
                        plots.append(self._save_plot_to_base64())

                if plots:  # Sauvegarder seulement si des plots ont été générés
                    distributions[column] = plots

            return distributions

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des visualisations: {str(e)}")
            raise

    def _save_plot_to_base64(self) -> str:
        """Convertit le plot matplotlib actuel en image base64"""
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def analyze_distribution(self, file_name: str, column_name: str) -> Dict[str, str]:
        """Analyse la distribution d'une colonne et suggère des transformations"""
        try:
            file_path = self.input_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Le fichier {file_name} n'existe pas")

            # Lecture du fichier CSV
            df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
            
            if column_name not in df.columns:
                raise ValueError(f"La colonne {column_name} n'existe pas")

            series = pd.to_numeric(df[column_name], errors='coerce')
            
            # Calculer les statistiques descriptives
            stats_dict = {
                'count': len(series),
                'missing': series.isnull().sum(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max()
            }

            # Tester la normalité
            if len(series.dropna()) > 2:  # Au moins 3 points pour le test
                _, p_value = stats.normaltest(series.dropna())
                stats_dict['normality_p_value'] = p_value
            
            # Préparer le prompt pour GPT-4
            prompt = f"""En tant qu'expert en analyse de données, analyse la distribution suivante et suggère des transformations pertinentes.

Statistiques de la colonne '{column_name}':
- Nombre d'observations: {stats_dict['count']}
- Valeurs manquantes: {stats_dict['missing']}
- Moyenne: {stats_dict['mean']:.2f}
- Médiane: {stats_dict['median']:.2f}
- Écart-type: {stats_dict['std']:.2f}
- Asymétrie (skewness): {stats_dict['skew']:.2f}
- Kurtosis: {stats_dict['kurtosis']:.2f}
- Min: {stats_dict['min']:.2f}
- Max: {stats_dict['max']:.2f}
- P-valeur test de normalité: {stats_dict.get('normality_p_value', 'N/A')}

Basé sur ces statistiques:
1. Décris la forme de la distribution
2. Identifie les potentiels problèmes (asymétrie, outliers, etc.)
3. Suggère des transformations spécifiques (log, racine carrée, carré, etc.) si nécessaire
4. Explique pourquoi ces transformations seraient bénéfiques

Réponds de manière concise et structurée."""

            # Obtenir l'analyse de GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Tu es un expert en analyse de données statistiques, spécialisé dans l'analyse de distributions et la suggestion de transformations de features."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            return {
                'statistics': stats_dict,
                'analysis': response.choices[0].message.content
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de la distribution: {str(e)}")
            raise 