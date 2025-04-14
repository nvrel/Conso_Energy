import pandas as pd
import os
from pathlib import Path
from config.features_config import CSV_CONFIG
from config.normalized_names import (
    FEATURES_NORMALIZED_NAMES,
    TARGETS_NORMALIZED_NAMES,
    COLUMN_DESCRIPTIONS,
    COLUMN_ALIASES
)
from openai import OpenAI, RateLimitError, AuthenticationError
from typing import Dict, List, Optional, Tuple
import re
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from logging_config import setup_logger
from http_interceptor import configure_openai_client
import numpy as np
import logging

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

logger = logging.getLogger(__name__)

class DataNormalizer:
    def __init__(self, input_dir: str = "data/csv", output_dir: str = "data/normalized_csv"):
        """Initialise le DataNormalizer avec les chemins des dossiers."""
        # Obtenir le chemin absolu du répertoire de travail
        base_dir = Path(os.getcwd())
        self.input_dir = base_dir / input_dir
        self.output_dir = base_dir / output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.logger = setup_logger(__name__)
        
        # Configuration OpenAI (optionnelle)
        self.client = None
        self.model = None
        try:
            self.client = configure_openai_client(test_connection=False)
            self.logger.info("Client OpenAI initialisé")
        except Exception as e:
            self.logger.warning(f"Mode dégradé activé - OpenAI non disponible: {str(e)}")
        
    def ensure_model_initialized(self):
        """S'assure que le modèle est initialisé avant utilisation."""
        if self.client and not self.model:
            try:
                self.model = self._detect_best_model()
                self.logger.info(f"Modèle sélectionné: {self.model}")
            except Exception as e:
                self.logger.error(f"Erreur lors de la détection du modèle: {str(e)}")
                raise
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        retry_error_cls=RateLimitError
    )
    def _detect_best_model(self, for_testing: bool = False) -> str:
        """Détecte le meilleur modèle disponible."""
        try:
            if for_testing:
                self.logger.info("Utilisation du modèle gpt-4o-mini pour les tests")
                return "gpt-4o-mini"
                
            self.logger.info("Détection du meilleur modèle OpenAI...")
            models = self.client.models.list()
            available_models = [m.id for m in models.data]
            self.logger.info(f"Modèles disponibles: {available_models}")
            
            # Priorité des modèles
            preferred_models = [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
            
            for model in preferred_models:
                if model in available_models:
                    self.logger.info(f"Modèle sélectionné: {model}")
                    return model
            
            default_model = available_models[0] if available_models else "gpt-3.5-turbo"
            self.logger.info(f"Aucun modèle préféré disponible, utilisation de {default_model}")
            return default_model
        except RateLimitError as e:
            self.logger.warning(f"Rate limit atteint lors de la détection du modèle: {str(e)}")
            raise
        except AuthenticationError as e:
            self.logger.error(f"Erreur d'authentification lors de la détection du modèle: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection du modèle: {str(e)}")
            return "gpt-3.5-turbo"
        
    def _clean_column_name(self, name: str) -> str:
        """Nettoie le nom de colonne pour la comparaison"""
        return re.sub(r'[^a-zA-Z0-9]', '', name.lower())
        
    def _find_matching_feature(self, column_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Trouve la correspondance pour un nom de colonne.
        Retourne (code_colonne, nom_normalisé) ou (None, None) si aucune correspondance.
        """
        cleaned_name = self._clean_column_name(column_name)
        
        # 1. Vérification directe des codes
        for code, normalized_name in {**FEATURES_NORMALIZED_NAMES, **TARGETS_NORMALIZED_NAMES}.items():
            if cleaned_name == self._clean_column_name(code):
                return code, normalized_name
        
        # 2. Vérification des noms normalisés
        for code, normalized_name in {**FEATURES_NORMALIZED_NAMES, **TARGETS_NORMALIZED_NAMES}.items():
            if cleaned_name == self._clean_column_name(normalized_name):
                return code, normalized_name
        
        # 3. Vérification des alias
        for code, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if cleaned_name == self._clean_column_name(alias):
                    return code, {**FEATURES_NORMALIZED_NAMES, **TARGETS_NORMALIZED_NAMES}[code]
        
        # 4. Si OpenAI est disponible, demander une suggestion
        if self.client and self.model:
            try:
                return self._ask_openai_for_match(column_name)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'appel à OpenAI: {str(e)}")
        
        return None, None

    def _ask_openai_for_match(self, column_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Demande à OpenAI de trouver une correspondance pour le nom de colonne"""
        if not self.client:
            return None, None
            
        self.ensure_model_initialized()
        
        prompt = f"""Given the following column name from a building energy dataset: '{column_name}'
        Match it to one of these features based on their descriptions:
        
        Available features and their descriptions:
        {self._format_features_for_prompt()}
        
        Return the matching feature code (X1-X8 or y1-y2) and its normalized name.
        If no match is found, return 'unknown'."""
        
        self.logger.info(f"Envoi de la requête à OpenAI pour la colonne: {column_name}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data expert specializing in building energy efficiency."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        match = response.choices[0].message.content.strip()
        self.logger.info(f"Réponse d'OpenAI pour {column_name}: {match}")
        
        if match in FEATURES_NORMALIZED_NAMES:
            return match, FEATURES_NORMALIZED_NAMES[match]
        elif match in TARGETS_NORMALIZED_NAMES:
            return match, TARGETS_NORMALIZED_NAMES[match]
        return None, None

    def _format_features_for_prompt(self) -> str:
        """Formate les features pour le prompt OpenAI"""
        features_text = []
        for code, desc in COLUMN_DESCRIPTIONS.items():
            normalized_name = FEATURES_NORMALIZED_NAMES.get(code) or TARGETS_NORMALIZED_NAMES.get(code)
            features_text.append(f"- {code} ({normalized_name}): {desc}")
        return "\n".join(features_text)
        
    def normalize_csv(self, input_file: Path) -> bool:
        """Normalise les noms de colonnes d'un fichier CSV"""
        try:
            self.logger.info(f"Début de la normalisation du fichier: {input_file}")
            
            # Lecture du fichier CSV
            df = pd.read_csv(input_file, sep=CSV_CONFIG['separator'], 
                           decimal=CSV_CONFIG['decimal'],
                           encoding=CSV_CONFIG['encoding'])
            
            # Création du mapping des colonnes
            column_mapping = {}
            for column in df.columns:
                code, normalized_name = self._find_matching_feature(column)
                if normalized_name:
                    column_mapping[column] = normalized_name
                    self.logger.info(f"Colonne '{column}' normalisée en '{normalized_name}'")
                else:
                    # Si aucune correspondance n'est trouvée, on garde le nom original nettoyé
                    column_mapping[column] = self._clean_column_name(column)
                    self.logger.warning(f"Aucune correspondance trouvée pour la colonne '{column}'")
            
            # Renommage des colonnes
            df = df.rename(columns=column_mapping)
            
            # Création du nom de fichier normalisé
            output_file = self.output_dir / f"normalized_{input_file.name}"
            
            # Sauvegarde du fichier normalisé
            df.to_csv(output_file, sep=CSV_CONFIG['separator'],
                     decimal=CSV_CONFIG['decimal'],
                     encoding=CSV_CONFIG['encoding'],
                     index=CSV_CONFIG['index'])
            
            self.logger.info(f"Fichier normalisé créé: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la normalisation de {input_file}: {str(e)}")
            return False
            
    def normalize_all_files(self) -> bool:
        """Normalise tous les fichiers CSV du répertoire d'entrée"""
        try:
            self.logger.info("Début de la normalisation des fichiers")
            success = True
            for file in self.input_dir.glob("*.csv"):
                if not self.normalize_csv(file):
                    success = False
            return success
        except Exception as e:
            self.logger.error(f"Erreur lors de la normalisation des fichiers: {str(e)}")
            return False
            
    def get_feature_descriptions(self) -> Dict:
        """Retourne les descriptions des features et targets"""
        return {
            'features': FEATURES_NORMALIZED_NAMES,
            'targets': TARGETS_NORMALIZED_NAMES
        }

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise les données d'un DataFrame."""
        # Conversion des colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Normalisation des valeurs manquantes
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Standardisation des colonnes numériques
        for col in numeric_cols:
            if df[col].std() != 0:  # Évite la division par zéro
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df 

    def list_normalized_files(self) -> list:
        """Liste tous les fichiers CSV normalisés dans le dossier normalized_csv."""
        try:
            self.logger.info("Recherche des fichiers normalisés...")
            if not self.output_dir.exists():
                self.logger.warning(f"Le dossier {self.output_dir} n'existe pas")
                return []
            
            # Récupérer tous les fichiers CSV du dossier
            normalized_files = [f.name for f in self.output_dir.glob("*.csv")]
            self.logger.info(f"Fichiers normalisés trouvés : {normalized_files}")
            return normalized_files
        except Exception as e:
            self.logger.error(f"Erreur lors de la liste des fichiers normalisés : {str(e)}")
            return [] 