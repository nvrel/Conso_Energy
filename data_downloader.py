import os
import urllib.request
import zipfile
from pathlib import Path
import logging
import shutil
import requests
from logging_config import setup_logger

class DataDownloader:
    def __init__(self, url: str, output_dir: str = "data"):
        """
        Initialise le téléchargeur de données.
        
        Args:
            url (str): URL du fichier à télécharger
            output_dir (str): Répertoire de sortie pour les données
        """
        self.url = url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration du logging
        self.logger = setup_logger(__name__)

    def download_file(self) -> Path:
        """
        Télécharge le fichier depuis l'URL spécifiée.
        
        Returns:
            Path: Chemin vers le fichier téléchargé
        """
        filename = self.url.split('/')[-1]
        output_path = self.output_dir / filename
        
        self.logger.info(f"Téléchargement du fichier depuis {self.url}")
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            self.logger.info(f"Fichier téléchargé avec succès: {filename}")
            return output_path
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement: {str(e)}")
            raise

    def extract_zip(self, zip_path: Path) -> Path:
        """
        Extrait le contenu d'un fichier ZIP et supprime le fichier ZIP.
        
        Args:
            zip_path (Path): Chemin vers le fichier ZIP
            
        Returns:
            Path: Chemin vers le répertoire d'extraction
        """
        self.logger.info(f"Extraction du fichier {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            self.logger.info(f"Fichier extrait avec succès dans {self.output_dir}")
            
            # Suppression du fichier ZIP après extraction
            zip_path.unlink()
            self.logger.info(f"Fichier ZIP {zip_path} supprimé")
            
            return self.output_dir
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction: {str(e)}")
            raise

    def download_and_extract(self) -> Path:
        """
        Télécharge et extrait le fichier en une seule opération.
        
        Returns:
            Path: Chemin vers le répertoire contenant les données extraites
        """
        zip_path = self.download_file()
        return self.extract_zip(zip_path)

    def clear_data_directory(self) -> bool:
        """
        Supprime tous les fichiers du répertoire de données.
        
        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        try:
            self.logger.info("Suppression des fichiers du répertoire de données")
            for item in self.output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            self.logger.info(f"Répertoire {self.output_dir} vidé avec succès")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des fichiers: {str(e)}")
            return False

if __name__ == "__main__":
    # Exemple d'utilisation
    url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    downloader = DataDownloader(url)
    
    try:
        data_dir = downloader.download_and_extract()
        print(f"Données disponibles dans: {data_dir}")
    except Exception as e:
        print(f"Erreur: {str(e)}")

from data_normalizer import DataNormalizer
normalizer = DataNormalizer() 