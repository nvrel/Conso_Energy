import pandas as pd
from pathlib import Path
import logging
import shutil
import os
from logging_config import setup_logger

class DataConverter:
    def __init__(self):
        self.input_dir = Path("data")
        self.output_dir = Path("data/csv")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = setup_logger(__name__)

    def convert_to_csv(self) -> bool:
        """Convertit tous les fichiers Excel en CSV"""
        try:
            self.logger.info("Début de la conversion des fichiers Excel en CSV")
            success = True
            
            for file in self.input_dir.glob('*.xls*'):
                try:
                    self.logger.info(f"Conversion du fichier: {file.name}")
                    df = pd.read_excel(file)
                    output_file = self.output_dir / f"{file.stem}.csv"
                    df.to_csv(output_file, sep=';', decimal=',', index=False, encoding='utf-8')
                    self.logger.info(f"Fichier converti avec succès: {output_file.name}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la conversion de {file.name}: {str(e)}")
                    success = False
                    
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur générale lors de la conversion: {str(e)}")
            return False

    def clear_csv_directory(self) -> bool:
        """Supprime tous les fichiers du dossier CSV"""
        try:
            self.logger.info("Suppression des fichiers du dossier CSV")
            for file in self.output_dir.glob('*'):
                if file.is_file():
                    file.unlink()
            self.logger.info("Suppression des fichiers CSV terminée avec succès")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des fichiers CSV: {str(e)}")
            return False

if __name__ == "__main__":
    # Exemple d'utilisation
    converter = DataConverter()
    
    try:
        if converter.convert_to_csv():
            print("Conversion terminée avec succès")
        else:
            print("Erreur lors de la conversion")
    except Exception as e:
        print(f"Erreur: {str(e)}") 