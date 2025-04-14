import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
import logging
import tarfile
import zipfile
import shutil
from urllib.parse import urljoin
import re
import threading

class OpenEIScraper:
    def __init__(self, base_url="https://en.openei.org/datasets/files/961/pub/", output_dir="data"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._cancel_event = threading.Event()
        self._current_file = None
        
    def cancel(self):
        """Annule le téléchargement en cours"""
        self._cancel_event.set()
        if self._current_file:
            self.logger.info(f"Annulation du téléchargement de {self._current_file}")
            
    def is_cancelled(self):
        """Vérifie si le téléchargement a été annulé"""
        return self._cancel_event.is_set()
        
    def reset_cancel(self):
        """Réinitialise l'état d'annulation"""
        self._cancel_event.clear()
        self._current_file = None
        
    def get_page_content(self):
        """Récupère le contenu de la page OpenEI"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la page: {str(e)}")
            return None
            
    def extract_links(self, html_content):
        """Extrait les liens vers les fichiers compressés"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        # Chercher les liens vers les fichiers .tar.gz et .zip
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and (href.endswith('.tar.gz') or href.endswith('.zip')):
                # Ne pas suivre les liens vers les sous-répertoires
                if '/' not in href:
                    full_url = urljoin(self.base_url, href)
                    links.append(full_url)
                
        return links
        
    def group_multi_part_files(self, links):
        """Groupe les fichiers multi-parties"""
        multi_part_files = {}
        
        # Expression régulière pour détecter les fichiers multi-parties
        pattern = r'(.+)\.part(\d+)\.(tar\.gz|zip)$'
        
        for link in links:
            filename = link.split('/')[-1]
            match = re.match(pattern, filename)
            
            if match:
                base_name = match.group(1)
                part_num = int(match.group(2))
                extension = match.group(3)
                
                if base_name not in multi_part_files:
                    multi_part_files[base_name] = {
                        'extension': extension,
                        'parts': {}
                    }
                    
                multi_part_files[base_name]['parts'][part_num] = link
            else:
                # Fichier simple
                multi_part_files[filename] = {
                    'extension': filename.split('.')[-1],
                    'parts': {1: link}
                }
                
        return multi_part_files
        
    def download_file(self, url):
        """Télécharge un fichier depuis l'URL"""
        try:
            filename = url.split('/')[-1]
            file_path = self.output_dir / filename
            
            # Vérifier si le téléchargement a été annulé
            if self.is_cancelled():
                return None
                
            # Vérifier si le fichier existe déjà
            if file_path.exists():
                self.logger.info(f"Le fichier {filename} existe déjà, passage au suivant")
                return file_path
                
            self.logger.info(f"Téléchargement de {filename}...")
            self._current_file = filename
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.is_cancelled():
                        # Supprimer le fichier partiellement téléchargé
                        if file_path.exists():
                            file_path.unlink()
                        return None
                        
                    if chunk:
                        f.write(chunk)
                        
            self._current_file = None
            return file_path
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
            return None
            
    def assemble_multi_part_file(self, base_name, parts_info):
        """Assemble les parties d'un fichier multi-parties"""
        try:
            # Trier les parties par numéro
            sorted_parts = sorted(parts_info['parts'].items())
            
            # Créer le fichier final
            final_path = self.output_dir / f"{base_name}.{parts_info['extension']}"
            
            with open(final_path, 'wb') as final_file:
                for part_num, part_url in sorted_parts:
                    if self.is_cancelled():
                        if final_path.exists():
                            final_path.unlink()
                        return None
                        
                    part_path = self.download_file(part_url)
                    if not part_path:
                        if self.is_cancelled():
                            if final_path.exists():
                                final_path.unlink()
                            return None
                        raise Exception(f"Erreur lors du téléchargement de la partie {part_num}")
                        
                    with open(part_path, 'rb') as part_file:
                        final_file.write(part_file.read())
                        
                    # Supprimer la partie après l'avoir ajoutée au fichier final
                    part_path.unlink()
                    
            return final_path
        except Exception as e:
            self.logger.error(f"Erreur lors de l'assemblage de {base_name}: {str(e)}")
            return None
            
    def extract_file(self, file_path):
        """Extrait un fichier compressé"""
        try:
            if self.is_cancelled():
                if file_path.exists():
                    file_path.unlink()
                return
                
            if file_path.suffixes == ['.tar', '.gz']:
                self.logger.info(f"Extraction de {file_path.name}...")
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(self.output_dir)
                    
            elif file_path.suffix == '.zip':
                self.logger.info(f"Extraction de {file_path.name}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                    
            # Supprimer le fichier compressé après extraction
            file_path.unlink()
            self.logger.info(f"Fichier {file_path.name} supprimé après extraction")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de {file_path}: {str(e)}")
            
    def process_all_files(self):
        """Traite tous les fichiers disponibles sur la page"""
        try:
            # Réinitialiser l'état d'annulation
            self.reset_cancel()
            
            # Récupérer le contenu de la page
            html_content = self.get_page_content()
            if not html_content:
                return False
                
            # Extraire les liens
            links = self.extract_links(html_content)
            if not links:
                self.logger.warning("Aucun fichier trouvé sur la page")
                return False
                
            # Grouper les fichiers multi-parties
            multi_part_files = self.group_multi_part_files(links)
            
            # Traiter chaque fichier (simple ou multi-parties)
            for base_name, file_info in multi_part_files.items():
                if self.is_cancelled():
                    return False
                    
                if len(file_info['parts']) > 1:
                    # Fichier multi-parties
                    final_path = self.assemble_multi_part_file(base_name, file_info)
                    if final_path:
                        self.extract_file(final_path)
                else:
                    # Fichier simple
                    file_path = self.download_file(file_info['parts'][1])
                    if file_path:
                        self.extract_file(file_path)
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement des fichiers: {str(e)}")
            return False 