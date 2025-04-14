import logging
import logging.handlers
from pathlib import Path
import os

def setup_logger(name: str) -> logging.Logger:
    """Configure un logger avec rotation des fichiers"""
    # Création du dossier logs s'il n'existe pas
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuration du logger principal
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Niveau le plus bas pour capturer tous les logs
    
    # Configuration du format des logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour le fichier avec rotation
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / 'app.log',
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Ajout des handlers au logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Configuration du logging pour les bibliothèques HTTP
    http_logger = logging.getLogger('httpcore')
    http_logger.setLevel(logging.DEBUG)
    http_logger.addHandler(file_handler)
    http_logger.addHandler(console_handler)
    
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.DEBUG)
    httpx_logger.addHandler(file_handler)
    httpx_logger.addHandler(console_handler)
    
    openai_logger = logging.getLogger('openai')
    openai_logger.setLevel(logging.DEBUG)
    openai_logger.addHandler(file_handler)
    openai_logger.addHandler(console_handler)
    
    # Configuration du logging pour l'intercepteur HTTP
    interceptor_logger = logging.getLogger('http_interceptor')
    interceptor_logger.setLevel(logging.DEBUG)
    interceptor_logger.addHandler(file_handler)
    interceptor_logger.addHandler(console_handler)
    
    return logger 