from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for
from data_downloader import DataDownloader
from data_converter import DataConverter
from data_normalizer import DataNormalizer
from data_analyzer import DataAnalyzer
from http_interceptor import configure_openai_client
from openai import AuthenticationError
import pandas as pd
import json
import os
from pathlib import Path
import tempfile
import shutil
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from logging_config import setup_logger
from openei_scraper import OpenEIScraper
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import base64
import seaborn as sns
from scipy.stats import fisher_exact, wilcoxon, kruskal

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger = setup_logger(__name__)

# Configuration des dossiers
DATA_DIR = Path("data")
CSV_DIR = Path("data/csv")
NORMALIZED_DIR = Path("data/normalized_csv")

# URL du dataset UCI
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"

# Configuration de l'application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['DATA_FOLDER'] = 'data'
app.config['CSV_FOLDER'] = 'data/csv'
app.config['NORMALIZED_FOLDER'] = 'data/normalized_csv'
app.config['VISUALIZATIONS_FOLDER'] = 'data/visualizations'

# Extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'zip', 'xlsx', 'xls', 'csv'}

# Initialisation des classes
data_downloader = DataDownloader(UCI_DATASET_URL)
data_converter = DataConverter()
data_normalizer = DataNormalizer()
data_analyzer = DataAnalyzer()

# Configuration du client OpenAI (optionnelle)
try:
    client = configure_openai_client()
    logger.info("Client OpenAI configuré avec succès")
except Exception as e:
    logger.warning(f"Impossible de configurer le client OpenAI: {str(e)}")
    client = None

# Dictionnaire pour stocker les timestamps des fichiers
file_timestamps = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("Accès à la page d'accueil")
    return render_template('index.html')

@app.route('/list-files')
def list_files():
    try:
        # Récupérer les fichiers des différents répertoires
        data_files = [f for f in os.listdir(app.config['DATA_FOLDER']) 
                     if f.endswith(('.xlsx', '.xls', '.csv', '.zip'))]
        
        csv_files = [f for f in os.listdir(app.config['CSV_FOLDER']) 
                    if f.endswith('.csv')]
        
        normalized_files = [f for f in os.listdir(app.config['NORMALIZED_FOLDER']) 
                          if f.endswith('.csv')]
        
        return jsonify({
            'data': data_files,
            'csv': csv_files,
            'normalized': normalized_files
        })
    except Exception as e:
        logger.error(f"Erreur lors de la liste des fichiers : {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-openai', methods=['POST'])
def test_openai():
    logger.info("Test de la connexion à OpenAI")
    try:
        if not client:
            return jsonify({'error': 'Client OpenAI non initialisé'}), 500
            
        # Test de la connexion en envoyant une requête simple avec le modèle de test
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Utilisation directe du modèle de test
            messages=[
                {"role": "system", "content": "Test de connexion"},
                {"role": "user", "content": "Répondez simplement 'OK'"}
            ],
            temperature=0.1
        )
        
        return jsonify({
            'success': True,
            'message': 'Connexion à OpenAI établie avec succès',
            'response': response.choices[0].message.content,
            'model': 'gpt-4o-mini'
        })
    except AuthenticationError as e:
        logger.error(f"Erreur d'authentification OpenAI: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Erreur d\'authentification',
            'message': 'Vérifiez votre clé API et l\'ID d\'organisation dans le fichier .env'
        }), 401
    except Exception as e:
        logger.error(f"Erreur lors du test OpenAI: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Tentative d'upload de fichier")
    if 'file' not in request.files:
        logger.error("Aucun fichier dans la requête")
        return jsonify({'error': 'Aucun fichier dans la requête'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("Aucun fichier sélectionné")
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
    if not allowed_file(file.filename):
        logger.error(f"Type de fichier non autorisé: {file.filename}")
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
        
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        result = data_downloader.handle_uploaded_file(temp_path)
        if result['success']:
            logger.info(f"Fichier {file.filename} traité avec succès")
            return jsonify({'message': result['message']})
        else:
            logger.error(f"Erreur lors du traitement du fichier {file.filename}")
            return jsonify({'error': result['message']}), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_file():
    logger.info("Tentative de téléchargement depuis URL")
    data = request.get_json()
    if not data or 'url' not in data:
        logger.error("URL non fournie")
        return jsonify({'error': 'URL non fournie'}), 400
        
    try:
        result = data_downloader.download_and_extract(data['url'])
        if result['success']:
            logger.info(f"Fichier téléchargé avec succès depuis {data['url']}")
            return jsonify({'message': result['message']})
        else:
            logger.error(f"Erreur lors du téléchargement depuis {data['url']}")
            return jsonify({'error': result['message']}), 500
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_files():
    logger.info("Début de la conversion des fichiers")
    try:
        success = data_converter.convert_to_csv()
        if success:
            logger.info("Conversion des fichiers terminée avec succès")
            return jsonify({'message': 'Conversion des fichiers terminée avec succès'})
        else:
            logger.error("Erreur lors de la conversion des fichiers")
            return jsonify({'error': 'Erreur lors de la conversion des fichiers'}), 500
    except Exception as e:
        logger.error(f"Erreur lors de la conversion: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/normalize', methods=['POST'])
def normalize_files():
    logger.info("Début de la normalisation des fichiers")
    try:
        result = data_normalizer.normalize_all_files()
        if result:
            logger.info("Normalisation des fichiers terminée avec succès")
            return jsonify({'message': 'Normalisation des fichiers terminée avec succès'})
        else:
            logger.error("Erreur lors de la normalisation des fichiers")
            return jsonify({'error': 'Erreur lors de la normalisation des fichiers'}), 500
    except Exception as e:
        logger.error(f"Erreur lors de la normalisation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    logger.info("Récupération des descriptions des features")
    try:
        descriptions = data_normalizer.get_feature_descriptions()
        logger.info("Descriptions des features récupérées avec succès")
        return jsonify(descriptions)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des descriptions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-missing-values', methods=['POST'])
def analyze_missing_values():
    """Endpoint pour analyser les valeurs manquantes d'un fichier"""
    logger.info("Début de l'analyse des valeurs manquantes")
    try:
        data = request.get_json()
        if not data or 'file_name' not in data:
            logger.error("Nom de fichier non fourni")
            return jsonify({'error': 'Nom de fichier non fourni'}), 400
            
        file_name = data['file_name']
        result = data_analyzer.analyze_missing_values(file_name)
        logger.info(f"Analyse des valeurs manquantes terminée pour {file_name}")
        return jsonify(result)
        
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {str(e)}")
        return jsonify({'error': f"Fichier non trouvé: {str(e)}"}), 404
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des valeurs manquantes: {str(e)}")
        return jsonify({'error': f"Une erreur est survenue lors de l'analyse des valeurs manquantes: {str(e)}"}), 500

@app.route('/handle-missing-values', methods=['POST'])
def handle_missing_values():
    """Endpoint pour gérer les valeurs manquantes d'un fichier"""
    logger.info("Début du traitement des valeurs manquantes")
    try:
        data = request.get_json()
        required_fields = ['file_name', 'method']
        if not data or not all(field in data for field in required_fields):
            logger.error("Paramètres manquants")
            return jsonify({'error': 'Paramètres requis manquants (file_name, method)'}), 400
            
        file_name = data['file_name']
        method = data['method']
        action = data.get('action', 'global')
        row_index = data.get('row_index')
        custom_value = data.get('custom_value')
        
        result = data_analyzer.handle_missing_values(
            file_name=file_name,
            method=method,
            action=action,
            row_index=row_index,
            custom_value=custom_value
        )
        
        if result:
            logger.info(f"Traitement des valeurs manquantes terminé pour {file_name}")
            return jsonify({'message': 'Traitement des valeurs manquantes terminé avec succès'})
        else:
            logger.error("Échec du traitement des valeurs manquantes")
            return jsonify({'error': 'Échec du traitement des valeurs manquantes'}), 500
            
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {str(e)}")
        return jsonify({'error': f"Fichier non trouvé: {str(e)}"}), 404
    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur lors du traitement des valeurs manquantes: {str(e)}")
        return jsonify({'error': f"Une erreur est survenue lors du traitement des valeurs manquantes: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear_files():
    """Endpoint pour supprimer tous les fichiers"""
    try:
        # Supprimer les fichiers dans le répertoire data
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob('*'):
                if file.is_file() and file.name not in ['csv', 'normalized_csv']:
                    file.unlink()
        
        # Supprimer les fichiers dans le répertoire csv
        csv_dir = data_dir / "csv"
        if csv_dir.exists():
            for file in csv_dir.glob('*.csv'):
                file.unlink()
        
        # Supprimer les fichiers dans le répertoire normalized_csv
        normalized_dir = data_dir / "normalized_csv"
        if normalized_dir.exists():
            for file in normalized_dir.glob('*.csv'):
                file.unlink()
        
        return jsonify({'message': 'Tous les fichiers ont été supprimés avec succès'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-csv/<filename>')
def download_csv(filename):
    """Endpoint pour télécharger un fichier CSV"""
    try:
        csv_path = CSV_DIR / filename
        if not csv_path.exists():
            return jsonify({
                'success': False,
                'message': 'Fichier non trouvé'
            }), 404
            
        return send_file(
            str(csv_path),
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erreur lors du téléchargement: {str(e)}'
        }), 500

@app.route('/scrape-openei', methods=['POST'])
def scrape_openei():
    """Endpoint pour scraper et télécharger les données OpenEI"""
    try:
        scraper = OpenEIScraper(output_dir=DATA_DIR)
        
        if scraper.process_all_files():
            return jsonify({
                'success': True,
                'message': 'Données OpenEI téléchargées et extraites avec succès'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Erreur lors du téléchargement des données OpenEI'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erreur lors du scraping: {str(e)}'
        }), 500

@app.route('/cancel-scrape', methods=['POST'])
def cancel_scrape():
    """Endpoint pour annuler le téléchargement en cours"""
    try:
        scraper.cancel()
        return jsonify({
            'success': True,
            'message': 'Téléchargement annulé avec succès'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erreur lors de l\'annulation: {str(e)}'
        }), 500

@app.route('/download-uci', methods=['POST'])
def download_uci():
    try:
        success = data_downloader.download_and_extract()
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Erreur lors du téléchargement des données UCI'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analysis')
def analysis():
    """Route pour afficher la page d'analyse des données"""
    try:
        # Récupérer la liste des fichiers normalisés
        normalized_files = [f for f in os.listdir(app.config['NORMALIZED_FOLDER']) 
                          if f.endswith('.csv')]
        return render_template('analysis.html', normalized_files=normalized_files)
    except Exception as e:
        logger.error(f"Erreur lors de l'accès à la page d'analyse : {str(e)}")
        return render_template('analysis.html', normalized_files=[])

@app.route('/column-types')
def column_types():
    """Route pour afficher la page de configuration des types de colonnes"""
    try:
        # Utiliser la même logique que /list-files qui fonctionne
        normalized_files = [f for f in os.listdir(app.config['NORMALIZED_FOLDER']) 
                          if f.endswith('.csv')]
        return render_template('column_types.html', normalized_files=normalized_files)
    except Exception as e:
        logger.error(f"Erreur lors de l'accès à la page des types de colonnes : {str(e)}")
        return render_template('column_types.html', normalized_files=[])

@app.route('/get-column-types')
def get_column_types():
    """Route pour récupérer les types de colonnes d'un fichier"""
    file_name = request.args.get('file')
    if not file_name:
        return jsonify({'error': 'Nom de fichier manquant'}), 400

    try:
        # Lire le fichier pour obtenir les colonnes
        file_path = os.path.join('data/normalized_csv', file_name)
        df = pd.read_csv(file_path, sep=';')
        columns = df.columns.tolist()

        # Charger la configuration existante si elle existe
        config_path = os.path.join('config', f'column_types_{file_name}.json')
        column_configs = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                column_configs = json.load(f)

        return jsonify({
            'columns': columns,
            'column_configs': column_configs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-column-types', methods=['POST'])
def save_column_types():
    """Route pour sauvegarder la configuration des types de colonnes"""
    try:
        data = request.json
        file_name = data.get('file')
        column_configs = data.get('column_configs')

        if not file_name or not column_configs:
            return jsonify({'error': 'Données manquantes'}), 400

        # Sauvegarder la configuration dans un fichier JSON
        config_path = os.path.join('config', f'column_types_{file_name}.json')
        with open(config_path, 'w') as f:
            json.dump(column_configs, f, indent=4)

        return jsonify({'message': 'Configuration sauvegardée avec succès'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-distributions', methods=['POST'])
def get_distributions():
    """Route pour obtenir les visualisations des distributions"""
    logger.info("Début de la génération des distributions")
    try:
        data = request.get_json()
        if not data or 'file_name' not in data:
            logger.error("Nom de fichier non fourni")
            return jsonify({'error': 'Nom de fichier non fourni'}), 400
            
        file_name = data['file_name']
        distributions = data_analyzer.generate_distribution_plots(file_name)
        logger.info(f"Distributions générées avec succès pour {file_name}")
        return jsonify({'distributions': distributions})
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération des distributions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-distribution', methods=['POST'])
def analyze_distribution():
    """Analyse la distribution d'une colonne avec GPT-4"""
    try:
        data = request.get_json()
        if not data or 'file_name' not in data or 'column_name' not in data:
            return jsonify({'error': 'file_name et column_name sont requis'}), 400

        file_name = data['file_name']
        column_name = data['column_name']
        
        app.logger.info(f"Analyse de la distribution de {column_name} dans {file_name}")
        
        # Vérifier que le client OpenAI est initialisé
        if not client:
            return jsonify({'error': 'Client OpenAI non initialisé'}), 500
        
        # Lire le fichier avec les bonnes options pour les nombres
        file_path = os.path.join(app.config['NORMALIZED_FOLDER'], file_name)
        df = pd.read_csv(file_path, sep=';')
        
        # Vérifier que la colonne existe
        if column_name not in df.columns:
            return jsonify({'error': f'Colonne {column_name} non trouvée'}), 404
        
        # Nettoyer et convertir la colonne en numérique
        try:
            # Essayer d'abord avec la conversion directe
            df[column_name] = pd.to_numeric(df[column_name].str.replace(',', '.'), errors='coerce')
        except:
            try:
                # Si ça échoue, essayer de nettoyer la chaîne
                df[column_name] = df[column_name].astype(str).str.replace(r'[^\d.,]', '')
                df[column_name] = pd.to_numeric(df[column_name].str.replace(',', '.'), errors='coerce')
            except:
                return jsonify({'error': f'Impossible de convertir la colonne {column_name} en valeurs numériques'}), 400
        
        # Vérifier si nous avons des valeurs valides
        if df[column_name].isna().all():
            return jsonify({'error': f'La colonne {column_name} ne contient pas de valeurs numériques valides'}), 400
        
        # Préparer le prompt pour GPT-4
        prompt = f"""
        En tant qu'expert en data science, analysez la distribution de la variable {column_name} pour un modèle linéaire.

        Statistiques :
        - Moyenne : {df[column_name].mean():.2f}
        - Médiane : {df[column_name].median():.2f}
        - Écart-type : {df[column_name].std():.2f}
        - Skewness : {df[column_name].skew():.2f}
        - Kurtosis : {df[column_name].kurtosis():.2f}

        Recommandez les transformations mathématiques (log, racine carrée, carré, cube, etc.) qui pourraient améliorer la linéarité de cette variable.
        Pour chaque transformation recommandée, donnez un score de pertinence de 1 à 10.
        Répondez uniquement avec la liste des transformations recommandées, sans explications supplémentaires.
        Format de réponse attendu :
        - Transformation1 (score/10)
        - Transformation2 (score/10)
        etc.
        """
        
        # Envoyer la requête à GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un expert en statistiques et en modélisation linéaire."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return jsonify({
            'suggestions': response.choices[0].message.content
        })
        
    except Exception as e:
        app.logger.error(f"Erreur lors de l'analyse de la distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-columns')
def get_columns():
    file_name = request.args.get('file_name')
    if not file_name:
        return jsonify({'error': 'Nom de fichier manquant'}), 400

    try:
        file_path = os.path.join(app.config['DATA_FOLDER'], file_name)
        df = pd.read_csv(file_path)
        return jsonify({'columns': df.columns.tolist()})
    except Exception as e:
        app.logger.error(f"Erreur lors de la lecture des colonnes: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.get_json()
        if not data or 'file_name' not in data:
            return jsonify({'error': 'Nom de fichier manquant'}), 400

        file_name = data['file_name']
        file_path = os.path.join(app.config['DATA_FOLDER'], file_name)
        
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            return jsonify({'error': 'Fichier non trouvé'}), 404

        # Vérifier si l'image existe déjà et si le fichier n'a pas été modifié
        image_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f"{file_name}.png")
        current_timestamp = os.path.getmtime(file_path)
        
        if os.path.exists(image_path) and file_name in file_timestamps and file_timestamps[file_name] == current_timestamp:
            # Retourner l'image existante
            return send_file(image_path, mimetype='image/png')

        # Générer la nouvelle visualisation
        df = pd.read_csv(file_path)
        fig = data_analyzer.visualize_data(df)
        
        # Sauvegarder l'image
        fig.savefig(image_path)
        
        # Mettre à jour le timestamp
        file_timestamps[file_name] = current_timestamp
        
        return send_file(image_path, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-visualization/<file_name>')
def download_visualization(file_name):
    try:
        # Vérifier que le fichier existe
        file_path = os.path.join(app.config['NORMALIZED_FOLDER'], file_name)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Fichier non trouvé'}), 404

        # Générer les visualisations
        distributions = data_analyzer.generate_distribution_plots(file_name)
        if not distributions:
            return jsonify({'error': 'Aucune visualisation générée'}), 500
        
        # Créer un fichier HTML temporaire avec les visualisations
        html_content = '<html><body style="background-color: white;">'
        for column, plots in distributions.items():
            html_content += f'<h2>{column}</h2><div style="display: flex; flex-wrap: wrap;">'
            for plot in plots:
                html_content += f'<img src="data:image/png;base64,{plot}" style="margin: 10px; max-width: 300px;">'
            html_content += '</div>'
        html_content += '</body></html>'
        
        # Sauvegarder le HTML dans un buffer
        buf = io.BytesIO()
        buf.write(html_content.encode('utf-8'))
        buf.seek(0)
        
        # Retourner le fichier
        return send_file(
            buf,
            mimetype='text/html',
            as_attachment=True,
            download_name=f'{file_name}_visualization.html'
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de la visualisation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-correlation-matrix', methods=['POST'])
def get_correlation_matrix():
    """Calcule et retourne la matrice de corrélation"""
    try:
        logger.info("Début du calcul de la matrice de corrélation")
        data = request.get_json()
        if not data or 'file_name' not in data:
            logger.error("Nom de fichier non fourni")
            return jsonify({'error': 'file_name est requis'}), 400

        file_name = data['file_name']
        logger.info(f"Traitement du fichier: {file_name}")
        file_path = os.path.join(app.config['NORMALIZED_FOLDER'], file_name)
        
        # Lire le fichier avec les bons paramètres
        logger.info("Lecture du fichier CSV")
        df = pd.read_csv(file_path, sep=';', decimal=',')
        logger.info(f"Colonnes trouvées: {df.columns.tolist()}")
        
        # Charger la configuration des types de colonnes
        config_path = os.path.join('config', f'column_types_{file_name}.json')
        column_types = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                column_types = json.load(f)
        
        # Sélectionner uniquement les colonnes marquées comme numériques dans la configuration
        numeric_cols = []
        for col in df.columns:
            col_type = column_types.get(col, {}).get('type', 'numeric')
            if col_type == 'numeric':
                numeric_cols.append(col)
                logger.info(f"Colonne {col} ajoutée aux colonnes numériques")
                
        logger.info(f"Colonnes numériques sélectionnées: {numeric_cols}")
        if len(numeric_cols) == 0:
            logger.error("Aucune colonne numérique trouvée")
            return jsonify({'error': 'Aucune colonne numérique trouvée dans le fichier'}), 400
            
        df_numeric = df[numeric_cols]
        
        # Calculer la matrice de corrélation
        logger.info("Calcul de la matrice de corrélation")
        corr_matrix = df_numeric.corr()
        
        # Convertir la matrice en format HTML avec des couleurs
        logger.info("Génération du graphique")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Matrice de Corrélation (Colonnes Numériques)')
        plt.tight_layout()
        
        # Sauvegarder le graphique dans un buffer
        logger.info("Sauvegarde du graphique")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convertir en base64
        logger.info("Conversion en base64")
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        logger.info("Calcul de la matrice de corrélation terminé avec succès")
        return jsonify({
            'correlation_matrix': image_base64,
            'columns': numeric_cols
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la matrice de corrélation: {str(e)}")
        logger.error(f"Type d'erreur: {type(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-fisher-tests', methods=['POST'])
def get_fisher_tests():
    """Effectue le test de Fisher sur les variables catégorielles"""
    try:
        logger.info("Début du calcul des tests de Fisher")
        data = request.get_json()
        if not data or 'file_name' not in data:
            logger.error("Nom de fichier non fourni")
            return jsonify({'error': 'file_name est requis'}), 400

        file_name = data['file_name']
        logger.info(f"Traitement du fichier: {file_name}")
        file_path = os.path.join(app.config['NORMALIZED_FOLDER'], file_name)
        
        # Lire le fichier
        logger.info("Lecture du fichier CSV")
        df = pd.read_csv(file_path, sep=';', decimal=',')
        
        # Charger la configuration des types de colonnes
        config_path = os.path.join('config', f'column_types_{file_name}.json')
        column_types = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                column_types = json.load(f)
        
        # Sélectionner uniquement les colonnes marquées comme catégorielles dans la configuration
        categorical_cols = []
        for col in df.columns:
            col_type = column_types.get(col, {}).get('type', 'numeric')
            if col_type == 'categorical':
                categorical_cols.append(col)
                logger.info(f"Colonne {col} ajoutée aux colonnes catégorielles")
                
        logger.info(f"Colonnes catégorielles sélectionnées: {categorical_cols}")
        if len(categorical_cols) < 2:
            logger.error("Moins de 2 colonnes catégorielles trouvées")
            return jsonify({'error': 'Il faut au moins 2 variables catégorielles pour effectuer le test de Fisher'}), 400
            
        # Créer un tableau de résultats
        results = []
        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                col1 = categorical_cols[i]
                col2 = categorical_cols[j]
                
                # Créer la table de contingence
                contingency_table = pd.crosstab(df[col1], df[col2])
                
                # Effectuer le test de Fisher
                odds_ratio, p_value = fisher_exact(contingency_table)
                
                results.append({
                    'variable1': col1,
                    'variable2': col2,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value
                })
                
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des tests de Fisher: {str(e)}")
        logger.error(f"Type d'erreur: {type(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        return jsonify({'error': str(e)}), 500

@app.route('/select-features', methods=['POST'])
def select_features():
    """Sélectionne les features en utilisant la matrice de corrélation et les tests de Fisher"""
    try:
        logger.info("Début de la sélection de features")
        data = request.get_json()
        if not data or 'file_name' not in data:
            logger.error("Nom de fichier non fourni")
            return jsonify({'error': 'file_name est requis'}), 400

        file_name = data['file_name']
        logger.info(f"Traitement du fichier: {file_name}")
        file_path = os.path.join(app.config['NORMALIZED_FOLDER'], file_name)
        
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            return jsonify({'error': f'Fichier {file_name} non trouvé'}), 404
        
        # Lire le fichier avec les bons paramètres
        logger.info("Lecture du fichier CSV")
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',')
            logger.info(f"Colonnes trouvées: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
            return jsonify({'error': f'Erreur lors de la lecture du fichier: {str(e)}'}), 500
        
        # Charger la configuration des types de colonnes
        config_path = os.path.join('config', f'column_types_{file_name}.json')
        column_types = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    column_types = json.load(f)
                    logger.info(f"Configuration des types de colonnes chargée: {column_types}")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier de configuration: {str(e)}")
                return jsonify({'error': f'Erreur lors de la lecture de la configuration: {str(e)}'}), 500
        else:
            logger.warning(f"Fichier de configuration {config_path} non trouvé")
        
        # Identifier les labels (heating_load et cooling_load)
        labels = ['heating_load', 'cooling_load']
        logger.info(f"Labels identifiés: {labels}")
        
        # Vérifier que les labels existent dans le DataFrame
        missing_labels = [label for label in labels if label not in df.columns]
        if missing_labels:
            logger.error(f"Labels manquants dans le fichier: {missing_labels}")
            return jsonify({'error': f'Labels manquants: {", ".join(missing_labels)}'}), 400
        
        # Séparer les features numériques et catégorielles
        numeric_features = []
        categorical_features = []
        for col in df.columns:
            if col not in labels:
                col_type = column_types.get(col, {}).get('type', 'numeric')
                if col_type == 'numeric':
                    numeric_features.append(col)
                elif col_type == 'categorical':
                    categorical_features.append(col)
        logger.info(f"Features numériques: {numeric_features}")
        logger.info(f"Features catégorielles: {categorical_features}")
        
        # Vérifier qu'il y a des features à traiter
        if not numeric_features and not categorical_features:
            logger.error("Aucune feature trouvée dans le fichier")
            return jsonify({'error': 'Aucune feature trouvée dans le fichier'}), 400
        
        # 1. Sélection basée sur la corrélation
        if numeric_features:
            logger.info("Calcul de la matrice de corrélation")
            try:
                df_numeric = df[numeric_features]
                corr_matrix = df_numeric.corr()
                logger.info("Matrice de corrélation calculée avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la matrice de corrélation: {str(e)}")
                return jsonify({'error': f'Erreur lors du calcul de la matrice de corrélation: {str(e)}'}), 500
            
            # Calculer la somme des corrélations absolues avec les labels pour chaque feature
            logger.info("Calcul des sommes de corrélations")
            feature_corr_sums = {}
            for feature in numeric_features:
                try:
                    corr_sum = sum(abs(df[feature].corr(df[label])) for label in labels)
                    feature_corr_sums[feature] = corr_sum
                except Exception as e:
                    logger.error(f"Erreur lors du calcul de la corrélation pour {feature}: {str(e)}")
                    continue
            logger.info(f"Sommes de corrélations: {feature_corr_sums}")
            
            # Trier les features par somme de corrélation décroissante
            sorted_features = sorted(feature_corr_sums.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"Features triées: {sorted_features}")
            
            # Sélectionner les features en évitant les corrélations élevées
            selected_numeric = []
            remaining_features = [feature for feature, _ in sorted_features]
            
            while remaining_features:
                current_feature = remaining_features[0]
                selected_numeric.append(current_feature)
                remaining_features = [
                    feature for feature in remaining_features[1:]
                    if abs(corr_matrix.loc[current_feature, feature]) <= 0.85
                ]
            logger.info(f"Features numériques sélectionnées: {selected_numeric}")
        else:
            selected_numeric = []
            feature_corr_sums = {}
        
        # 2. Sélection basée sur les tests de Fisher
        if categorical_features:
            logger.info("Sélection des features catégorielles")
            selected_categorical = []
            for i in range(len(categorical_features)):
                feature1 = categorical_features[i]
                keep_feature = True
                
                for selected in selected_categorical:
                    try:
                        contingency_table = pd.crosstab(df[feature1], df[selected])
                        _, p_value = fisher_exact(contingency_table)
                        if p_value < 0.05:
                            keep_feature = False
                            break
                    except Exception as e:
                        logger.error(f"Erreur lors du test de Fisher entre {feature1} et {selected}: {str(e)}")
                        continue
                
                if keep_feature:
                    selected_categorical.append(feature1)
            logger.info(f"Features catégorielles sélectionnées: {selected_categorical}")
        else:
            selected_categorical = []
        
        # 3. Dernière passe avec les tests statistiques
        logger.info("Début des tests statistiques")
        final_selected_numeric = selected_numeric.copy()
        test_results = []
        
        # Liste des variables catégorielles à éliminer
        categorical_to_remove = set()
        
        for numeric_feature in selected_numeric:
            logger.info(f"Traitement de la feature numérique: {numeric_feature}")
            feature_results = {
                'feature': numeric_feature,
                'tests': [],
                'kept': True
            }
            
            for categorical_feature in selected_categorical:
                logger.info(f"Test avec la feature catégorielle: {categorical_feature}")
                try:
                    # Convertir la colonne numérique en float
                    try:
                        # Essayer d'abord la conversion directe
                        df[numeric_feature] = pd.to_numeric(df[numeric_feature], errors='coerce')
                    except:
                        try:
                            # Si ça échoue, essayer de convertir les virgules en points
                            df[numeric_feature] = pd.to_numeric(df[numeric_feature].astype(str).str.replace(',', '.'), errors='coerce')
                        except Exception as e:
                            logger.error(f"Erreur lors de la conversion de {numeric_feature}: {str(e)}")
                            test_result = {
                                'categorical_feature': categorical_feature,
                                'test_type': 'N/A',
                                'error': f'Erreur de conversion: {str(e)}'
                            }
                            feature_results['tests'].append(test_result)
                            continue
                    
                    # Grouper par la feature catégorielle
                    groups = df.groupby(categorical_feature)[numeric_feature]
                    logger.info(f"Nombre de groupes: {len(groups)}")
                    
                    if len(groups) == 2:
                        try:
                            group1, group2 = groups
                            # Convertir les groupes en listes de valeurs numériques
                            values1 = group1[1].dropna().tolist()
                            values2 = group2[1].dropna().tolist()
                            logger.info(f"Taille des groupes: {len(values1)}, {len(values2)}")
                            
                            if values1 and values2:  # Vérifier que les groupes ne sont pas vides
                                _, p_value = wilcoxon(values1, values2)
                                test_result = {
                                    'categorical_feature': categorical_feature,
                                    'test_type': 'Wilcoxon',
                                    'p_value': float(p_value),  # Convertir en float
                                    'significant': bool(p_value < 0.05),  # Convertir en bool
                                    'group1_size': len(values1),
                                    'group2_size': len(values2),
                                    'group1_mean': float(sum(values1)/len(values1)),  # Convertir en float
                                    'group2_mean': float(sum(values2)/len(values2))  # Convertir en float
                                }
                                logger.info(f"Test de Wilcoxon réussi: {test_result}")
                                
                                # Si le test est significatif, marquer la variable catégorielle pour suppression
                                if p_value < 0.05:
                                    categorical_to_remove.add(categorical_feature)
                            else:
                                test_result = {
                                    'categorical_feature': categorical_feature,
                                    'test_type': 'Wilcoxon',
                                    'error': 'Groupes vides'
                                }
                                logger.warning(test_result['error'])
                        except Exception as e:
                            logger.error(f"Erreur lors du test de Wilcoxon: {str(e)}")
                            test_result = {
                                'categorical_feature': categorical_feature,
                                'test_type': 'Wilcoxon',
                                'error': str(e)
                            }
                    elif len(groups) > 2:
                        try:
                            # Convertir les groupes en listes de valeurs numériques
                            group_values = []
                            for group in groups:
                                values = group[1].dropna().tolist()
                                if values:  # Ne pas ajouter de groupes vides
                                    group_values.append(values)
                            logger.info(f"Nombre de groupes non vides: {len(group_values)}")
                            
                            if len(group_values) >= 2:  # Vérifier qu'on a au moins 2 groupes non vides
                                _, p_value = kruskal(*group_values)
                                test_result = {
                                    'categorical_feature': categorical_feature,
                                    'test_type': 'Kruskal-Wallis',
                                    'p_value': float(p_value),  # Convertir en float
                                    'significant': bool(p_value < 0.05),  # Convertir en bool
                                    'group_sizes': [len(values) for values in group_values],
                                    'group_means': [float(sum(values)/len(values)) for values in group_values]  # Convertir en float
                                }
                                logger.info(f"Test de Kruskal-Wallis réussi: {test_result}")
                                
                                # Si le test est significatif, marquer la variable catégorielle pour suppression
                                if p_value < 0.05:
                                    categorical_to_remove.add(categorical_feature)
                            else:
                                test_result = {
                                    'categorical_feature': categorical_feature,
                                    'test_type': 'Kruskal-Wallis',
                                    'error': 'Pas assez de groupes non vides pour effectuer le test'
                                }
                                logger.warning(test_result['error'])
                        except Exception as e:
                            logger.error(f"Erreur lors du test de Kruskal-Wallis: {str(e)}")
                            test_result = {
                                'categorical_feature': categorical_feature,
                                'test_type': 'Kruskal-Wallis',
                                'error': str(e)
                            }
                    else:
                        test_result = {
                            'categorical_feature': categorical_feature,
                            'test_type': 'N/A',
                            'error': f'La variable catégorielle a {len(groups)} catégorie(s), il faut au moins 2 catégories pour effectuer un test'
                        }
                        logger.warning(test_result['error'])
                except Exception as e:
                    logger.error(f"Erreur lors du groupement des données: {str(e)}")
                    test_result = {
                        'categorical_feature': categorical_feature,
                        'test_type': 'N/A',
                        'error': str(e)
                    }
                
                feature_results['tests'].append(test_result)
            
            test_results.append(feature_results)
        
        # Éliminer les variables catégorielles qui ont un impact significatif
        final_selected_categorical = [cat for cat in selected_categorical if cat not in categorical_to_remove]
        logger.info(f"Variables catégorielles éliminées: {list(categorical_to_remove)}")
        logger.info(f"Variables catégorielles finales: {final_selected_categorical}")
        
        # Combiner les features sélectionnées
        selected_features = final_selected_numeric + final_selected_categorical
        logger.info(f"Features finales sélectionnées: {selected_features}")
        
        # Préparer les résultats
        results = {
            'selected_features': selected_features,
            'numeric_features': {
                'all': numeric_features,
                'selected': final_selected_numeric,
                'correlation_sums': feature_corr_sums
            },
            'categorical_features': {
                'all': categorical_features,
                'selected': final_selected_categorical,
                'removed': list(categorical_to_remove)
            },
            'test_results': test_results
        }
        
        logger.info("Sélection de features terminée avec succès")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Erreur lors de la sélection des features: {str(e)}")
        logger.error(f"Type d'erreur: {type(e)}")
        logger.error(f"Traceback: {e.__traceback__}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Démarrage de l'application Flask")
    try:
        # Créer les répertoires s'ils n'existent pas
        logger.info("Création des répertoires si nécessaire")
        DATA_DIR.mkdir(exist_ok=True)
        CSV_DIR.mkdir(exist_ok=True)
        NORMALIZED_DIR.mkdir(exist_ok=True)
        
        # Vérifier l'initialisation des composants
        logger.info("Vérification des composants")
        logger.info(f"DataDownloader initialisé: {data_downloader is not None}")
        logger.info(f"DataConverter initialisé: {data_converter is not None}")
        logger.info(f"DataNormalizer initialisé: {data_normalizer is not None}")
        logger.info(f"DataAnalyzer initialisé: {data_analyzer is not None}")
        logger.info(f"Client OpenAI initialisé: {client is not None}")
        
        # Lancer l'application avec une configuration plus permissive
        logger.info("Démarrage du serveur Flask")
        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            threaded=True,
            use_reloader=False  # Désactiver le reloader pour éviter les problèmes de double démarrage
        )
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application: {str(e)}")
        raise 