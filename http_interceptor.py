import logging
import json
from typing import Dict, Any, Optional
import httpx
from openai import OpenAI
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger('http_interceptor')

class HTTPInterceptor:
    def __init__(self):
        self.logger = logger

    def log_request(self, request: httpx.Request) -> None:
        """Log les détails d'une requête HTTP."""
        try:
            # Extraire les informations de la requête
            method = request.method
            url = str(request.url)
            parsed_url = urlparse(url)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            # Extraire les headers (en excluant les informations sensibles)
            headers = dict(request.headers)
            if 'authorization' in headers:
                headers['authorization'] = 'Bearer [REDACTED]'
            if 'openai-organization' in headers:
                headers['openai-organization'] = '[REDACTED]'
            
            # Extraire le corps de la requête
            body = None
            if request.content:
                try:
                    body = json.loads(request.content.decode('utf-8'))
                except:
                    body = request.content.decode('utf-8')

            # Log les informations
            self.logger.info("=" * 80)
            self.logger.info("REQUEST:")
            self.logger.info(f"Method: {method}")
            self.logger.info(f"URL: {url}")
            self.logger.info(f"Path: {path}")
            self.logger.info(f"Query Params: {query_params}")
            self.logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            self.logger.info(f"Body: {json.dumps(body, indent=2) if isinstance(body, dict) else body}")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"Erreur lors du logging de la requête: {str(e)}")

    def log_response(self, response: httpx.Response) -> None:
        """Log les détails d'une réponse HTTP."""
        try:
            # Extraire les informations de la réponse
            status_code = response.status_code
            headers = dict(response.headers)
            
            # Extraire le corps de la réponse
            body = None
            if response.content:
                try:
                    body = json.loads(response.content.decode('utf-8'))
                except:
                    body = response.content.decode('utf-8')

            # Log les informations
            self.logger.info("=" * 80)
            self.logger.info("RESPONSE:")
            self.logger.info(f"Status Code: {status_code}")
            self.logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            self.logger.info(f"Body: {json.dumps(body, indent=2) if isinstance(body, dict) else body}")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"Erreur lors du logging de la réponse: {str(e)}")

    def log_error(self, error: Exception, response: Optional[httpx.Response] = None) -> None:
        """Log les détails d'une erreur HTTP."""
        try:
            error_details = {
                'type': type(error).__name__,
                'message': str(error),
                'response': {
                    'status_code': response.status_code if response else None,
                    'headers': dict(response.headers) if response else None,
                    'body': response.content.decode('utf-8') if response and response.content else None
                } if response else None
            }
            
            self.logger.error("=" * 80)
            self.logger.error("ERROR:")
            self.logger.error(json.dumps(error_details, indent=2))
            self.logger.error("=" * 80)
        except Exception as e:
            self.logger.error(f"Erreur lors du logging de l'erreur: {str(e)}")

class InterceptingTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport):
        self.transport = transport
        self.interceptor = HTTPInterceptor()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.interceptor.log_request(request)
        try:
            response = self.transport.handle_request(request)
            self.interceptor.log_response(response)
            return response
        except Exception as e:
            self.interceptor.log_error(e)
            raise

def configure_http_client() -> httpx.Client:
    """Configure un client HTTP avec l'intercepteur."""
    transport = httpx.HTTPTransport(verify=False)  # Désactive la vérification SSL
    intercepting_transport = InterceptingTransport(transport)
    return httpx.Client(transport=intercepting_transport, verify=False)  # Désactive également la vérification au niveau du client

def configure_openai_client(test_connection: bool = False) -> OpenAI:
    """Configure le client OpenAI avec l'intercepteur."""
    import os
    from openai import AuthenticationError
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise AuthenticationError("OPENAI_API_KEY non trouvée dans les variables d'environnement")
    
    # Récupérer l'organisation ID de manière optionnelle
    org_id = os.getenv('OPENAI_ORG_ID')
    
    # Créer le client avec ou sans organisation ID
    client_config = {
        'api_key': api_key,
        'http_client': configure_http_client()
    }
    
    if org_id:
        client_config['organization'] = org_id
    
    try:
        client = OpenAI(**client_config)
        if test_connection:
            # Tester la connexion seulement si demandé
            client.models.list()
        return client
    except AuthenticationError as e:
        if test_connection and "OpenAI-Organization header should match organization for API key" in str(e):
            logger.warning("L'organisation ID ne correspond pas à la clé API. Tentative sans organisation ID...")
            # Réessayer sans organisation ID
            client_config.pop('organization', None)
            client = OpenAI(**client_config)
            if test_connection:
                client.models.list()  # Tester la connexion
            return client
        raise 