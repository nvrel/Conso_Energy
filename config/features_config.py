from typing import Dict, List, Optional, Literal

# Types de données possibles
DataType = Literal['numeric', 'categorical_ordered', 'categorical_unordered', 'date']

class FeatureConfig:
    def __init__(
        self,
        normalized_name: str,
        description: str,
        unit: str,
        data_type: DataType,
        categories: Optional[List[str]] = None
    ):
        self.normalized_name = normalized_name
        self.description = description
        self.unit = unit
        self.data_type = data_type
        self.categories = categories

# Configuration des caractéristiques
FEATURES_CONFIG: Dict[str, FeatureConfig] = {
    'X1': FeatureConfig(
        normalized_name='relative_compactness',
        description='Compacité relative',
        unit='-',
        data_type='numeric'
    ),
    'X2': FeatureConfig(
        normalized_name='surface_area',
        description='Surface',
        unit='m²',
        data_type='numeric'
    ),
    'X3': FeatureConfig(
        normalized_name='wall_area',
        description='Surface des murs',
        unit='m²',
        data_type='numeric'
    ),
    'X4': FeatureConfig(
        normalized_name='roof_area',
        description='Surface du toit',
        unit='m²',
        data_type='numeric'
    ),
    'X5': FeatureConfig(
        normalized_name='overall_height',
        description='Hauteur totale',
        unit='m',
        data_type='numeric'
    ),
    'X6': FeatureConfig(
        normalized_name='orientation',
        description='Orientation',
        unit='degrés',
        data_type='categorical_ordered',
        categories=['Nord', 'Est', 'Sud', 'Ouest']
    ),
    'X7': FeatureConfig(
        normalized_name='glazing_area',
        description='Surface vitrée',
        unit='m²',
        data_type='numeric'
    ),
    'X8': FeatureConfig(
        normalized_name='glazing_area_distribution',
        description='Distribution de la surface vitrée',
        unit='-',
        data_type='categorical_ordered',
        categories=['Uniforme', 'Nord', 'Est', 'Sud', 'Ouest']
    )
}

# Configuration des cibles
TARGETS_CONFIG: Dict[str, FeatureConfig] = {
    'y1': FeatureConfig(
        normalized_name='heating_load',
        description='Charge de chauffage',
        unit='kWh/m²',
        data_type='numeric'
    ),
    'y2': FeatureConfig(
        normalized_name='cooling_load',
        description='Charge de refroidissement',
        unit='kWh/m²',
        data_type='numeric'
    )
}

# Configuration pour le format CSV
CSV_CONFIG = {
    'separator': ';',
    'decimal': ',',
    'encoding': 'utf-8',
    'index': False
} 