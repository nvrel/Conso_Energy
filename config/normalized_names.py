"""
Configuration des noms normalisés pour les colonnes.
Si un nom n'est pas défini ici, OpenAI sera utilisé pour le suggérer.
"""

# Noms normalisés pour les features (X1-X8)
FEATURES_NORMALIZED_NAMES = {
    'X1': 'relative_compactness',
    'X2': 'surface_area',
    'X3': 'wall_area',
    'X4': 'roof_area',
    'X5': 'overall_height',
    'X6': 'orientation',
    'X7': 'glazing_area',
    'X8': 'glazing_distribution'
}

# Noms normalisés pour les targets (y1-y2)
TARGETS_NORMALIZED_NAMES = {
    'y1': 'heating_load',
    'y2': 'cooling_load'
}

# Descriptions des colonnes pour aider OpenAI à faire les correspondances
COLUMN_DESCRIPTIONS = {
    'X1': 'Rapport sans dimension mesurant la compacité de la forme du bâtiment',
    'X2': 'Surface totale de l\'enveloppe du bâtiment en mètres carrés',
    'X3': 'Surface totale des murs extérieurs du bâtiment en mètres carrés',
    'X4': 'Surface du toit du bâtiment en mètres carrés',
    'X5': 'Hauteur globale du bâtiment en mètres',
    'X6': 'Orientation principale du bâtiment par rapport aux points cardinaux',
    'X7': 'Surface des vitrages de l\'enveloppe en mètres carrés ou en pourcentage',
    'X8': 'Répartition des vitrages sur les différentes façades du bâtiment',
    'y1': 'Charge thermique nécessaire pour le chauffage du bâtiment en kWh/m²',
    'y2': 'Charge thermique pour la climatisation en kWh/m²'
}

# Alias possibles pour chaque colonne
COLUMN_ALIASES = {
    'X1': ['relative compactness', 'compacité relative', 'compactness'],
    'X2': ['surface area', 'surface totale', 'total area'],
    'X3': ['wall area', 'surface des murs', 'wall surface'],
    'X4': ['roof area', 'surface du toit', 'roof surface'],
    'X5': ['overall height', 'hauteur totale', 'building height'],
    'X6': ['orientation', 'building orientation', 'direction'],
    'X7': ['glazing area', 'surface vitrée', 'window area'],
    'X8': ['glazing distribution', 'distribution vitrage', 'window distribution'],
    'y1': ['heating load', 'charge chauffage', 'heating energy'],
    'y2': ['cooling load', 'charge climatisation', 'cooling energy']
} 