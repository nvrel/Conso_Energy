import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import numpy as np

def visualiser_batiment_3D(variables):
    # Récupération des variables prioritaires
    f_glazing_area = variables.get('f_glazing_area', 20)
    f_orientation = variables.get('f_orientation', 0)
    f_overall_height = variables.get('f_overall_height', 10)
    f_relative_compactness = variables.get('f_relative_compactness', 0.8)
    f_wall_area = variables.get('f_wall_area', 200)

    # Surface totale du bâtiment avec aléa ±30%
    surface = int(f_wall_area * f_relative_compactness * random.uniform(0.7, 1.3))
    # Ratio largeur/profondeur aléatoire (0.5 = profond, 2.0 = large)
    ratio = random.uniform(0.5, 2.0)
    largeur = int((surface * ratio) ** 0.5)
    profondeur = max(1, int(surface / max(largeur, 1)))
    hauteur = max(1, int(f_overall_height * random.uniform(0.9, 1.1)))

    # Forme variable : si compacité faible, bâtiment plus allongé
    if f_relative_compactness < 0.7:
        largeur = int(largeur * random.uniform(1.4, 1.6))
        profondeur = int(profondeur * random.uniform(0.6, 0.8))

    # --- Premier cube (principal, avec rotation aléatoire) ---
    x = [0, largeur, largeur, 0, 0, largeur, largeur, 0]
    y = [0, 0, profondeur, profondeur, 0, 0, profondeur, profondeur]
    z = [0, 0, 0, 0, hauteur, hauteur, hauteur, hauteur]
    angle = random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_rot = [cos_a * xi - sin_a * yi for xi, yi in zip(x, y)]
    y_rot = [sin_a * xi + cos_a * yi for xi, yi in zip(x, y)]
    verts1 = [
        [ [x_rot[0],y_rot[0],z[0]], [x_rot[1],y_rot[1],z[1]], [x_rot[2],y_rot[2],z[2]], [x_rot[3],y_rot[3],z[3]] ], # bas
        [ [x_rot[4],y_rot[4],z[4]], [x_rot[5],y_rot[5],z[5]], [x_rot[6],y_rot[6],z[6]], [x_rot[7],y_rot[7],z[7]] ], # haut
        [ [x_rot[0],y_rot[0],z[0]], [x_rot[1],y_rot[1],z[1]], [x_rot[5],y_rot[5],z[5]], [x_rot[4],y_rot[4],z[4]] ], # côté 1 (avant)
        [ [x_rot[2],y_rot[2],z[2]], [x_rot[3],y_rot[3],z[3]], [x_rot[7],y_rot[7],z[7]], [x_rot[6],y_rot[6],z[6]] ], # côté 2 (arrière)
        [ [x_rot[1],y_rot[1],z[1]], [x_rot[2],y_rot[2],z[2]], [x_rot[6],y_rot[6],z[6]], [x_rot[5],y_rot[5],z[5]] ], # côté 3 (droite)
        [ [x_rot[4],y_rot[4],z[4]], [x_rot[7],y_rot[7],z[7]], [x_rot[3],y_rot[3],z[3]], [x_rot[0],y_rot[0],z[0]] ], # côté 4 (gauche)
    ]
    color1 = 'lightgreen'  # Couleur fixe pour le cube

    # --- Pyramide sur le toit du premier cube ---
    # Base = sommet haut du premier cube
    base = [
        [x_rot[4], y_rot[4], z[4]],
        [x_rot[5], y_rot[5], z[5]],
        [x_rot[6], y_rot[6], z[6]],
        [x_rot[7], y_rot[7], z[7]]
    ]
    # Sommet de la pyramide (centré sur la base)
    base_cx = sum([p[0] for p in base]) / 4
    base_cy = sum([p[1] for p in base]) / 4
    base_cz = z[4]
    pyr_height = hauteur * random.uniform(0.3, 0.7)
    apex = [base_cx, base_cy, base_cz + pyr_height]
    # Faces de la pyramide
    verts_pyr = [
        [base[0], base[1], apex],
        [base[1], base[2], apex],
        [base[2], base[3], apex],
        [base[3], base[0], apex],
        base  # base carrée
    ]
    color_pyr = 'gold'  # Couleur fixe pour la pyramide

    # Création de la figure 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Bâtiment 3D complexe (cube et pyramide)")

    # Premier cube
    ax.add_collection3d(Poly3DCollection(verts1, facecolors=color1, linewidths=1, edgecolors='black', alpha=0.7))
    # Pyramide
    ax.add_collection3d(Poly3DCollection(verts_pyr, facecolors=color_pyr, linewidths=1, edgecolors='saddlebrown', alpha=0.8))

    # Ajout de la surface vitrée sur la face avant du premier cube
    facade_largeur = np.linalg.norm([x_rot[1] - x_rot[0], y_rot[1] - y_rot[0]])
    facade_hauteur = hauteur
    surface_facade = facade_largeur * facade_hauteur
    if f_glazing_area > 0 and surface_facade > 0:
        glazing_height = min(facade_hauteur, f_glazing_area / max(facade_largeur, 1))
        glazing_width = min(facade_largeur, f_glazing_area / max(glazing_height, 1))
        x0, y0 = x_rot[0], y_rot[0]
        x1, y1 = x_rot[1], y_rot[1]
        dx = (x1 - x0) / facade_largeur
        dy = (y1 - y0) / facade_largeur
        base_x = x0 + (facade_largeur - glazing_width) / 2 * dx
        base_y = y0 + (facade_largeur - glazing_width) / 2 * dy
        glazing_verts = [
            [base_x, base_y, 0],
            [base_x + glazing_width * dx, base_y + glazing_width * dy, 0],
            [base_x + glazing_width * dx, base_y + glazing_width * dy, glazing_height],
            [base_x, base_y, glazing_height]
        ]
        ax.add_collection3d(Poly3DCollection([glazing_verts], facecolors='deepskyblue', linewidths=1, edgecolors='navy', alpha=0.8))

    # Affichage des axes
    ax.set_xlabel('Largeur (m)')
    ax.set_ylabel('Profondeur (m)')
    ax.set_zlabel('Hauteur (m)')
    ax.set_xlim(min(x_rot), max(x_rot))
    ax.set_ylim(min(y_rot), max(y_rot))
    ax.set_zlim(0, hauteur + pyr_height)

    # Affichage des paramètres dans la légende
    orientation_str = ['Nord', 'Est', 'Sud', 'Ouest'][f_orientation] if f_orientation in [0,1,2,3] else str(f_orientation)
    plt.figtext(0.02, 0.02,
        f"Surface vitrée: {f_glazing_area} m²\n"
        f"Orientation: {orientation_str}\n"
        f"Hauteur: {hauteur} m\n"
        f"Compacité: {f_relative_compactness}\n"
        f"Surface des murs: {f_wall_area} m²\n"
        f"Couleurs: {color1}, {color_pyr}\n"
        f"Angle de rotation: {int(np.degrees(angle))}°\n"
        f"Surface totale: {surface} m²",
        fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7)
    )

    plt.show()

if __name__ == "__main__":
    variables = {
        'f_glazing_area': 30,
        'f_orientation': 2,
        'f_overall_height': 12,
        'f_relative_compactness': 0.85,
        'f_wall_area': 220
    }
    visualiser_batiment_3D(variables) 