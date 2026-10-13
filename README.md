# Vision 3D : notes et implémentations

## Contenu
- `camera_geometry.py` : modèle sténopé, projection, back-projection, intrinsèques, extrinsèques.
- `lss_toy.py` : implémentation jouet de Lift-Splat-Shoot (LSS) sur une caméra et une grille BEV 4×4.
- `tests.py` : tests unitaires pour la géométrie caméra.
- `notes/` : synthèses de lecture (3DGS, GaussianBEV, Reconstruction Matters).

## Géométrie caméra
Ce que c'est : projection d'un point 3D monde vers un pixel image.
Formules implémentées : `Xc = R Xw + t`, `u = f Xc/Zc + cx`, `v = f Yc/Zc + cy`.
Vérifié par : aller-retour world ↔ camera, back-projection, perte de profondeur, parallaxe.

## LSS jouet
Ce que c'est : reproduction minimale de la mécanique Lift-Splat de LSS.
Hypothèses : une caméra, feature map 4×4, C = 2, D = {1, 2, 3}, grille BEV 4×4.
Vérifié par : conservation de masse (somme du nuage = somme BEV + points ignorés), matrice de contributions symétrique.

## Ce qui n'est pas fait
- Pas de CNN, pas de softmax appris, pas de gradient.
- Pas de multimodal (LiDAR, RADAR).
- Pas de détection 3D ni de HD map.

