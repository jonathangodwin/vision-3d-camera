import numpy as np
from camera_geometry import *

H, W = 4, 4  # Image size
C = 2        # Context size
D = np.array([1.0, 2.0, 3.0])  # Depths
f, cx, cy = 2.0, 2.0, 2.0      # Intrinsics of the camera
K = intrinsics_matrix(f, f, cx, cy)
E = np.eye(4)                  # Extrinsics camera to world
res = 1.0                      # BEV Grid resolution
BEV_H, BEV_W = 4, 4            # BEV Grid Size

# Fake feature map: each patch (h, w) has C features
F = np.zeros((H, W, C))
for h in range(H):
    for w in range(W):
        F[h, w] = [h + w, 10 * h + w]

# Context is the feature itself in this toy
c = F.copy()

# Depth distribution: same for every patch in this toy
alpha = np.zeros((H, W, D.shape[0]))  # Size (H, W, len(D))
alpha_values = np.array([0.2, 0.5, 0.3])
for h in range(H):
    for w in range(W):
        alpha[h, w, :] = alpha_values.copy()

# ____________________ LIFT _________________________

# For each patch of the feature map and each candidate depth,
# create a 3D point in the camera coordinate system (back-projection),
# and a feature vector weighted by the depth probability.
points = []
for h in range(H):
    for w in range(W):
        for d in range(D.shape[0]):
            points.append(backproject(w, h, D[d], f, cx, cy))
points = np.array(points)

lifted_features = np.zeros((H * W * D.shape[0], C))
idx = 0
for h in range(H):
    for w in range(W):
        for k in range(D.shape[0]):
            lifted_features[idx] = alpha[h, w, k] * c[h, w]
            idx += 1

# ____________________ SPLAT ________________________

# Project each 3D point into the BEV grid and accumulate its features.
BEV_grid = np.zeros((BEV_H, BEV_W, C))
counts = np.zeros((BEV_H, BEV_W))
offset = 2
ignored_features = np.array([0.0, 0.0])  # features falling outside the grid

for k in range(points.shape[0]):
    q = np.ones((4))
    q[:3] = points[k]
    q = np.dot(E, q)
    qx, qy = q[0], q[1]
    i = int(np.floor(qx / res) + offset)
    j = int(np.floor(qy / res) + offset)
    if 0 <= i < BEV_W and 0 <= j < BEV_H:
        BEV_grid[i, j] += lifted_features[k]
        counts[i, j] += 1
    else:
        ignored_features += lifted_features[k]

# ____________________ VERIFICATION _________________

print("Mass conservation test")
S_nuage = lifted_features.sum(axis=0)
S_bev = BEV_grid.sum(axis=(0, 1))
print(f"Sum of the point cloud: {S_nuage}")
print(f"S_bev={S_bev} + ignored={ignored_features} = {S_bev + ignored_features}")

print("Contribution distribution")
print(counts)