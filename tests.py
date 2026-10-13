from camera_geometry import *
import numpy as np
import matplotlib.pyplot as plt

f, cx, cy = 500, 320, 240
R, t = np.eye(3), np.zeros(3)
t2 = np.array([-1.0, 0.0, 0.0])   # caméra 2 décalée de 1 m sur x

assert np.allclose(project_camera(np.array([0., 0., 5.]), f, cx, cy), [320, 240])
assert np.allclose(project_camera(np.array([1., 0., 5.]), f, cx, cy), [420, 240])
assert np.allclose(project_camera(np.array([2., 0., 10.]), f, cx, cy), [420, 240])
assert np.allclose(backproject(420, 240, 5, f, cx, cy), [1, 0, 5])
assert project_full(np.array([1., 0., 5.]), np.array([[0,0,1],[0,1,0],[-1,0,0]]), t, f, cx, cy) is None

# Parallax test
# See how the difference (disparity) between two points from different cameras evolve
# When Z become greater
# Helps us deal with the depth problem when moving from the camera coordinates system to the image coordinates system
for Z in [5, 10, 20, 40]:
    point = np.array([1.0, 0.0, Z])
    u1, _ = project_full(point, R, t, f, cx, cy)
    u2, _ = project_full(point, R, t2, f, cx, cy)
    print(f"Z = {Z:>3}  u1 = {u1:7.2f}  u2 = {u2:7.2f}  disparité = {u1 - u2:6.2f}")

K = intrinsics_matrix(f,f,cx, cy)

for Xc in [np.array([0., 0., 5.]),
           np.array([1., 0., 5.]),
           np.array([0., 2., 4.]),
           np.array([-1., 0.5, 10.])]:
    assert np.allclose(project_camera(Xc, f, cx, cy), project_camera_K(Xc, K))

R_test = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # 90 deg around z
t_test = np.array([0.5, -2.0, 1.0])
Xw = np.array([1.0, 2.0, 3.0])
Xc = world_to_camera(Xw, R_test, t_test)
Xw_back = camera_to_world(Xc, R_test, t_test)
assert np.allclose(Xw, Xw_back)

print("OK")