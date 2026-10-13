import numpy as np

# Takes a 3D Xw point from the world coordinates system
# Returns the same point, after rotation (R) and translation t
# In the 3D camera coordinates system
def world_to_camera (Xw, R, t) : 
    return np.dot(R,Xw) + t

# Take a point from the camera coordinates system
# Projects that point in the the image coordinates system
# Knowing distance f from the optical center
# And the optical center coordinates (cx, cy)
# Returns coordinates (u,v) in the the image coordinates system
def project_camera (Xc, f, cx, cy) : 
    x, y, z = Xc
    u = f * x / z + cx
    v = f * y / z + cy

    return np.array([u,v]) 

# Applies successively the coordinates system changes from the world (Xw) to the camera (Xc)
# And projects the Xc point to the image coordinates system (U, V)
# Finally, returns the new point (u,v) if Zc > 0 in front of the camera
# None if Zc <= 0. Behind or at the same position with the camera
def project_full (X_w, R, t, f, cx, cy) : 
    Xc = world_to_camera(X_w, R, t)
    Zc = Xc[2]
    if (Zc > 0) : 
        u, v = project_camera (Xc, f, cx, cy)
        return np.array([u, v])
    else : 
        return None


# Returns the point Xc in the World coordinates system
def camera_to_world (Xc, R, t) : 
    return np.dot (np.transpose(R), (Xc - t))

# Knowing the depth (Zc), distance from the optical center f, and optical center coordinates (cx, cy)
# Returns the (Xc, Yc, Zc) point in the camera coordinates system
def backproject (u,v, depth, f, cx, cy) : 
    return np.array([
        (depth / f) * (u - cx), 
        (depth / f) * (v - cy), 
        depth])


# Instead of using f, cx, cy like that,
# Let's put it in a matrix K 
# Which is intrinsics matrix
def intrinsics_matrix(fx, fy, cx, cy) : 
    return np.array ([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

# Same as project camera but with the intrinsics matrix
def project_camera_K(Xc, K) : 
    p = np.dot (K, Xc)
    return (p / p[2])[:2]

