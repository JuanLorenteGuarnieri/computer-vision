import numpy as np

# Define the points 3D A, B, C, D
A = np.array([3.44, 0.80, 0.82])
B = np.array([4.20, 0.80, 0.82])
C = np.array([4.20, 0.60, 0.82])
D = np.array([3.55, 0.60, 0.82])

# Create the matrix of points in homogeneous coordinates (add a column of ones)
points = np.array([A, B, C, D])
points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

# Apply SVD to the point matrix
u, s, vh = np.linalg.svd(points_homogeneous)

# The normal vector to the plane is the last vector of vh (corresponding to the smallest singular value)
plane_eq = vh[-1, :]



# Define a function to compute the distance of a point to the plane
def point_to_plane_distance(plane, point):
    """
    Compute the distance of a point to a plane.
    Parameters:
        plane: plane equation coefficients.
        point: point coordinates.
    Returns:
        Distance from the point to the plane.
    """
    a, b, c, d = plane
    x_p, y_p, z_p = point
    # Calculate the distance using the plane equation
    distance = abs(a * x_p + b * y_p + c * z_p + d) / np.sqrt(a**2 + b**2 + c**2)
    return distance

# Compute the distances for points A, B, C, D, and E
E = np.array([-0.01, 2.6, 1.21])

d_A = point_to_plane_distance(plane_eq, A)
d_B = point_to_plane_distance(plane_eq, B)
d_C = point_to_plane_distance(plane_eq, C)
d_D = point_to_plane_distance(plane_eq, D)
d_E = point_to_plane_distance(plane_eq, E)

np.set_printoptions(precision=4,linewidth=1024,suppress=True)

# Display the plane equation
print("TASK 4")
print()
print("Plane equation: ")
print(plane_eq.T)

# Print the distances
print()
print("Distances from points to the plane:")
print(f"A: {d_A:.2f} m")
print(f"B: {d_B:.2f} m")
print(f"C: {d_C:.2f} m")
print(f"D: {d_D:.2f} m")
print(f"E: {d_E:.2f} m")
