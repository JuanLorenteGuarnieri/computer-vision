import numpy as np
import plotData as pd
import matplotlib.pyplot as plt

# Function to load the intrinsic matrix (K) and distortion coefficients (D) from a file
def load_calibration_data(K_file, D_file):
    K = np.loadtxt(K_file)  # Load intrinsic matrix K
    D = np.loadtxt(D_file)  # Load distortion coefficients D
    return K, D

def load_transformation_matrix(file_path):
    """Load the transformation matrix from a file (either T_wc1 or T_wc2)."""
    return np.loadtxt(file_path)

def load_points(file_path):
    """Load 2D points from a file (in homogeneous coordinates)."""
    return np.loadtxt(file_path)

# Function to unproject a 2D point (u) back into 3D space using the Kannala-Brandt model
def unproject_kannala_brandt(u, K, D):
    """
    Unprojection usando el modelo Kannala-Brandt.

    Parámetros:
    - u: Punto en la imagen (np.array([u_x, u_y])).
    - K: Matriz intrínseca de la cámara (3x3).
    - D: Coeficientes de distorsión [k1, k2, k3, k4].

    Retorna:
    - v: Vector en la esfera unitaria (np.array([x, y, z])).
    """
    # Paso 1: Convertir el punto en la imagen a coordenadas de la cámara
    u_hom = np.array([u[0], u[1], 1.0])  # Coordenadas homogéneas
    x_c = np.linalg.inv(K) @ u_hom       # Coordenadas en el sistema de la cámara

    # Paso 2: Calcular r y phi
    r = np.sqrt((x_c[0]**2 + x_c[1]**2) / x_c[2]**2)
    phi = np.arctan2(x_c[1], x_c[0])

    # Paso 3: Resolver el polinomio para obtener theta usando los coeficientes de D
    # Polinomio de 9º grado: d(theta) = r
    # Expresado como: k4 * theta^9 + k3 * theta^7 + k2 * theta^5 + k1 * theta^3 + theta - r = 0
    coeffs = [D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -r]  # Coeficientes del polinomio
    roots = np.roots(coeffs)                      # Soluciones del polinomio

    # Filtrar soluciones reales
    theta_solutions = roots[np.isreal(roots)].real
    # Seleccionar la única solución positiva (asumimos que es única y válida)
    theta = theta_solutions[theta_solutions >= 0][0]

    # Paso 4: Calcular el vector en la esfera unitaria usando theta y phi
    v = np.array([
        np.sin(theta) * np.cos(phi),  # x
        np.sin(theta) * np.sin(phi),  # y
        np.cos(theta)                 # z
    ])

    return v

# Kannala-Brandt projection model (already defined in the previous code)
def project_kannala_brandt(X, K, D):
    # Extract 3D coordinates
    x, y, z = X

    # Normalize the 3D point by the z-coordinate
    r = np.sqrt(x**2 + y**2)

    # Apply the Kannala-Brandt model distortion
    theta = np.arctan(r / z)
    theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)

    # Project the point into the image plane
    x_proj = K[0, 0] * theta_d * (x / r)
    y_proj = K[1, 1] * theta_d * (y / r)
    u = np.array([x_proj + K[0, 2], y_proj + K[1, 2], 1])  # 2D projection with homogeneous coordinates
    return u

# Function to load point matches from a text file (1 column per point, 3 rows for coordinates)
def load_points(file):
    points = np.loadtxt(file)  # Load the points (1 column per point)
    return points.T  # Transpose to get points in row-major format

def plot_stereo_camera_axis(ax, T_wc0, T_c0c1, T_c0c2, name):
    pd.drawRefSystem(ax, T_wc0, '-', name+'_W')
    pd.drawRefSystem(ax, T_wc0 @ T_c0c1, '-', name+'_C1')
    pd.drawRefSystem(ax, T_wc0 @ T_c0c2, '-', name+'_C2')
    
def triangulate_point(ray1, ray2, T_wc1, T_wc2):
    """
    Triangulate a 3D point given two rays and their camera poses.
    """
    # Origin of the rays in world coordinates
    origin1 = T_wc1[:3, 3]
    origin2 = T_wc2[:3, 3]
    
    # Directions of the rays in world coordinates
    direction1 = T_wc1[:3, :3] @ ray1
    direction2 = T_wc2[:3, :3] @ ray2
    
    # Calculate the 3D point that minimizes distance to both rays
    A = np.stack([direction1, -direction2], axis=1)
    b = origin2 - origin1
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Compute the 3D point in world coordinates
    point_3d = (origin1 + t[0] * direction1 + origin2 + t[1] * direction2) / 2
    return point_3d

def triangulate_points(x1, x2, K1, D1, K2, D2, T_wc1, T_wc2):
    """
    Triangulate multiple 3D points from pairs of 2D image points in a stereo system.
    """
    points_3d = []
    for i in range(x1.shape[0]):
        u1, u2 = x1[i], x2[i]
        
        # Unproject points into 3D rays using the respective calibration data
        ray1 = unproject_kannala_brandt(u1, K1, D1)
        ray2 = unproject_kannala_brandt(u2, K2, D2)
        
        # Triangulate the 3D point
        point_3d = triangulate_point(ray1, ray2, T_wc1, T_wc2)
        points_3d.append(point_3d)
        
    return np.array(points_3d)

def main():
    # Load calibration data (using the provided text files)

    K1_file = './p5/ext/K_1.txt'
    D1_file = './p5/ext/D1_k_array.txt'
    K2_file = './p5/ext/K_2.txt'
    D2_file = './p5/ext/D2_k_array.txt'

    K1, D1 = load_calibration_data(K1_file, D1_file)
    K2, D2 = load_calibration_data(K2_file, D2_file)
    T_wc1 = load_transformation_matrix('./p5/ext/T_wc1.txt')
    T_wc2 = load_transformation_matrix('./p5/ext/T_wc2.txt')

    # Load point matches from the provided files (x1.txt, x2.txt, x3.txt, x4.txt)
    x1 = load_points('./p5/ext/x1.txt')
    x2 = load_points('./p5/ext/x2.txt')
    x3 = load_points('./p5/ext/x3.txt')
    x4 = load_points('./p5/ext/x4.txt')

    # Assuming that the virtual 3D points are provided for verification
    X1 = np.array([3, 2, 10])  # 3D point X1
    X2 = np.array([-5, 6, 7])  # 3D point X2
    X3 = np.array([1, 5, 14])  # 3D point X3

    # Project the 3D points using the Kannala-Brandt model for Camera 1 (K1, D1)
    u1_proj = project_kannala_brandt(X1, K1, D1)
    u2_proj = project_kannala_brandt(X2, K1, D1)
    u3_proj = project_kannala_brandt(X3, K1, D1)

    # Print the results
    print("Projected u1:", u1_proj)
    print("Projected u2:", u2_proj)
    print("Projected u3:", u3_proj)

    # Optionally, calculate the difference between projected and expected results if you have them
    # Assuming the ground truth 2D points are known (you can replace them with the actual values)
    u1_expected = np.array([503.387, 450.1594])
    u2_expected = np.array([267.9465, 580.4671])
    u3_expected = np.array([441.0609, 493.0671])

    diff_u1 = np.linalg.norm(u1_proj[:2] - u1_expected)
    diff_u2 = np.linalg.norm(u2_proj[:2] - u2_expected)
    diff_u3 = np.linalg.norm(u3_proj[:2] - u3_expected)

    print("Difference for u1:", diff_u1)
    print("Difference for u2:", diff_u2)
    print("Difference for u3:", diff_u3)


    # Unproject the points back to 3D space
    X1_unproj = unproject_kannala_brandt(u1_proj, K1, D1)
    X2_unproj = unproject_kannala_brandt(u2_proj, K1, D1)
    X3_unproj = unproject_kannala_brandt(u3_proj, K1, D1)
    #Normalize the X1 and X2 numpy vectors
    x1_normalized = X1 / np.linalg.norm(X1)
    x2_normalized = X2 / np.linalg.norm(X2)
    x3_normalized = X3 / np.linalg.norm(X3)
    # X1_unproj /= np.linalg.norm(X1_unproj)
    # X2_unproj /= np.linalg.norm(X2_unproj)
    # X3_unproj /= np.linalg.norm(X3_unproj)
    
    print("norm X1:", x1_normalized)
    print("norm X2:", x2_normalized)
    print("norm X3:", x3_normalized)

    # Print the unprojected points (back to 3D space)
    print("Unprojected X1:", X1_unproj)
    print("Unprojected X2:", X2_unproj)
    print("Unprojected X3:", X3_unproj)

    # Optionally, compare the unprojected 3D points with the original ones
    # print("Difference for X1:", np.linalg.norm(X1_unproj - X1))
    # print("Difference for X2:", np.linalg.norm(X2_unproj - X2))
    # print("Difference for X3:", np.linalg.norm(X3_unproj - X3))

    # Perform triangulation for each pair of points
    # X = triangulate(x1, x2, K1, K2, T_wc1, T_wc2)

    # Print the triangulated 3D point
    # print("Triangulated 3D point:", X)

    # Plot camera axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_stereo_camera_axis(ax, T_wc1, T_wc1, T_wc2, 'R0')
    plt.show()

    # Triangulate 3D points
    points_3d = triangulate_points(x1, x2, K1, D1, K2, D2, T_wc1, T_wc2)
    print("3D points:\n", points_3d)
    
    # Create the figure and system references.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pd.drawRefSystem(ax, np.eye(4), '-', 'W')   # World.
    pd.drawRefSystem(ax, T_wc1, '-', 'C1')     # Camera 1.
    pd.drawRefSystem(ax, T_wc2, '-', 'C2')     # Camera 2.

    # Plot the points over the figure.
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', label='Computed', marker='o')
    plt.show()





# Run the main function
if __name__ == "__main__":
    main()
