import numpy as np
import plotData as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from scipy.optimize import least_squares

# Cross-product matrix for rotation vector
def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]], 
                  [x[2], 0, -x[0]], 
                  [-x[1], x[0], 0]])
    return M

# Inverse cross-product for retrieving rotation vector from a skew-symmetric matrix
def crossMatrixInv(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def project_point(K, R, t, X):
    """Projects a 3D point X in reference frame onto 2D image using intrinsic matrix K, rotation R, and translation t."""
    X_cam = R @ X + t
    x_proj = K @ X_cam
    x_proj /= x_proj[2]  # Convert to homogeneous coordinates
    return x_proj[:2]  # Return only the 2D coordinates




def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Calculate the reprojection residuals for bundle adjustment using two views.
    
    Parameters:
        Op (array): Optimization parameters including T_21 (rotation and translation between views) 
                    and X1 (3D points in reference frame 1).
        x1Data (array): (3 x nPoints) 2D points in image 1 (homogeneous coordinates).
        x2Data (array): (3 x nPoints) 2D points in image 2 (homogeneous coordinates).
        K_c (array): (3 x 3) intrinsic calibration matrix.
        nPoints (int): Number of 3D points.
        
    Returns:
        res (array): Residuals, which are the errors between the observed 2D matched points 
                     and the projected 3D points.
    """
    # Extract rotation (theta) and translation (t) from optimization parameters
    theta = Op[:3]                # Rotation vector (3 parameters)
    t_21 = Op[3:6]                # Translation vector (3 parameters)
    X1 = Op[6:].reshape((3, nPoints))  # 3D points (each with 3 coordinates)
    
    # Compute rotation matrix from rotation vector theta using exponential map
    R_21 = expm(crossMatrix(theta))  # Compute R_21 from theta

    # Residuals array
    residuals = []

    # Compute residuals for each point
    for i in range(nPoints):
        # Project point in ref 1 to ref 2
        x1_proj = project_point(K_c, R_21, t_21, X1[:, i])
        
        # Calculate residuals for x and y coordinates
        residuals.extend((x1_proj - x2Data[:2, i]).tolist())
        # print("Residual nº ", i, " completed")
    return np.array(residuals)

def bundle_adjustment(x1Data, x2Data, K_c, T_init, X_in):
    """
    Perform bundle adjustment using least-squares optimization.
    x1Data: Observed 2D points in image 1
    x2Data: Observed 2D points in image 2
    K_c: Intrinsic calibration matrix
    T_init: Initial transformation parameters (theta, t)
    X_in: Initial 3D points
    """
    
    # Definir la fracción de los puntos a usar en la muestra
    sample_fraction = 0.3  # Por ejemplo, el 30% de los puntos
    nPoints_sample = int(sample_fraction * X_in.shape[1])

    # Seleccionar una muestra de los índices de puntos
    sample_indices = np.random.choice(X_in.shape[1], nPoints_sample, replace=False)

    # Tomar los puntos de la muestra usando los índices seleccionados
    X_init_sample = X_in[:, sample_indices]
    x1Data_sample = x1Data[:, sample_indices]
    x2Data_sample = x2Data[:, sample_indices]
    # Puntos iniciales en 3D (X_init_sample) que ya tienes
    X_init = X_init_sample[:3, :]

    initial_params = np.hstack([T_init[:3], T_init[3:], X_init.T.flatten()])
    # initial_params = np.hstack([[ 0.011, 2.6345, 1.4543], [-1.4445, -2.4526, 18.1895], X_init.T.flatten()])

    # Run bundle adjustment optimization
    result = least_squares(resBundleProjection, initial_params, args=(x1Data_sample, x2Data_sample, K_c, nPoints_sample), method='trf') #method='lm'

    # Retrieve optimized parameters
    Op_opt = result.x
    theta_opt = Op_opt[:3]
    t_opt = Op_opt[3:6]
    X_opt = Op_opt[6:].reshape((nPoints_sample, 3))

    # Return optimized rotation, translation, and 3D points
    return expm(crossMatrix(theta_opt)), t_opt, X_opt

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


def triangulate(x1, x2, K1, K2, T_wc1, T_wc2):
    """
    Triangulate points using two sets of 2D points and camera matrices.

    Arguments:
    x1 -- Points in camera 1's image (shape 3xN, homogeneous coordinates)
    x2 -- Points in camera 2's image (shape 3xN, homogeneous coordinates)
    K1, K2 -- Intrinsic matrices of camera 1 and 2
    T_wc1, T_wc2 -- Extrinsic transformations (world to camera)

    Returns:
    X -- Triangulated 3D points (shape 4xN, homogeneous coordinates)
    """
    # Convert extrinsics to projection matrices
    P1 = K1 @ T_wc1[:3, :]  # Projection matrix for camera 1
    P2 = K2 @ T_wc2[:3, :]  # Projection matrix for camera 2

    # Number of points
    n_points = x1.shape[1]
    X = np.zeros((4, n_points))

    for i in range(n_points):
        # Formulate the linear system A * X = 0
        A = np.vstack([
            x1[0, i] * P1[2, :] - P1[0, :],
            x1[1, i] * P1[2, :] - P1[1, :],
            x2[0, i] * P2[2, :] - P2[0, :],
            x2[1, i] * P2[2, :] - P2[1, :]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X[:, i] = Vt[-1]  # Homogeneous 3D point (last row of V)

    # Normalize homogeneous coordinates
    X /= X[3, :]
    return X
  
  
  
  
  

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)
    


def resBundleProjectionFishEye(Op, x1Data, x2Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, nPoints):
    """
    Calculate reprojection residuals for bundle adjustment with four fish-eye cameras.
    
    Parameters:
        Op (array): Optimization parameters including T_wAwB (6 params) and 3D points (X).
        x1Data (array): (3 x nPoints) Observed 2D points in image 1.
        x2Data (array): (3 x nPoints) Observed 2D points in image 2.
        K_1, K_2 (array): Intrinsic calibration matrices for the two cameras.
        D1_k, D2_k (array): Distortion coefficients for the two cameras.
        T_wc1, T_wc2 (array): Extrinsic transformations for the stereo cameras.
        nPoints (int): Number of 3D points.
        
    Returns:
        res (array): Residuals between observed and projected points.
    """
    
    # Extract rotation (theta) and translation (t) from optimization parameters
    theta = Op[:3]
    t_21 = Op[3:6]
    X1 = Op[6:].reshape((3, nPoints))  # 3D points

    # Compute rotation matrix from theta
    R_21 = expm(crossMatrix(theta))  # Rotation matrix

    residuals = []

    for i in range(nPoints):
        # Project 3D points in camera 1
        x1_proj = project_kannala_brandt(X1[:, i], K_1, D1_k)

        # Transform 3D points to camera 2 reference frame
        X2 = R_21 @ X1[:, i] + t_21

        # Project 3D points in camera 2
        x2_proj = project_kannala_brandt(X2, K_2, D2_k)

        # Compute residuals for both cameras
        residuals.extend((x1_proj[:2] - x1Data[i, :2]).tolist())
        residuals.extend((x2_proj[:2] - x2Data[i, :2]).tolist())

    return np.array(residuals)
  
    # Extract T_wAwB (rotation and translation) and 3D points (X)
    theta = Op[:3]  # Rotation vector
    t_wAwB = Op[3:6]  # Translation vector
    X = Op[6:].reshape((3, nPoints))  # 3D points
    
    # Compute T_wAwB from theta and translation
    R_wAwB = expm(crossMatrix(theta))
    T_wAwB = np.eye(4)
    T_wAwB[:3, :3] = R_wAwB
    T_wAwB[:3, 3] = t_wAwB
    
    # Compute residuals
    residuals = []
    for i in range(nPoints):
        # Transform point to camera 1
        print(X[:, i])
        # X_c1 = T_wc1 @ np.vstack((X[:, i], 1))
        # x1_proj = project_kannala_brandt(X_c1[:3], K_1, D1_k)
        x1_proj = project_kannala_brandt(X[:, i], K_1, D1_k)
        
        # Transform point to camera 2
        # X_c2 = T_wc2 @ np.vstack((T_wAwB @ np.hstack((X[:, i], 1))[:3], 1))
        # x2_proj = project_kannala_brandt(X_c2[:3], K_2, D2_k)
        x2_proj = project_kannala_brandt(X[:, i], K_2, D2_k)
        
        # Compute residuals
        res_x1 = x1_proj - x1Data[:2, i]
        res_x2 = x2_proj - x2Data[:2, i]
        
        residuals.extend(res_x1.tolist() + res_x2.tolist())
    
    return np.array(residuals)


def bundle_adjustment_fish_eye(x1Data, x2Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, T_init, X_in):
    """
    Perform bundle adjustment for fish-eye stereo setup.
    """
    # Flatten initial parameters (T_wAwB and 3D points)
    initial_params = np.hstack([T_init[:3], T_init[3:], X_in.flatten()])
    
    # Run least-squares optimization
    result = least_squares(
        resBundleProjectionFishEye, initial_params,
        args=(x1Data, x2Data, K_1, K_2, D1_k, D2_k, T_wc1, T_wc2, X_in.shape[0]),
        method='trf'
    )
    
    # Retrieve optimized parameters
    Op_opt = result.x
    theta_opt = Op_opt[:3]
    t_opt = Op_opt[3:6]
    X_opt = Op_opt[6:].reshape((-1, 3))
    
    # Compute optimized T_wAwB
    R_opt = expm(crossMatrix(theta_opt))
    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt
    
    return T_opt, X_opt

def plotImages(img1, img2, img3, x1Data, x1_p, x1_p_opt, x2Data, x2_p, x2_p_opt, x3Data, x3_p, x3_p_opt):
  
    # Imagen 1
    plt.figure(4)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')  # Residuals originales
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo', label='Original Projection')
    plt.plot(x1_p_opt[0, :], x1_p_opt[1, :], 'go', label='Optimized Projection')  # Proyecciones optimizadas
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 1')

    # Imagen 2
    plt.figure(5)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo', label='Original Projection')
    plt.plot(x2_p_opt[0, :], x2_p_opt[1, :], 'go', label='Optimized Projection')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 2')

    # Imagen 3
    plt.figure(6)
    plt.imshow(img3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo', label='Original Projection')
    plt.plot(x3_p_opt[0, :], x3_p_opt[1, :], 'go', label='Optimized Projection')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 3')

    print('Close the figures to continue.')
    plt.show()
    
def drawSystem(T_wc1, T_wc2, points_3d):
    # Create the figure and system references.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pd.drawRefSystem(ax, np.eye(4), '-', 'W')   # World.
    pd.drawRefSystem(ax, T_wc1, '-', 'C1')     # Camera 1.
    pd.drawRefSystem(ax, T_wc2, '-', 'C2')     # Camera 2.

    # Plot the points over the figure.
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', label='Computed', marker='o')
    plt.show()
    fig3D = plt.figure(1)
    return
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    drawRefSystem(ax, T_wc3, '-', 'C3')

    ax2 = ax
    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    ax2.scatter(X_opt.T[0, :], X_opt.T[1, :], X_opt.T[2, :], marker='.', c='g')
    plotNumbered3DPoints(ax, X_w, 'r', 0.1)
    plotNumbered3DPoints(ax, X_opt, 'g', 0.1)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    