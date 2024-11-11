#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 3
#
# Title: Bundle Adjustment and Multiview Geometry
#
# Date: 26 October 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as scAlg
import csv
import scipy as sc
import scipy.optimize as scOptim
import scipy.io as sio

def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList


def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

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
    
    

def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate points in 3D from two sets of corresponding points in two images.
    Params:
        P1 (np.ndarray): First camera projection matrix (3x4).
        P2 (np.ndarray): Second camera projection matrix (3x4).
        pts1 (np.ndarray): 2D points in the first camera.
        pts2 (np.ndarray): 2D points in the second camera.
    Returns:
        np.ndarray: Triangulated 3D points.
    """
    n_points = pts1.shape[0]
    pts_3d_hom = np.zeros((n_points, 4))

    for i in range(n_points):
        A = np.array([
            (pts1[i, 0] * P1[2, :] - P1[0, :]),
            (pts1[i, 1] * P1[2, :] - P1[1, :]),
            (pts2[i, 0] * P2[2, :] - P2[0, :]),
            (pts2[i, 1] * P2[2, :] - P2[1, :])
        ])

        _, _, Vt = np.linalg.svd(A)
        pts_3d_hom[i] = Vt[-1]  # Last singular vector is the solution.

    # Convert to non-homogeneous coordinates.
    pts_3d = pts_3d_hom[:, :3] / pts_3d_hom[:, 3][:, np.newaxis]
    
    return pts_3d


def triangulate_points_from_cameras(R, t, K, pts1, pts2):
    """
    Triangulate points 3D, given two sets of points projected in 2D in two cameras.
    Params:
        R (np.ndarray): Rotation matrix between the cameras.
        t (np.ndarray): Translation vector between the cameras.
        K (np.ndarray): Intrinsic camera matrix.
        pts1 (np.ndarray): Points in the first camera.
        pts2 (np.ndarray): Points in the second camera.
    Returns:
        np.ndarray: Triangulated 3D points.
    """

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    return triangulate_points(P1, P2, pts1, pts2)


import numpy as np
from scipy.linalg import expm, logm

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

def resBundleProjection2(Op, x1Data, x2Data, K_c, nPoints):
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
    X1 = Op[6:].reshape((nPoints, 3))  # 3D points (each with 3 coordinates)
    
    # Compute rotation matrix from rotation vector theta using exponential map
    R_21 = expm(crossMatrix(theta))  # Compute R_21 from theta

    # Residuals array
    residuals = []

    # Compute residuals for each point
    for i in range(nPoints):
        # Get the 3D point in reference 1
        X = X1[i]

        # Project point X to image 1 (should match x1Data)
        x1_proj = K_c @ X
        x1_proj /= x1_proj[2]  # Normalize to homogeneous coordinates

        # Project point X to reference frame 2
        X2 = R_21 @ X + t_21  # Transform to ref 2

        # Project X2 to image 2 (should match x2Data)
        x2_proj = K_c @ X2
        x2_proj /= x2_proj[2]  # Normalize to homogeneous coordinates

        # Calculate residuals as the difference between observed and projected points
        residual_x1 = x1Data[:, i] - x1_proj[:2]  # Difference in image 1
        residual_x2 = x2Data[:, i] - x2_proj[:2]  # Difference in image 2
        
        # Append to residuals
        residuals.extend(residual_x1)
        residuals.extend(residual_x2)
        # print("Residual nº ", i, " completed")
    return np.array(residuals)


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

from scipy.optimize import least_squares
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


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_wc1 = np.loadtxt('T_w_c1.txt')
    T_wc2 = np.loadtxt('T_w_c2.txt')
    T_wc3 = np.loadtxt('T_w_c3.txt')
    K_c = np.loadtxt('K_c.txt')
    X_w = np.loadtxt('X_w.txt')

    x1Data = np.loadtxt('x1Data.txt')
    x2Data = np.loadtxt('x2Data.txt')
    x3Data = np.loadtxt('x3Data.txt')


    #Plot the 3D cameras and the 3D points
    fig3D = plt.figure(1)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_wc1, '-', 'C1')
    drawRefSystem(ax, T_wc2, '-', 'C2')
    drawRefSystem(ax, T_wc3, '-', 'C3')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', 0.1)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    # plt.show()      # CHANGED


    #Read the images
    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'
    path_image_3 = 'image3.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)


    # Construct the matches
    kpCv1 = []
    kpCv2 = []
    kpCv3 = []
    for kPoint in range(x1Data.shape[1]):
        kpCv1.append(cv2.KeyPoint(x1Data[0, kPoint], x1Data[1, kPoint],1))
        kpCv2.append(cv2.KeyPoint(x2Data[0, kPoint], x2Data[1, kPoint],1))
        kpCv3.append(cv2.KeyPoint(x3Data[0, kPoint], x3Data[1, kPoint],1))

    matchesList12 = np.hstack((np.reshape(np.arange(0, x1Data.shape[1]),(x2Data.shape[1],1)),
                                        np.reshape(np.arange(0, x1Data.shape[1]), (x1Data.shape[1], 1)),np.ones((x1Data.shape[1],1))))

    matchesList13 = matchesList12
    dMatchesList12 = indexMatrixToMatchesList(matchesList12)
    dMatchesList13 = indexMatrixToMatchesList(matchesList13)

    imgMatched12 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_2, kpCv2, dMatchesList12,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    imgMatched13 = cv2.drawMatches(image_pers_1, kpCv1, image_pers_3, kpCv3, dMatchesList13,
                                   None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(2)
    plt.imshow(imgMatched12)
    plt.title("{} matches between views 1 and 2".format(len(dMatchesList12)))
    # plt.draw()      # CHANGED

    plt.figure(3)
    plt.imshow(imgMatched13)
    plt.title("{} matches between views 1 and 3".format(len(dMatchesList13)))
    print('Close the figures to continue.')
    # plt.show()      # CHANGED

    # Project the points
    x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_w
    x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2) @ X_w
    x3_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_w
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]


    # Plot the 2D points
    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo')
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.title('Image 1')
    # plt.draw()      # CHANGED

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.title('Image 2')
    # plt.draw()      # CHANGED

    plt.figure(6)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.title('Image 3')
    print('Close the figures to continue.')
    # plt.show()      # CHANGED



    # Convertimos la rotación inicial T_wc2[:3, :3] a theta
    theta_init = crossMatrixInv(logm(T_wc2[:3, :3].astype('float64')))
    # Obtenemos el vector de traslación inicial de T_wc2
    t_init = T_wc2[:3, 3]

    # Preparamos los parámetros iniciales (theta y t juntos)
    T_init = np.hstack([theta_init, t_init])

    # Ejecutamos el ajuste de bundle adjustment
    R_opt, t_opt, X_opt = bundle_adjustment(x1Data, x2Data, K_c, T_init, X_w)

    X_opt = triangulate_points_from_cameras(R_opt, t_opt, K_c, x1Data.T, x2Data.T).T
    X_opt = (T_wc1 @ np.vstack([X_opt, np.ones((1, X_opt.shape[1]))])).T

    print("initial_theta: " + str(T_wc2[:3, :3]))
    print("optimized_theta: " + str(R_opt))
    print("initial_t_21: " + str(T_wc2[:3,3:4]))
    print("optimized_t_21: " + str(t_opt))
    print("initial_X1: " + str(X_w.T))
    print("optimized_X1: " + str(X_opt))
    
    fig3D = plt.figure(1)

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
    
    # Crear la matriz de transformación optimizada para la cámara C2
    T_wc2_opt = np.eye(4)
    T_wc2_opt[:3, :3] = R_opt
    T_wc2_opt[:3, 3] = t_opt
    
    # Proyectar los puntos optimizados en cada imagen usando T_wc1, T_wc2_opt, T_wc3
    x1_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_opt.T
    x2_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_opt) @ X_opt.T
    x3_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_opt.T

    # Normalizar las coordenadas para obtener las proyecciones en píxeles
    x1_p_opt /= x1_p_opt[2, :]
    x2_p_opt /= x2_p_opt[2, :]
    x3_p_opt /= x3_p_opt[2, :]
    
    # Imagen 1
    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plotResidual(x1Data, x1_p, 'k-')  # Residuals originales
    plt.plot(x1_p[0, :], x1_p[1, :], 'bo', label='Original Projection')
    plt.plot(x1_p_opt[0, :], x1_p_opt[1, :], 'go', label='Optimized Projection')  # Proyecciones optimizadas
    plt.plot(x1Data[0, :], x1Data[1, :], 'rx')
    plotNumberedImagePoints(x1Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 1')

    # Imagen 2
    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plotResidual(x2Data, x2_p, 'k-')
    plt.plot(x2_p[0, :], x2_p[1, :], 'bo', label='Original Projection')
    plt.plot(x2_p_opt[0, :], x2_p_opt[1, :], 'go', label='Optimized Projection')
    plt.plot(x2Data[0, :], x2Data[1, :], 'rx')
    plotNumberedImagePoints(x2Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 2')

    # Imagen 3
    plt.figure(6)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plotResidual(x3Data, x3_p, 'k-')
    plt.plot(x3_p[0, :], x3_p[1, :], 'bo', label='Original Projection')
    plt.plot(x3_p_opt[0, :], x3_p_opt[1, :], 'go', label='Optimized Projection')
    plt.plot(x3Data[0, :], x3Data[1, :], 'rx')
    plotNumberedImagePoints(x3Data[0:2, :], 'r', 4)
    plt.legend()
    plt.title('Image 3')

    print('Close the figures to continue.')
    plt.show()