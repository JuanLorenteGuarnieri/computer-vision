# Lorente Guarnieri, Juan (NIP 816020)
# Bielsa Uche, Jaime (NIP 819033)
# File: task_1.py
# Date: October 13th, 2024
# Master in Graphics, Robotics and Computer Vision, Universidad de Zaragoza.
# Subject: Compuer Vision
# Description: tasks 2 and 3 of the second practice of the subject.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import plotData as pd
sys.path.append(os.path.abspath('./p2/ext'))

    


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

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    """
    Calculate the reprojection residuals for bundle adjustment using two views in a vectorized manner.
    
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
    theta = Op[:3]                     # Rotation vector (3 parameters)
    t_21 = Op[3:6]                     # Translation vector (3 parameters)
    X1 = Op[6:].reshape((nPoints, 3))  # 3D points (each with 3 coordinates)

    # Compute rotation matrix from rotation vector theta using exponential map
    R_21 = expm(crossMatrix(theta))    # Compute R_21 from theta

    # Project all 3D points to image 1 (camera 1)
    x1_proj = K_c @ X1.T               # Project all points in one operation
    x1_proj /= x1_proj[2]              # Normalize to homogeneous coordinates

    # Transform all points from frame 1 to frame 2
    X2 = (R_21 @ X1.T).T + t_21        # Transform points to reference frame 2

    # Project all transformed points to image 2 (camera 2)
    x2_proj = K_c @ X2.T               # Project all transformed points in one operation
    x2_proj /= x2_proj[2]              # Normalize to homogeneous coordinates

    # Calculate residuals as the difference between observed and projected points
    residual_x1 = (x1Data[:2] - x1_proj[:2]) * (x1Data[:2] - x1_proj[:2])   # Difference for image 1
    residual_x2 = (x2Data[:2] - x2_proj[:2]) * (x2Data[:2] - x2_proj[:2])   # Difference for image 2

    # Concatenate and return residuals as a flat array
    residuals = np.hstack((residual_x1, residual_x2)).flatten()
    return residuals
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
        residual_x1 = (x1Data[:, i] - x1_proj[:2]) * (x1Data[:, i] - x1_proj[:2])  # Difference in image 1
        residual_x2 = (x2Data[:, i] - x2_proj[:2]) * (x2Data[:, i] - x2_proj[:2])  # Difference in image 2

        # Append to residuals
        residuals.extend(residual_x1)
        residuals.extend(residual_x2)
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
    sample_fraction = 1 #0.3  # Por ejemplo, el 30% de los puntos
    x1Data_sample = x1Data
    x2Data_sample = x2Data
    X_init = X_in[:3, :]
    nPoints_sample = int(sample_fraction * X_in.shape[1])
    if sample_fraction < 1:
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
    # scale_factor = np.linalg.norm(t_init) / np.linalg.norm(t_opt)
    # t_opt *= scale_factor
    X_opt = Op_opt[6:].reshape((nPoints_sample, 3))

    # Return optimized rotation, translation, and 3D points
    return expm(crossMatrix(theta_opt)), t_opt, X_opt


def obtain_proyection_matrices(K, R, t):
    """
    Compute the projection matrices P1 and P2 from the intrinsic matrix, rotation and translation.
    Params:
        K: Camera intrinsic matrix (3x3).
        R: Rotation matrix (3x3).
        t: Translation vector (3x1).
    """
    # First projection matrix P1 assumes camera 1 is at the origin.
    # Second projection matrix P2 uses the correct rotation and translation.
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    
    return P1, P2


def compute_homography(K1, K2, R, t, plane_normal, d):
    """
    Compute the homography from the floor plane.
    Params:
        K1, K2: Intrinsic camera matrices of camera 1 and 2.
        R: Rotation matrix from camera 1 to camera 2.
        t: Translation vector from camera 1 to camera 2.
        plane_normal: Normal vector of the plane in the first camera frame.
        d: Distance from the origin to the plane.
    """
    t_nT = np.outer(t, plane_normal)
    H = K2 @ (R + t_nT / d) @ np.linalg.inv(K1)
    return H


def visualize_point_transfer(H, img1, img2, pts1):
    """
    Visualize point transfer using the estimated homography.
    Params:
        H: Homography matrix.
        img1, img2: The two images.
        pts1: Points in the first image to transfer.
    Returns:
        pts2: Transferred points in the second image.
    """
    # Convert points to homogeneous coordinates
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    
    # Apply homography to transfer points
    pts2_h = (H @ pts1_h.T).T
    
    # Convert back to 2D
    pts2 = pts2_h[:, :2] / pts2_h[:, 2].reshape(-1, 1)
    
    # Plot the points on both images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(pts1[:, 0], pts1[:, 1], color='r')
    plt.title("Points in Image 1")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(pts2[:, 0], pts2[:, 1], color='b')
    plt.title("Transferred Points in Image 2")
    
    plt.show()

    return pts2


def compute_homography(pts_src, pts_dst):
    """
    findHomography implementation.
    """
    A = []
    
    # Build the matrix A as seen in the slides.
    for i in range(pts_src.shape[0]):
        x, y = pts_src[i, 0], pts_src[i, 1] 
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    
    # Get the eigen values.
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    # Normalize the homography matrix.
    H = H / H[2, 2]

    return H


def normalize_points(pts):
    """
    Normalize points so that the centroid is at the origin and the average distance to the origin is sqrt(2).
    """
    # Get the centroid of the points.
    centroid = np.mean(pts, axis=0)
    
    # Substract the centroid from the points so that the centroid is at the origin.
    pts_shifted = pts - centroid
    
    # Compute the average distance to the origin.
    avg_dist = np.mean(np.linalg.norm(pts_shifted, axis=1))
    
    # Scale the points so that the average distance is sqrt(2).
    scale = np.sqrt(2) / avg_dist
    pts_normalized = pts_shifted * scale
    
    # Build and return the transformation matrix.
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    return pts_normalized, T


def compute_fundamental_matrix(x1, x2):
    """
    Estimate the fundamental matrix using the 8-point algorithm.
    Params:
        x1, x2: Corresponding points in the two images.
    """
    # Normalie the points.
    x1_normalized, T1 = normalize_points(x1)
    x2_normalized, T2 = normalize_points(x2)
    
    # Build the matrix A.
    N = x1_normalized.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1x = x1_normalized[i, 0]
        y1x = x1_normalized[i, 1]
        x2x = x2_normalized[i, 0]
        y2x = x2_normalized[i, 1]
        A[i] = [x1x*x2x, x1x*y2x, x1x, y1x*x2x, y1x*y2x, y1x, x2x, y2x, 1]
    
    # Solve Af = 0 using SVD.
    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint.
    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0  # Third singular value is 0.
    F_normalized = U @ np.diag(S) @ Vt
    
    F = T2.T @ F_normalized @ T1
    
    # Normalize the fundamental matrix so that the last element is 1.
    return F / F[2, 2]


def decompose_essential_matrix(E):
    """
    Decompose the essential matrix E into two possible rotation matrices (R1, R2) and a translation vector t.
    Params:
        E: Essential matrix (3x3).
    Returns:
        R1, R2: Two possible rotation matrices (3x3).
        t: Translation vector (3x1).
    """
    # Essential matrix SVD.
    U, S, Vt = np.linalg.svd(E)
    
    # Check if E has two equal singular values and one of them close to zero.
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    # Aux matrix W.
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    # Possible rotations.
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Translation is the third element of U.
    t = U[:, 2]
    
    # Assure that the rotation matrices are valid (determinant = 1).
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return R1, R2, t


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


def triangulate_points2(P1, P2, pts1, pts2):
    """
    Triangulate points in 3D from two sets of corresponding points in two images.
    Params:
        P1 (np.ndarray): First camera projection matrix (3x4).
        P2 (np.ndarray): Second camera projection matrix (3x4).
        pts1 (np.ndarray): 2D points in the first camera.
        pts2 (np.ndarray): 2D points in the second camera.
    """
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)[:, :2]
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)[:, :2]

    pts_4d_h = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    pts_3d = pts_4d_h[:3] / pts_4d_h[3]  # Convert from homogeneous to 3D coordinates.
    return pts_3d.T


def is_valid_solution(R, t, K, pts1, pts2):
    """
    Check if a solution (R, t) generates valid 3D points (in front of both cameras).
    Params:
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        K (np.ndarray): Intrinsic camera matrix.
        pts1 (np.ndarray): Points in the first image.
        pts2 (np.ndarray): Points in the second image.
    """
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    # Triangulate points.
    pts_3d = triangulate_points(P1, P2, pts1, pts2)

    # Check if the points are in front of both cameras (Z coordinate positive).
    pts_cam1 = pts_3d[:, 2]
    pts_cam2 = (R @ pts_3d.T + t.reshape(-1, 1))[2, :]

    # Return true if all points are in front of both cameras.
    return np.all(pts_cam1 > 0) and np.all(pts_cam2 > 0)


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


#################### 2.1 Epipolar lines visualization ########################

# Load images and the fundamental matrix.
img1 = cv2.cvtColor(cv2.imread('./p2/ext/image1.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./p2/ext/image2.png'), cv2.COLOR_BGR2RGB)
F_21 = np.loadtxt('./p2/ext/F_21_test.txt') # Testing fundamental matrix.
F_estimated = F_21


def draw_epipolar_line(img2, F, pt1):
    """
    Draw a epipolar line in img2 corresponding to the point pt1 in image 1.
    """

    pt1_h = np.array([pt1[0], pt1[1], 1]).reshape(3, 1) # Convert to homogeneous coordinates.
    line = F @ pt1_h                                    # Get the epipolar line in img2.
    x0, x1 = 0, img2.shape[1]                           # Set the limits of the line.

    y0 = int(-line[2].item() / line[1].item())
    y1 = int(-(line[2].item() + line[0].item() * x1) / line[1].item())

    img2_with_line = img2.copy()                        # Drar the line.
    cv2.line(img2_with_line, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return img2_with_line


def onclick(event):
    """
    Callback function for the click event.
    """
    if event.inaxes is not None:
        pt1 = [int(event.xdata), int(event.ydata)]                              # Get click point coordinates.
        print(f"Point selected: {event.xdata}, {event.ydata}")                  # Print the selected point.
        img2_with_epipolar_line = draw_epipolar_line(img2, F_estimated, pt1)    # Draw the epipolar line in img2.
        plt.figure(figsize=(8, 6))                                              # Show the images with the epipolar line.
        plt.imshow(img2_with_epipolar_line)
        plt.title(f"Epipolar line in image 2 for the point {pt1} in image 1")
        plt.axis('off')
        plt.show()

#################### 2.2 Fundamental matrix definition ########################

# Ensamble T matrix.
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    Params:
        R_w_c: Rotation matrix.
        t_w_c: Translation vector.
    Returns:
        np.array: SE(3) matrix.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c
    
# Load the camera poses and the intrinsic matrix.
T_w_c1 = np.loadtxt('./p2/ext/T_w_c1.txt')
T_w_c2 = np.loadtxt('./p2/ext/T_w_c2.txt')
K_c = np.loadtxt('./p2/ext/K_c.txt')

# Get the rotation and translation matrices.
R_w_c1 = T_w_c1[:3, :3]
t_w_c1 = T_w_c1[:3, 3]
R_w_c2 = T_w_c2[:3, :3]
t_w_c2 = T_w_c2[:3, 3]

# Compute the relative rotation and translation matrices.
R_21 = R_w_c2.T @ R_w_c1

# Compute the relative translation vector.
T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1

# Get the rotation and translation matrices.
R_c2_c1 = T_c2_c1[:3, :3]
t_21 = T_c2_c1[:3, 3]

# Skew-symmetric translation matrix.
T_x = np.array([[0, -t_21[2], t_21[1]],
                [t_21[2], 0, -t_21[0]],
                [-t_21[1], t_21[0], 0]])

# Fundamental matrix estimation.
F_21 = np.linalg.inv(K_c.T) @ T_x @ R_21 @ np.linalg.inv(K_c)
F_estimated = F_21
np.savetxt('./p2/ext/F_21.txt', F_21)

#################### 2.3 Fundamental matrix linear estimation with eight point solution ########################

# Load the point correspondences.
x1 = np.loadtxt('./p2/ext/x1Data.txt')
x2 = np.loadtxt('./p2/ext/x2Data.txt')

# OpenCV implementation to estimate the fundamental matrix.
F_estimated, mask = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_8POINT)

#################### 2.4 Pose estimation from two views ########################

# Compute the essential matrix.
E_21 = K_c.T @ F_21 @ K_c

# OpenCV function to decompose the matrix.
R1, R2, t = cv2.decomposeEssentialMat(E_21)
t=t.ravel()

# Create the possible solutions for T_21.
T_21_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

# Get the correct solution among the four possible ones.
pts1 = x1.T
pts2 = x2.T
for i, (R, t) in enumerate(T_21_solutions):
    if is_valid_solution(R, t, K_c, pts1, pts2):
        print(f"Correct answer: R{(i//2)+1}, t{'+' if i % 2 == 0 else '-'}")
        R_correct = R
        t_correct = t
        break

# Transform and save the triangulated points.
X_3D = triangulate_points_from_cameras(R_correct, t_correct, K_c, pts1, pts2).T
X_3D = T_w_c1 @ np.vstack([X_3D, np.ones((1, X_3D.shape[1]))])
np.savetxt('./p2/ext/X_triangulated.txt', X_3D.T)


#################### 2.5 Results presentation ########################

# Load reference and triangulated points.
X_w_ref = np.loadtxt('./p2/ext/X_w.txt').T
X_w_triangulated = np.loadtxt('./p2/ext/X_triangulated.txt')



########################################################################
####################### LAB 4 2.1 two views ###########################
########################################################################

# Convertimos la rotación inicial T_wc2[:3, :3] a theta
theta_init = crossMatrixInv(logm(R_correct.astype('float64')))
# Obtenemos el vector de traslación inicial de T_wc2
t_init = t_correct

# Preparamos los parámetros iniciales (theta y t juntos)
T_init = np.hstack([theta_init, t_init])


# Ejecutamos el ajuste de bundle adjustment
R_opt, t_opt, X_opt = bundle_adjustment(x1, x2, K_c, T_init, (X_w_triangulated @ T_w_c1).T)

X_3D = triangulate_points_from_cameras(R_opt, t_opt, K_c, pts1, pts2).T
X_3D = T_w_c1 @ np.vstack([X_3D, np.ones((1, X_3D.shape[1]))])

print("initial_theta: " + str(R_correct))
print("optimized_theta: " + str(R_opt))
print("initial_t_21: " + str(t_correct))
print("optimized_t_21: " + str(t_opt))
# print("initial_X1: " + str(X_w_triangulated))
# print("optimized_X1: " + str(X_opt))

# Create the figure and system references.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pd.drawRefSystem(ax, np.eye(4), '-', 'W')   # World.
pd.drawRefSystem(ax, T_w_c1, '-', 'C1')     # Camera 1.
pd.drawRefSystem(ax, T_w_c2, '-', 'C2')     # Camera 2.

# Plot the points over the figure.
ax.scatter(X_w_ref[:, 0], X_w_ref[:, 1], X_w_ref[:, 2], c='r', label='Reference', marker='o')
ax.scatter(X_w_triangulated[:, 0], X_w_triangulated[:, 1], X_w_triangulated[:, 2], c='b', label='Triangulated', marker='^')
ax.scatter(X_3D.T[:, 0], X_3D.T[:, 1], X_3D.T[:, 2], c='g', label='Triangulated_opt', marker='^')

# Compute the euclidean distance between the reference and the triangulated points, then show the mean and median.
distances = np.linalg.norm(X_w_ref[:, :3] - X_w_triangulated[:, :3], axis=1)

# Show mean and median distances.
print ("Triangulation acurracy, compared to reference:")
print (f"Mean distance: {np.mean(distances)}")
print (f"Median distance: {np.median(distances)}")

# Axis labels, legend, and plot show.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

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

# Crear la matriz de transformación optimizada para la cámara C2
T_wc2_opt = np.eye(4)
T_wc2_opt[:3, :3] = R_opt
T_wc2_opt[:3, 3] = t_opt

scale_factor = np.linalg.norm(R_correct) / np.linalg.norm(R_opt)
scale_factor = np.linalg.norm(t_correct) / np.linalg.norm(t_opt)
t_opt_scaled = t_opt * scale_factor
print("Scale factor: ", scale_factor)
T_wc2_opt *= scale_factor

# Proyectar los puntos optimizados en cada imagen usando T_wc1, T_wc2_opt
x1_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c1) @ X_3D
x2_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2) @ X_3D
# x2_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_opt) @ X_3D

x1_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c1) @ X_w_ref.T
x2_p = K_c @ np.eye(3, 4) @ np.linalg.inv(T_w_c2) @ X_w_ref.T
# Normalizar las coordenadas para obtener las proyecciones en píxeles
x1_p_opt /= x1_p_opt[2, :]
x2_p_opt /= x2_p_opt[2, :]
x1_p /= x1_p[2, :]
x2_p /= x2_p[2, :]
    
    
# Imagen 1
plt.figure(4)
plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plotResidual(x1, x1_p, 'k-')  # Residuals originales
plotResidual(x1, x1_p_opt, 'k-')  # Residuals optimizado
plt.plot(x1_p[0, :], x1_p[1, :], 'bo', label='Original Projection')
plt.plot(x1_p_opt[0, :], x1_p_opt[1, :], 'go', label='Optimized Projection')  # Proyecciones optimizadas
plt.plot(x1[0, :], x1[1, :], 'rx')
plotNumberedImagePoints(x1[0:2, :], 'r', 4)
plt.legend()
plt.title('Image 1')

# Imagen 2
plt.figure(5)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plotResidual(x2, x2_p, 'k-')
plotResidual(x2, x2_p_opt, 'k-')  # Residuals optimizado
plt.plot(x2_p[0, :], x2_p[1, :], 'bo', label='Original Projection')
plt.plot(x2_p_opt[0, :], x2_p_opt[1, :], 'go', label='Optimized Projection')
plt.plot(x2[0, :], x2[1, :], 'rx')
plotNumberedImagePoints(x2[0:2, :], 'r', 4)
plt.legend()
plt.title('Image 2')
plt.show()



########################################################################
################### LAB 4 3.0 Perspective-N-Point #######################
########################################################################

x3 = np.loadtxt('./p2/ext/x3Data.txt')
# Convert the 3D points to (n, 1, 2) format as required by solvePnP
imagePoints1 = np.ascontiguousarray(x1[0:2, :].T).reshape((x1.shape[1], 1, 2))  # (nPoints, 1, 2)
imagePoints2 = np.ascontiguousarray(x2[0:2, :].T).reshape((x2.shape[1], 1, 2))  # (nPoints, 1, 2)
imagePoints3 = np.ascontiguousarray(x3[0:2, :].T).reshape((x3.shape[1], 1, 2))  # (nPoints, 1, 2)

# Now, apply the solvePnP to estimate the pose of the third camera with respect to the first
# We can use the triangulated points or reference points as object points
objectPoints = X_w_ref[:,:3] # 3D object points (nPoints, 3)
# objectPoints = (X_w_ref @ T_w_c1)[:,:3]  # 3D object points (nPoints, 3)

# Set distortion coefficients to zero (we have undistorted images)
distCoeffs = np.zeros(4)

print("objectPoints: " + str(objectPoints.shape))
print("imagePoints1: " + str(imagePoints1.shape))

# Solve PnP using the EPNP algorithm
retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints3, K_c, distCoeffs, flags=cv2.SOLVEPNP_EPNP)

print("retval: " + str(retval))
print("rvec: " + str(rvec.T))
print("tvec: " + str(tvec))

R_w_c3_pnp = expm(crossMatrix(rvec.T[0])).T
t_w_c3_pnp = tvec.ravel()
# T_w_c3_pnp = np.linalg.inv(T_w_c1) @ ensamble_T(R_w_c3_pnp, t_w_c3_pnp)
T_w_c3_pnp = np.eye(4)
T_w_c3_pnp[:3, :3] = R_w_c3_pnp
T_w_c3_pnp[:3, 3] = -(T_w_c1[:3, :3] @ t_w_c3_pnp)

T_w_c3 = np.loadtxt('./p2/ext/T_w_c3.txt')

# Create the figure and system references.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pd.drawRefSystem(ax, np.eye(4), '-', 'W')   # World.
pd.drawRefSystem(ax, T_w_c1, '-', 'C1')     # Camera 1.
pd.drawRefSystem(ax, T_w_c2, '-', 'C2')     # Camera 2.
pd.drawRefSystem(ax, T_w_c3, '-', 'C3 GT')     # Camera 3 GT.
pd.drawRefSystem(ax, T_w_c3_pnp, '-', 'C3 PNP')     # Camera 3 PNP.
plt.show()

########################################################################
###################### LAB 4 4.0 three views ##########################
########################################################################

