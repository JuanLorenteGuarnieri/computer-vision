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

# Show the first image and set the click event.
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Click on a point in image 1 to generate the epipolar line in image 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

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

# Check the epipolar lines with the same interface as 2.1.
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Click on a point in image 1 to generate the epipolar line in image 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


#################### 2.3 Fundamental matrix linear estimation with eight point solution ########################

# Load the point correspondences.
x1 = np.loadtxt('./p2/ext/x1Data.txt')
x2 = np.loadtxt('./p2/ext/x2Data.txt')

# OpenCV implementation to estimate the fundamental matrix.
F_estimated, mask = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_8POINT)

# Check the epipolar lines with the same interface as 2.1.
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Click on a point in image 1 to generate the epipolar line in image 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


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

# Create the figure and system references.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pd.drawRefSystem(ax, np.eye(4), '-', 'W')   # World.
pd.drawRefSystem(ax, T_w_c1, '-', 'C1')     # Camera 1.
pd.drawRefSystem(ax, T_w_c2, '-', 'C2')     # Camera 2.

# Plot the points over the figure.
ax.scatter(X_w_ref[:, 0], X_w_ref[:, 1], X_w_ref[:, 2], c='r', label='Reference', marker='o')
ax.scatter(X_w_triangulated[:, 0], X_w_triangulated[:, 1], X_w_triangulated[:, 2], c='b', label='Triangulated', marker='^')

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


#################### 3.1 Homography definition ########################

# Load the point correspondences and the plane parameters.
x1_floor = np.loadtxt('./p2/ext/x1FloorData.txt')[:2, :].T
x2_floor = np.loadtxt('./p2/ext/x2FloorData.txt')[:2, :].T
Pi_1 = np.loadtxt('./p2/ext/Pi_1.txt')
n = Pi_1[:3]
d = Pi_1[-1]

# Get the homography matrix.
H = K_c @ (R_c2_c1 - (t_21.reshape(3,1) @ n.reshape(1,3)) / d) @ np.linalg.inv(K_c)
print ("3.1 Homography matrix:")
print (H)


#################### 3.2 Point transfer visualization ########################

# Transfer points from image 1 to image 2 using the homography.
x1_homogeneous = np.vstack([x1, np.ones((1, x1.shape[1]))])
x2_estimated = H @ x1_homogeneous

# Convert to inhomogeneous coordinates.
x2_estimated /= x2_estimated[2, :]
pts_3_2 = visualize_point_transfer(H, img1, img2, x1_floor)

#################### 3.3 Homography linear estimation from matches ########################

# Estimate the homography
H_estimated = compute_homography(x1_floor, x2_floor)

# Visualize the point transfer
pts_3_3 = visualize_point_transfer(H_estimated, img1, img2, x1_floor)

# Compure distance between both results.
distances = np.linalg.norm(pts_3_2 - pts_3_3, axis=1)
print ("Estimated homography accuracy, comparing its results with the ones from the computed homography:")
print (f"Mean distance: {np.mean(distances)}")
print (f"Median distance: {np.median(distances)}")