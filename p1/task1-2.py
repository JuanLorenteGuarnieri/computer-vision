import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import os
import scipy.linalg as scAlg

sys.path.append(os.path.abspath('./p1/ext'))

from plotData_v15 import ensamble_T

# # Function to compute homogeneous projection
# def project_points(P, X):
#     X_h = np.hstack((X, np.ones((X.shape[0], 1))))
#     # x_h = P @ X_h  # Apply projection matrix
#     x = x_h[:2] / x_h[2]  # Convert to inhomogeneous (x, y) by dividing by the third coordinate
#     return x

def project_points(P, X):
    X_h = np.vstack((X, np.ones((1, X.shape[1]))))  # Convert to homogeneous coordinates
    x_h = P @ X_h  # Apply projection matrix
    x = x_h[:2] / x_h[2]  # Convert to inhomogeneous (x, y) by dividing by the third coordinate
    return x

def project_points_SVD(X):
    u, s, vh = np.linalg.svd(X.T)
    s[2] = 0  # If all the points are lying on the line s[2] = 0, therefore we impose it
    xProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    # xProjectedOnTheLine /= xProjectedOnTheLine[2, :]
    return xProjectedOnTheLine


# Function to compute the line equation from two points in homogeneous coordinates
def compute_line(point1, point2):
    return np.cross(point1, point2)

# Function to compute intersection of two lines
def compute_intersection(line1, line2):
    p = np.cross(line1, line2)
    return p / p[2]  # Convert from homogeneous to inhomogeneous

# Cargar los datos de los archivos .txt
K = np.loadtxt('p1/ext/K.txt')
R_w_c1 = np.loadtxt('p1/ext/R_w_c1.txt')
t_w_c1 = np.loadtxt('p1/ext/t_w_c1.txt')
R_w_c2 = np.loadtxt('p1/ext/R_w_c2.txt')
t_w_c2 = np.loadtxt('p1/ext/t_w_c2.txt')

T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
T_w_c2 = ensamble_T(R_w_c2, t_w_c2)

# Projection matrices P1 and P2
P1 = K @ T_w_c1[:3,:]
P2 = K @ T_w_c2[:3,:]

# Define the points 3D A, B, C, D
A = np.array([3.44, 0.80, 0.82])
B = np.array([4.20, 0.80, 0.82])
C = np.array([4.20, 0.60, 0.82])
D = np.array([3.55, 0.60, 0.82])

# Create the matrix of points in homogeneous coordinates (add a column of ones)
points = np.array([A, B, C, D])

points_3D = np.array([[3.44, 4.20, 4.20, 3.55, -0.01],
                      [0.80, 0.80, 0.60, 0.60, 2.60],
                      [0.82, 0.82, 0.82, 0.82, 1.21]])



# Project points to both images
proj_image1_SVD = project_points_SVD(points_3D)
proj_image2_SVD = project_points_SVD(points_3D)
proj_image1 = project_points(P1, points_3D)
proj_image2 = project_points(P2, points_3D)

proj_image1_homog = np.vstack((proj_image1, np.ones((1, proj_image1.shape[1]))))
proj_image2_homog = np.vstack((proj_image2, np.ones((1, proj_image2.shape[1]))))

# Compute the line l_ab and l_cd
l_ab = compute_line(proj_image1_homog[:, 0], proj_image1_homog[:, 1])  # Line from A to B
l_cd = compute_line(proj_image1_homog[:, 2], proj_image1_homog[:, 3])  # Line from C to D

# Compute the intersection point p_12 of the lines l_ab and l_cd
p_12 = compute_intersection(l_ab, l_cd)

# Compute the 3D infinite point AB_inf (direction of the line through A and B)
AB_inf = points_3D[:, 0] - points_3D[:, 1]  # Direction vector
AB_inf = np.append(AB_inf, 0)  # Infinite point, so append 0 to make it homogeneous

# Project the infinite point AB_inf to image 1 to get the vanishing point ab_inf
ab_inf = P1 @ AB_inf
ab_inf = ab_inf[:2] / ab_inf[2]  # Convert to inhomogeneous coordinates

# Plot the points, lines, and intersection in the image
def plot_image_with_lines_and_points(img_path, points, line1, line2, intersection, title):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    # Plot points
    plt.scatter(points[0, :], points[1, :], c='r', marker='+', s=100, label='Projected Points')

    # Plot lines
    x_vals = np.array([0, img.shape[1]])
    plt.plot(x_vals, -(line1[0] * x_vals + line1[2]) / line1[1], 'g', label='Line l_ab')
    plt.plot(x_vals, -(line2[0] * x_vals + line2[2]) / line2[1], 'b', label='Line l_cd')

    # Plot intersection
    plt.scatter(intersection[0], intersection[1], c='y', marker='x', s=200, label='Intersection p_12')

    # Plot vanishing point
    plt.scatter(ab_inf[0], ab_inf[1], c='m', marker='o', s=150, label='Vanishing Point ab_inf')

    plt.title(title)
    plt.legend()
    plt.show()

# Task 1. Projection matrices
print(proj_image1_SVD)
print(proj_image2_SVD)
# Task 2. 2D Lines and vanishing points
plot_image_with_lines_and_points("p1/ext/Image1.jpg", proj_image1, l_ab, l_cd, p_12, "Image 1 with Lines and Intersection")
plot_image_with_lines_and_points("p1/ext/Image2.jpg", proj_image2, l_ab, l_cd, p_12, "Image 2 with Lines and Intersection")