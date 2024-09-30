import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import scipy.linalg as scAlg
import plotData_v15 as pd
sys.path.append(os.path.abspath('./p1/ext'))

def inhom2hom (x, multiplier=1):
    """
    Convert inhomogeneous coordinates to homogeneous coordinates, using a
    multiplier for the homogeneous coordinate.
    Parameters:
        x: inhomogeneous coordinates.
        multiplier: multiplier for the homogeneous coordinate.
    Returns:
        Homogeneous coordinates.
    """
    return np.vstack((x, np.ones((1, x.shape[1]))*multiplier))

def hom2inhom (x):
    """
    Convert homogeneous coordinates to inhomogeneous coordinates.
    Parameters:
        x: homogeneous coordinates.
    Returns:
        Inhomogeneous coordinates.
    """
    sz = x.shape[0]
    return x[:sz-1] / x[sz-1]

def compute_projection_matrix(K, Rt):
    """
    Compute the projection matrix P = K [R|t].
    Parameters:
        K: intrinsic camera matrix.
        Rt: extrinsic camera matrix.
    Returns:
        Projection matrix P.
    """
    return np.dot(K, Rt)

def project_points_hom(P, X, lbda = 1):
    """
    Project points in homogeneous coordinates.
    Parameters:
        P: projection matrix.
        X: 3D points in homogeneous coordinates.
        lbda: scale factor.
    Returns:
        Projected points in homogeneous coordinates.
    """
    return np.dot(P, X) * lbda

def project_points_inhom(P, X, lbda = 1, multiplier=1):
    """
    Project points in inhomogeneous coordinates.
    Parameters:
        P: projection matrix.
        X: 3D points in inhomogeneous coordinates.
        lbda: scale factor.
    Returns:
        Projected points in inhomogeneous coordinates.
    """
    sol = project_points_hom(P, inhom2hom(X, multiplier = 1), lbda)
    return hom2inhom(sol)

def project_points_v0(P, X, lbda = 1):
    X_h = np.vstack((X, np.ones((1, X.shape[1]))))  # Convert to homogeneous coordinates
    x_h = (P @ X_h) * lbda  # Apply projection matrix
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
    """
    Compute the line equation from two points in homogeneous coordinates.
    Parameters:
        point1: first point in homogeneous coordinates.
        point2: second point in homogeneous coordinates.
    Returns:
        Line equation coefficients.
    """
    return np.cross(point1, point2)

# Function to compute intersection of two lines
def compute_intersection(line1, line2):
    """
    Compute the intersection of two lines.
    Parameters:
        line1: first line equation coefficients.
        line2: second line equation coefficients.
    Returns:
        Intersection point.
    """
    p = np.cross(line1, line2)
    return p / p[2]  # Convert from homogeneous to inhomogeneous

# Load the txt files data.
K = np.loadtxt('ext/K.txt')
R_w_c1 = np.loadtxt('ext/R_w_c1.txt')
t_w_c1 = np.loadtxt('ext/t_w_c1.txt')
R_w_c2 = np.loadtxt('ext/R_w_c2.txt')
t_w_c2 = np.loadtxt('ext/t_w_c2.txt')

# Compute the transformation matrices T_c1_w and T_c2_w, inverting the given
# rotation and translation matrices.
T_c1_w = np.linalg.inv(pd.ensamble_T(R_w_c1, t_w_c1))[:3, :]
T_c2_w = np.linalg.inv(pd.ensamble_T(R_w_c2, t_w_c2))[:3, :]

# Get the projection matrices P1 and P2.
P1 = compute_projection_matrix(K, T_c1_w)
P2 = compute_projection_matrix(K, T_c2_w)

# Define the homogeneous 3D points A - E.
points_AE = np.array([[3.44, 4.20, 4.20, 3.55, -0.01],
                      [0.80, 0.80, 0.60, 0.60, 2.60],
                      [0.82, 0.82, 0.82, 0.82, 1.21]])

# Project points to both images.
proj_image1 = project_points_inhom(P1, points_AE)
proj_image2 = project_points_inhom(P2, points_AE)

# Convert the projected points to homogeneous coordinates in 2D.
proj_image1_homog = np.vstack((proj_image1, np.ones((1, proj_image1.shape[1]))))
proj_image2_homog = np.vstack((proj_image2, np.ones((1, proj_image2.shape[1]))))

# Compute the lines l_ab and l_cd
l_ab = np.array([compute_line(proj_image1_homog[:, 0], proj_image1_homog[:, 1]),
                 compute_line(proj_image2_homog[:, 0], proj_image2_homog[:, 1])])
l_cd = np.array([compute_line(proj_image1_homog[:, 2], proj_image1_homog[:, 3]),
                 compute_line(proj_image2_homog[:, 2], proj_image2_homog[:, 3])])

# Compute the intersection point p_12 of the lines l_ab and l_cd.
p_12 = np.array([compute_intersection(l_ab[0], l_cd[0]),
                 compute_intersection(l_ab[1], l_cd[1])])

# Compute the 3D infinite point AB_inf (direction of the line through A and B).
AB_inf = np.array([points_AE[:, 1] - points_AE[:, 0]]).T    # Direction vector.
AB_inf = np.vstack([AB_inf, [0]])                           # Infinite point, so append 0 to make it homogeneous.

# Project the infinite point AB_inf to each image to get the vanishing point.
AB_inf_pts = np.hstack([project_points_hom(P1, AB_inf), project_points_hom(P2, AB_inf)])
AB_inf_pts = hom2inhom(AB_inf_pts)

# Plot the points, lines, and intersection in the image.
def plot_image_with_lines_and_points(img_path, points, line1, line2, intersection, ab_inf, title):

    # Image.
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

    # Points.
    plt.scatter(points[0, :], points[1, :], c='r', marker='+', s=100, label='Projected Points')
    pd.plotLabeledImagePoints(points, ['a','b','c','d','e'], 'r', (20, -20))

    # Lines.
    x_vals = np.array([np.min(points[0, :]), np.max(points[0, :])])
    plt.plot(x_vals, -(line1[0] * x_vals + line1[2]) / line1[1], 'g', label='Line l_ab')
    plt.plot(x_vals, -(line2[0] * x_vals + line2[2]) / line2[1], 'b', label='Line l_cd')

    # Intersection and vanishing point.
    plt.scatter(intersection[0], intersection[1], c='y', marker='x', s=200, label='Intersection p_12')
    plt.text(intersection[0]+20, intersection[1]-20, 'p_12', color='y')
    plt.scatter(ab_inf[0], ab_inf[1], c='m', marker='o', s=50, label='Vanishing Point ab_inf')
    plt.text(ab_inf[0]+20, ab_inf[1]-20, 'ab_inf', color='m')

    # Title and window.
    plt.title(title)
    plt.show()

# Task 1.
np.set_printoptions(precision=4,linewidth=1024,suppress=True)
print ("TASK 1")
print ("Projection matrix for image 1")
print (P1)
print ("Projection matrix for image 2")
print (P2)
print ("Points projected in image 1")
print (proj_image1)
print ("Points projected in image 2")
print (proj_image2)
print ()

# Task 2.
print ("TASK 2")
print ("Lines l_ab and in images 1 and 2")
print (l_ab)
print ("Lines l_cd and in images 1 and 2")
print (l_cd)
print ("Intersection point p_12 in images 1 and 2")
print (p_12)
print ("3D infinite point AB_inf")
print (AB_inf)
print ("Vanishing point ab_inf in images 1 and 2")
print (AB_inf_pts)
plot_image_with_lines_and_points("ext/Image1.jpg", proj_image1, l_ab[0], l_cd[0], p_12[0], AB_inf_pts[:, 0], "Image 1 with lines and intersection")
plot_image_with_lines_and_points("ext/Image2.jpg", proj_image2, l_ab[1], l_cd[1], p_12[1], AB_inf_pts[:, 1], "Image 2 with lines and intersection")