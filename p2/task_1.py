# Lorente Guarnieri, Juan (NIP 816020)
# Bielsa Uche, Jaime (NIP 819033)
# File: task_1.py
# Date: October 13th, 2024
# Master in Graphics, Robotics and Computer Vision, Universidad de Zaragoza.
# Subject: Compuer Vision
# Description: task 1 of the second practice of the subject.


import numpy as np
import cv2
import plotData as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data.
K_c = np.loadtxt('./p2/ext/K_c.txt')
T_w_c1 = np.loadtxt('./p2/ext/T_w_c1.txt')
T_w_c2 = np.loadtxt('./p2/ext/T_w_c2.txt')
x1 = np.loadtxt('./p2/ext/x1Data.txt')
x2 = np.loadtxt('./p2/ext/x2Data.txt')

# Get transformation matrices from world to camera (inverting the known ones).
T_c1_w = np.linalg.inv(T_w_c1)
T_c2_w = np.linalg.inv(T_w_c2)

# Compute projection matrices P1 and P2.
P1 = K_c @ T_c1_w[:3, :]
P2 = K_c @ T_c2_w[:3, :]

# Triangulate points, convert to non-homogeneous coordinates and save them.
x_hom = cv2.triangulatePoints(P1, P2, x1, x2)
X_3D = x_hom[:3, :] / x_hom[3, :]
np.savetxt('./p2/ext/X_w_triangulated.txt', X_3D.T)

# Visualize triangulated points. Define the figure and the 3D plot.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot reference systems and points.
pd.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
pd.drawRefSystem(ax, T_w_c1, '-', 'C1')
pd.drawRefSystem(ax, T_w_c2, '-', 'C2')
ax.scatter(X_3D[0, :], X_3D[1, :], X_3D[2, :], c='m', marker='.')

# Plot the results.
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_zlim([0, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
