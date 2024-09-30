import numpy as np
import cv2

# Cargar datos
K_c = np.array([[458.654, 0, 367.215],
                [0, 457.296, 248.375],
                [0, 0, 1]])

T_w_c1 = np.loadtxt('./p2/ext/T_w_c1.txt')
T_w_c2 = np.loadtxt('./p2/ext/T_w_c2.txt')
x1 = np.loadtxt('./p2/ext/x1Data.txt')  # (2, N)
x2 = np.loadtxt('./p2/ext/x2Data.txt')  # (2, N)

# Obtener matrices de proyección P1 y P2
P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Matriz de proyección para cámara 1
P2 = K_c @ T_w_c2[:3, :]  # Matriz de proyección para cámara 2

# Triangulación de puntos
X_homogeneous = cv2.triangulatePoints(P1, P2, x1, x2)

# Convertir a coordenadas no homogéneas
X_3D = X_homogeneous[:3, :] / X_homogeneous[3, :]

# Guardar resultados
np.savetxt('./p2/ext/X_w.txt', X_3D.T)

# Visualización de los puntos 3D triangulados
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_3D[0, :], X_3D[1, :], X_3D[2, :], c='r', marker='o')
plt.show()
