import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('./p2/ext'))

import plotData

#################### 2.1 Epipolar lines visualization ########################
# Cargar imágenes y matriz fundamental
img1 = cv2.cvtColor(cv2.imread('./p2/ext/image1.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./p2/ext/image2.png'), cv2.COLOR_BGR2RGB)
F_21 = np.loadtxt('./p2/ext/F_21_test.txt')  # Matriz fundamental de prueba

# Definir función para dibujar líneas epipolares en la segunda imagen
def draw_epipolar_line(img2, F, pt1):
    """
    Dibuja una línea epipolar en img2 correspondiente al punto pt1 en la imagen 1.
    """
    # pt1 en coordenadas homogéneas
    pt1_h = np.array([pt1[0], pt1[1], 1]).reshape(3, 1)
    
    # Obtener la línea epipolar en img2
    line = F @ pt1_h
    
    # Definir extremos de la línea
    x0, x1 = 0, img2.shape[1]
    
    y0 = int(-line[2].item() / line[1].item())
    y1 = int(-(line[2].item() + line[0].item() * x1) / line[1].item())

    
    # Dibujar línea
    img2_with_line = img2.copy()
    cv2.line(img2_with_line, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return img2_with_line

# Punto en imagen 1 seleccionado manualmente (o clickeado)
pt1 = [250, 300]  # Cambiar esto por el punto que elijas

# Dibujar la línea epipolar en la segunda imagen
img2_with_epipolar_line = draw_epipolar_line(img2, F_21, pt1)

# Mostrar la imagen con la línea epipolar
plt.imshow(img2_with_epipolar_line)
plt.title('Línea epipolar en la segunda imagen')
plt.show()

#################### 2.2 Fundamental matrix definition ########################
# Cargar las matrices de transformación
T_w_c1 = np.loadtxt('./p2/ext/T_w_c1.txt')
T_w_c2 = np.loadtxt('./p2/ext/T_w_c2.txt')

# Cargar la matriz intrínseca
K_c = np.array([[458.654, 0, 367.215],
                [0, 457.296, 248.375],
                [0, 0, 1]])

# Obtener las rotaciones y traslaciones
R_w_c1 = T_w_c1[:3, :3]
t_w_c1 = T_w_c1[:3, 3]

R_w_c2 = T_w_c2[:3, :3]
t_w_c2 = T_w_c2[:3, 3]

# Calcular rotación y traslación relativa
R_21 = R_w_c2 @ R_w_c1.T
t_21 = t_w_c2 - R_21 @ t_w_c1

# Matriz antisimétrica de la traslación
T_x = np.array([[0, -t_21[2], t_21[1]],
                [t_21[2], 0, -t_21[0]],
                [-t_21[1], t_21[0], 0]])

# Calcular la matriz fundamental
F_21 = np.linalg.inv(K_c.T) @ T_x @ R_21 @ np.linalg.inv(K_c)

# Guardar la matriz fundamental
np.savetxt('./p2/ext/F_21.txt', F_21)

#################### 2.3 Fundamental matrix linear estimation with eight point solution ########################

# Cargar las correspondencias de puntos
x1 = np.loadtxt('./p2/ext/x1Data.txt')
x2 = np.loadtxt('./p2/ext/x2Data.txt')

# Estimar la matriz fundamental usando el método de los ocho puntos
F_estimated, mask = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_8POINT)

# Visualizar las líneas epipolares usando la matriz estimada
# (similar al paso 2.1)
# Punto en imagen 1 seleccionado manualmente (o clickeado)
pt1 = [250, 300]  # Cambiar esto por el punto que elijas

# Dibujar la línea epipolar en la segunda imagen
img2_with_epipolar_line = draw_epipolar_line(img2, F_estimated, pt1)

# Mostrar la imagen con la línea epipolar
plt.imshow(img2_with_epipolar_line)
plt.title('Línea epipolar en la segunda imagen')
plt.show()

#################### 2.4 Pose estimation from two views ########################

# Cálculo de la matriz esencial
E_21 = K_c.T @ F_21 @ K_c

# Descomponer la matriz esencial en R y t
_, R1, R2, t = cv2.decomposeEssentialMat(E_21)

# Generar las cuatro posibles soluciones para T_21
T_21_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

# Desambiguar la solución correcta triangulando puntos
# Podemos probar cada combinación (R, t) y seleccionar la que genere puntos 3D delante de ambas cámaras.

#################### 2.5 Results presentation ########################

# Cargar los puntos 3D de referencia
X_w_ref = np.loadtxt('./p2/ext/X_w.txt')  # Puntos 3D de referencia

# Cargar los puntos 3D obtenidos mediante triangulación
X_w_triangulated = np.loadtxt('./p2/ext/X_triangulated.txt')  # Puntos 3D triangulados por nosotros

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar los sistemas de referencia de las cámaras
plotData.drawRefSystem(ax, np.eye(4), '-', 'W')  # Sistema de referencia mundial
plotData.drawRefSystem(ax, T_w_c1, '-', 'C1')  # Cámara 1
plotData.drawRefSystem(ax, T_w_c2, '-', 'C2')  # Cámara 2

# Dibujar los puntos 3D de referencia
ax.scatter(X_w_ref[:, 0], X_w_ref[:, 1], X_w_ref[:, 2], c='r', label='Referencia', marker='o')

# Dibujar los puntos 3D triangulados
ax.scatter(X_w_triangulated[:, 0], X_w_triangulated[:, 1], X_w_triangulated[:, 2], c='b', label='Triangulados', marker='^')

# Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar leyenda
ax.legend()

# Mostrar la gráfica
plt.show()

#################### 3.1 Homography definition ########################

# Cargar la normal del plano y la distancia
n = np.array([0.0149, 0.9483, 0.3171])
d = -1.7257

# Calcular la homografía
H = K_c @ (R_21 - (t_21 @ n.T) / d) @ np.linalg.inv(K_c)

#################### 3.2 Point transfer visualization ########################

# Transferir puntos desde la imagen 1 a la imagen 2 usando la homografía
x1_homogeneous = np.vstack([x1, np.ones((1, x1.shape[1]))])
x2_estimated = H @ x1_homogeneous

# Convertir a coordenadas inhomogéneas
x2_estimated /= x2_estimated[2, :]

#################### 3.3 Homography linear estimation from matches ########################

# Cargar puntos de correspondencia en el suelo
x1_floor = np.loadtxt('./p2/ext/x1FloorData.txt')
x2_floor = np.loadtxt('./p2/ext/x2FloorData.txt')

# Estimar la homografía
H_estimated, _ = cv2.findHomography(x1_floor.T, x2_floor.T)

# Comparar la homografía estimada con la calculada previamente


