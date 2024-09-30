import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('./p2/ext'))

import plotData


def normalize_points(pts):
    """
    Normaliza los puntos para que el centroide esté en el origen y la distancia promedio al origen sea sqrt(2).
    """
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    
    pts_h = np.column_stack((pts, np.ones(pts.shape[0])))
    pts_normalized = (T @ pts_h.T).T
    return pts_normalized[:, :2], T

def compute_fundamental_matrix(x1, x2):
    """
    Estima la matriz fundamental usando el método de los 8 puntos.
    x1, x2: arrays de tamaño Nx2 con las coordenadas de los puntos en ambas imágenes.
    """
    # Normalizar los puntos
    x1_normalized, T1 = normalize_points(x1)
    x2_normalized, T2 = normalize_points(x2)
    
    # Construir la matriz A
    N = x1_normalized.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1x = x1_normalized[i, 0]
        y1x = x1_normalized[i, 1]
        x2x = x2_normalized[i, 0]
        y2x = x2_normalized[i, 1]
        A[i] = [x1x*x2x, x1x*y2x, x1x, y1x*x2x, y1x*y2x, y1x, x2x, y2x, 1]
    
    # Resolver Af = 0 usando SVD
    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)
    
    # Imponer la condición de rango 2
    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0  # Forzar el tercer valor singular a ser 0
    F_normalized = U @ np.diag(S) @ Vt
    
    # Desnormalizar la matriz fundamental
    F = T2.T @ F_normalized @ T1
    
    return F / F[2, 2]  # Normalizar para que el último elemento sea 1


def decompose_essential_matrix(E):
    """
    Descompone la matriz esencial E en dos posibles matrices de rotación (R1, R2)
    y un vector de traslación t.
    
    E: Matriz esencial (3x3).
    
    Retorna:
        R1, R2: Dos posibles matrices de rotación (3x3).
        t: Vector de traslación (3x1).
    """
    # SVD de la matriz esencial
    U, S, Vt = np.linalg.svd(E)
    
    # Asegurarse de que E tenga dos valores singulares iguales y uno cercano a cero
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    # Matriz auxiliar W (para crear las matrices de rotación)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    # Posibles soluciones para la rotación
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # La traslación es el tercer vector de U
    t = U[:, 2]
    
    # Asegurar que R1 y R2 sean rotaciones válidas (determinante = +1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return R1, R2, t


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangula puntos 3D a partir de dos vistas usando las matrices de proyección de las cámaras.
    P1, P2: Matrices de proyección 3x4 de las dos cámaras.
    pts1, pts2: Correspondencias de puntos en las dos imágenes.
    """
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)[:, :2]
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)[:, :2]

    pts_4d_h = cv2.triangulatePoints(P1, P2, pts1_h.T, pts2_h.T)
    pts_3d = pts_4d_h[:3] / pts_4d_h[3]  # Convertir de coordenadas homogéneas a 3D
    return pts_3d.T

def is_valid_solution(R, t, K, pts1, pts2):
    """
    Verifica si una solución (R, t) genera puntos 3D válidos (delante de ambas cámaras).
    """
    # Matriz de proyección de la primera cámara (cámara de referencia)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Matriz de proyección de la segunda cámara para la solución dada
    P2 = K @ np.hstack((R, t.reshape(-1, 1)))
    
    # Triangular puntos
    pts_3d = triangulate_points(P1, P2, pts1, pts2)
    
    # Verificar si los puntos triangulados están delante de ambas cámaras (coordenada Z positiva)
    pts_cam1 = pts_3d[:, 2]  # Coordenada Z en la primera cámara
    pts_cam2 = (R @ pts_3d.T + t.reshape(-1, 1))[2, :]  # Coordenada Z en la segunda cámara
    
    # Los puntos son válidos si están delante de ambas cámaras
    return np.all(pts_cam1 > 0) and np.all(pts_cam2 > 0)

#################### 2.1 Epipolar lines visualization ########################
# Cargar imágenes y matriz fundamental
img1 = cv2.cvtColor(cv2.imread('./p2/ext/image1.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('./p2/ext/image2.png'), cv2.COLOR_BGR2RGB)
F_21 = np.loadtxt('./p2/ext/F_21_test.txt')  # Matriz fundamental de prueba
F_estimated = F_21

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


# Función de callback para el clic del mouse
def onclick(event):
    if event.inaxes is not None:
        pt1 = [int(event.xdata), int(event.ydata)]  # Obtener coordenadas del clic
        print(f"Punto seleccionado en imagen 1: {pt1}")
        
        # Dibujar línea epipolar en la imagen 2
        img2_with_epipolar_line = draw_epipolar_line(img2, F_estimated, pt1)
        
        # Mostrar la imagen 2 con la línea epipolar
        plt.figure(figsize=(8, 6))
        plt.imshow(img2_with_epipolar_line)
        plt.title(f"Línea epipolar en imagen 2 para el punto {pt1} en imagen 1")
        plt.axis('off')
        plt.show()

# Mostrar la imagen 1 y esperar al clic del usuario
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Haz clic en un punto de la imagen 1 para generar la línea epipolar en la imagen 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


#################### 2.2 Fundamental matrix definition ########################
# Cargar las matrices de transformación
T_w_c1 = np.loadtxt('./p2/ext/T_w_c1.txt')
T_w_c2 = np.loadtxt('./p2/ext/T_w_c2.txt')

# Cargar la matriz intrínseca
K_c = np.loadtxt('./p2/ext/K_c.txt')

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
F_estimated = F_21

# Guardar la matriz fundamental
np.savetxt('./p2/ext/F_21.txt', F_21)

# Visualizar las líneas epipolares usando la matriz estimada
# (similar al paso 2.1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Haz clic en un punto de la imagen 1 para generar la línea epipolar en la imagen 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

#################### 2.3 Fundamental matrix linear estimation with eight point solution ########################

# Cargar las correspondencias de puntos
x1 = np.loadtxt('./p2/ext/x1Data.txt')
x2 = np.loadtxt('./p2/ext/x2Data.txt')

# Estimate the fundamental matrix using the method of the eight points
F_estimated = compute_fundamental_matrix(x1, x2)

# OpenCV implementation to estimate the fundamental matrix
# F_estimated, mask = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_8POINT)

# Visualizar las líneas epipolares usando la matriz estimada
# (similar al paso 2.1)
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img1)
ax.set_title("Haz clic en un punto de la imagen 1 para generar la línea epipolar en la imagen 2")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

#################### 2.4 Pose estimation from two views ########################

# Cálculo de la matriz esencial
E_21 = K_c.T @ F_21 @ K_c

# Descomponer la matriz esencial en R y t
R1, R2, t = decompose_essential_matrix(E_21)

# OpenCV function to decompose the matrix
# _, R1, R2, t = cv2.decomposeEssentialMat(E_21)

# Generar las cuatro posibles soluciones para T_21
T_21_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

# Desambiguar la solución correcta triangulando puntos
# Podemos probar cada combinación (R, t) y seleccionar la que genere puntos 3D delante de ambas cámaras.

# Correspondencias de puntos en ambas imágenes (pts1 y pts2)
# Estos son los puntos que has detectado en las dos imágenes.
pts1 = np.array([...])  # Puntos en la imagen 1
pts2 = np.array([...])  # Puntos en la imagen 2

# Desambiguar la solución correcta
for i, (R, t) in enumerate(T_21_solutions):
    if is_valid_solution(R, t, K_c, pts1, pts2):
        print(f"Solución correcta: R{i + 1}, t{'+' if i % 2 == 0 else '-'}")
        R_correcta = R
        t_correcta = t
        break
    
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


