import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath('./p2/ext'))

import plotData as pd

def obtain_proyection_matrices(K, R_correcta, t_correcta):
    """
    Calcula las matrices de proyección P1 y P2 a partir de la matriz intrínseca, rotación y traslación correctas.
    
    :param K: Matriz intrínseca de la cámara (3x3)
    :param R_correcta: Matriz de rotación correcta (3x3)
    :param t_correcta: Vector de traslación correcto (3x1)
    :return: P1 y P2, las matrices de proyección de las dos cámaras
    """
    # Primera matriz de proyección P1 asume que la cámara 1 está en el origen
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Segunda matriz de proyección P2 utiliza la rotación y traslación correctas
    P2 = K @ np.hstack((R_correcta, t_correcta.reshape(3, 1)))
    
    return P1, P2

def compute_homography(K1, K2, R, t, plane_normal, d):
    """
    Compute the homography from the floor plane.
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
    H: Homography matrix.
    img1, img2: The two images.
    pts1: Points in the first image to transfer.
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

def calculate_homography(pts_src, pts_dst):
    "Implementación de findHomografy"
    A = []
    
    # Creamos matriz como en transparencias
    for i in range(pts_src.shape[0]):
        x, y = pts_src[i, 0], pts_src[i, 1] 
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    
    # Obtenemos los valores propios
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    H = H / H[2, 2]     # Reescalamos

    return H

def normalize_points(pts):
    """
    Normaliza los puntos para que el centroide esté en el origen y la distancia promedio al origen sea sqrt(2).
    """
    # Calcular el centroide de los puntos
    centroid = np.mean(pts, axis=0)
    
    # Restar el centroide para trasladar los puntos al origen
    pts_shifted = pts - centroid
    
    # Calcular la distancia promedio de los puntos al origen
    avg_dist = np.mean(np.linalg.norm(pts_shifted, axis=1))
    
    # Escalar los puntos para que la distancia promedio sea sqrt(2)
    scale = np.sqrt(2) / avg_dist
    pts_normalized = pts_shifted * scale
    
    # Construir la matriz de transformación
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    return pts_normalized, T

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
    
    
    T_c1_w = np.linalg.inv(ensamble_T(R_w_c1, t_w_c1))[:3, :]
    T_c2_w = np.linalg.inv(ensamble_T(R_w_c2, t_w_c2))[:3, :]

    P1 = np.dot(K, T_c1_w)
    P2 =np.dot(K, T_c2_w) 
    
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
# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c
    
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

# T_c1_w = np.linalg.inv(ensamble_T(R_w_c1, t_w_c1))[:3, :]
# T_c2_w = np.linalg.inv(ensamble_T(R_w_c2, t_w_c2))[:3, :]

# Calcular rotación y traslación relativa
R_21 = R_w_c2.T @ R_w_c1

# t_21 = t_w_c2 - R_21 @ t_w_c1
T_c2_c1 = np.linalg.inv(T_w_c2) @ T_w_c1

R_c2_c1 = T_c2_c1[:3, :3]
t_21 = T_c2_c1[:3, 3]

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
#F_estimated = compute_fundamental_matrix(x1, x2)

# OpenCV implementation to estimate the fundamental matrix
F_estimated, mask = cv2.findFundamentalMat(x1.T, x2.T, cv2.FM_8POINT)

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
#R1, R2, t = decompose_essential_matrix(E_21)

# OpenCV function to decompose the matrix
R1, R2, t = cv2.decomposeEssentialMat(E_21)

# Generar las cuatro posibles soluciones para T_21
T_21_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

# Desambiguar la solución correcta triangulando puntos
# Podemos probar cada combinación (R, t) y seleccionar la que genere puntos 3D delante de ambas cámaras.

# Correspondencias de puntos en ambas imágenes (pts1 y pts2)
# Estos son los puntos que has detectado en las dos imágenes.
pts1 = x1.T
pts2 = x2.T

# Desambiguar la solución correcta
for i, (R, t) in enumerate(T_21_solutions):
    if is_valid_solution(R, t, K_c, pts1, pts2):
        print(f"Solución correcta: R{i + 1}, t{'+' if i % 2 == 0 else '-'}")
        R_correcta = R
        t_correcta = t
        break

# Save the correct points, transforming them to 3D, into the file.

R_2 = R_w_c1 @ R_correcta
t_2 = (t_w_c1 + R_w_c1 @ t_correcta).reshape(3,1)
P1 = K_c @ np.linalg.inv(T_w_c1)[:3, :]
T_w_c2 = pd.ensamble_T(R_2, t_2)
P2 = K_c @ T_w_c2[:3, :]

# P1, P2 = obtain_proyection_matrices(K_c, R_correcta, t_correcta)
X_hom = cv2.triangulatePoints(P1, P2, x1, x2)
X_3D = X_hom[:3, :] / X_hom[3, :]
np.savetxt('./p2/ext/X_triangulated.txt', X_3D.T)
    
#################### 2.5 Results presentation ########################

# # Cargar los puntos 3D de referencia
X_w_ref = np.loadtxt('./p2/ext/X_w.txt')  # Puntos 3D de referencia

# # Cargar los puntos 3D obtenidos mediante triangulación
X_w_triangulated = np.loadtxt('./p2/ext/X_triangulated.txt')  # Puntos 3D triangulados por nosotros

# # Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Dibujar los sistemas de referencia de las cámaras
plotData.drawRefSystem(ax, np.eye(4), '-', 'W')  # Sistema de referencia mundial
plotData.drawRefSystem(ax, T_w_c1, '-', 'C1')  # Cámara 1
plotData.drawRefSystem(ax, T_w_c2, '-', 'C2')  # Cámara 2

# # Dibujar los puntos 3D de referencia
ax.scatter(X_w_ref[:, 0], X_w_ref[:, 1], X_w_ref[:, 2], c='r', label='Referencia', marker='o')

# # Dibujar los puntos 3D triangulados
ax.scatter(X_w_triangulated[:, 0], X_w_triangulated[:, 1], X_w_triangulated[:, 2], c='b', label='Triangulados', marker='^')

# # Etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# # Mostrar leyenda
ax.legend()

# # Mostrar la gráfica
plt.show()

#################### 3.1 Homography definition ########################

# Load the point correspondences
x1_floor = np.loadtxt('./p2/ext/x1FloorData.txt')[:2, :].T
x2_floor = np.loadtxt('./p2/ext/x2FloorData.txt')[:2, :].T

# Cargar la normal del plano y la distancia
Pi_1 = np.loadtxt('./p2/ext/Pi_1.txt')
n = Pi_1[:3]
d = Pi_1[-1]

# Calcular la homografía
H = K_c @ (R_c2_c1 - (t_21.reshape(3,1) @ n.reshape(1,3)) / d) @ np.linalg.inv(K_c)
print(H)

#################### 3.2 Point transfer visualization ########################

# Transferir puntos desde la imagen 1 a la imagen 2 usando la homografía
x1_homogeneous = np.vstack([x1, np.ones((1, x1.shape[1]))])
x2_estimated = H @ x1_homogeneous

# Convertir a coordenadas inhomogéneas
x2_estimated /= x2_estimated[2, :]
visualize_point_transfer(H, img1, img2, x1_floor)

#################### 3.3 Homography linear estimation from matches ########################

# Estimate the homography
H_estimated = calculate_homography(x1_floor, x2_floor)

# Visualize the point transfer
visualize_point_transfer(H_estimated, img1, img2, x1_floor)

