import numpy as np
import plotData as pd
import matplotlib.pyplot as plt
import utils as utils

def main():
    # Load calibration data (using the provided text files)

    K1_file = './p5/ext/K_1.txt'
    D1_file = './p5/ext/D1_k_array.txt'
    K2_file = './p5/ext/K_2.txt'
    D2_file = './p5/ext/D2_k_array.txt'

    K1, D1 = utils.load_calibration_data(K1_file, D1_file)
    K2, D2 = utils.load_calibration_data(K2_file, D2_file)
    T_wc1 = utils.load_transformation_matrix('./p5/ext/T_wc1.txt')
    T_wc2 = utils.load_transformation_matrix('./p5/ext/T_wc2.txt')

    # Load point matches from the provided files (x1.txt, x2.txt, x3.txt, x4.txt)
    x1 = utils.load_points('./p5/ext/x1.txt')
    x2 = utils.load_points('./p5/ext/x2.txt')
    x3 = utils.load_points('./p5/ext/x3.txt')
    x4 = utils.load_points('./p5/ext/x4.txt')
    
    
    T_wAwB_gt = np.loadtxt('./p5/ext/T_wAwB_gt.txt')
    T_wAwB_seed = np.loadtxt('./p5/ext/T_wAwB_seed.txt')
    print ('T_wAwB_gt.shape: ', T_wAwB_gt.shape)
    print ('T_wAwB_seed.shape: ', T_wAwB_seed.shape)
    print ('D1_k.shape: ', D1.shape)
    print ('D2_k.shape: ', D2.shape)
    print ('K_1.shape: ', K1.shape)
    print ('K_2.shape: ', K2.shape)
    print ('3d coordinates.shape: ', x1.shape)

    # Assuming that the virtual 3D points are provided for verification
    X1 = np.array([3, 2, 10])  # 3D point X1
    X2 = np.array([-5, 6, 7])  # 3D point X2
    X3 = np.array([1, 5, 14])  # 3D point X3

    # Project the 3D points using the Kannala-Brandt model for Camera 1 (K1, D1)
    u1_proj = utils.project_kannala_brandt(X1, K1, D1)
    u2_proj = utils.project_kannala_brandt(X2, K1, D1)
    u3_proj = utils.project_kannala_brandt(X3, K1, D1)

    # Print the results
    print("Projected u1:", u1_proj)
    print("Projected u2:", u2_proj)
    print("Projected u3:", u3_proj)

    # Optionally, calculate the difference between projected and expected results if you have them
    # Assuming the ground truth 2D points are known (you can replace them with the actual values)
    u1_expected = np.array([503.387, 450.1594])
    u2_expected = np.array([267.9465, 580.4671])
    u3_expected = np.array([441.0609, 493.0671])

    diff_u1 = np.linalg.norm(u1_proj[:2] - u1_expected)
    diff_u2 = np.linalg.norm(u2_proj[:2] - u2_expected)
    diff_u3 = np.linalg.norm(u3_proj[:2] - u3_expected)

    print("Difference for u1:", diff_u1)
    print("Difference for u2:", diff_u2)
    print("Difference for u3:", diff_u3)


    # Unproject the points back to 3D space
    X1_unproj = utils.unproject_kannala_brandt(u1_proj, K1, D1)
    X2_unproj = utils.unproject_kannala_brandt(u2_proj, K1, D1)
    X3_unproj = utils.unproject_kannala_brandt(u3_proj, K1, D1)
    #Normalize the X1 and X2 numpy vectors
    x1_normalized = X1 / np.linalg.norm(X1)
    x2_normalized = X2 / np.linalg.norm(X2)
    x3_normalized = X3 / np.linalg.norm(X3)
    # X1_unproj /= np.linalg.norm(X1_unproj)
    # X2_unproj /= np.linalg.norm(X2_unproj)
    # X3_unproj /= np.linalg.norm(X3_unproj)
    
    print("norm X1:", x1_normalized)
    print("norm X2:", x2_normalized)
    print("norm X3:", x3_normalized)

    # Print the unprojected points (back to 3D space)
    print("Unprojected X1:", X1_unproj)
    print("Unprojected X2:", X2_unproj)
    print("Unprojected X3:", X3_unproj)

    # Compare the unprojected 3D points with the original ones
    print("Difference for X1:", np.linalg.norm(X1_unproj - x1_normalized))
    print("Difference for X2:", np.linalg.norm(X2_unproj - x2_normalized))
    print("Difference for X3:", np.linalg.norm(X3_unproj - x3_normalized))

    # Perform triangulation for each pair of points
    # X = triangulate(x1, x2, K1, K2, T_wc1, T_wc2)

    # Print the triangulated 3D point
    # print("Triangulated 3D point:", X)

    # Triangulate 3D points
    points_3d = utils.triangulate_points(x1, x2, K1, D1, K2, D2, T_wc1, T_wc2)
    # points_3d = triangulate(x1, x2, K1, K2, T_wc1, T_wc2)

    # print("3D points:\n", points_3d)
    
    # Create the figure and system references and plot the 3D points.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    utils.plot_stereo_camera_axis(ax, np.eye(4), T_wc1, T_wc2, 'R0')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='g', label='Computed', marker='x')
    pd.plotNumbered3DPoints(ax, points_3d.T, 'r', (0, 0, 0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()

    from scipy.linalg import expm, logm
    
    # Convert initial rotation to theta
    theta_init = utils.crossMatrixInv(logm(T_wAwB_seed[:3, :3].astype('float64')))
    t_init = T_wAwB_seed[:3, 3]

    # Prepare initial pose parameters
    T_init = np.hstack([theta_init, t_init])

    # Run bundle adjustment for fisheye
    T_opt, X_opt = utils.bundle_adjustment_fish_eye(
        x1, x2, x3, x4, K1, K2, D1, D2, T_wc1, T_wc2, T_init, points_3d
    )

    # Update 3D points with optimized results
    # X_opt_2 = utils.triangulate_points(x1, x2, K1, D1, K2, D2, T_wc1, T_wc2)
    # X_opt = triangulate_points_from_cameras(R_opt, t_opt, K1, x1.T, x2.T).T
    # X_opt_2 = (T_wc1 @ np.vstack([X_opt, np.ones((1, X_opt.shape[1]))])).T

    print("initial_T_wAwB_seed: " + str(T_wc2[:3,3:4]))
    print("optimized_T_wAwB_seed: " + str(T_opt))
    # print("initial_X1: " + str(X_w.T))
    # print("optimized_X1: " + str(X_opt))

    
    # # Proyectar los puntos optimizados en cada imagen usando T_wc1, T_wc2_opt, T_wc3
    # x1_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc1) @ X_opt.T
    # x2_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc2_opt) @ X_opt.T
    # # x3_p_opt = K_c @ np.eye(3, 4) @ np.linalg.inv(T_wc3) @ X_opt.T

    # # Normalizar las coordenadas para obtener las proyecciones en píxeles
    # x1_p_opt /= x1_p_opt[2, :]
    # x2_p_opt /= x2_p_opt[2, :]
    # x3_p_opt /= x3_p_opt[2, :]
    
    utils.drawSystem(T_wc1, T_wc2, X_opt)





# Run the main function
if __name__ == "__main__":
    main()
