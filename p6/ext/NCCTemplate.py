#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
from scipy.interpolate import RectBivariateSpline
from interpolationFunctions import int_bilineal, numerical_gradient
from scipy.ndimage import map_coordinates, sobel

def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch

    # Mean of the patch
    i0_mean = np.mean(i0)
    i0_std = np.std(i0)  # Standard deviation of the patch

    # Initialize the result array
    result = np.zeros(search_area.shape, dtype=np.float64)

    # Margins to avoid boundary issues
    margin_y = i0.shape[0] // 2
    margin_x = i0.shape[1] // 2

    # Iterate over the search area (excluding margins)
    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            # Extract the corresponding region in the search area
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]

            # Compute mean and std of the region
            i1_mean = np.mean(i1)
            i1_std = np.std(i1)

            # Avoid division by zero
            if i1_std == 0 or i0_std == 0:
                result[i, j] = 0  # Invalid or texture-less region
            else:
                # Compute the NCC value
                ncc = np.sum((i0 - i0_mean) * (i1 - i1_mean)) / (i0_std * i1_std * i0.size)
                result[i, j] = ncc
    return result

def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):

    # Attention!! we are not checking the padding
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    result = normalized_cross_correlation(patch, search_area)

    iMax, jMax = np.where(result == np.amax(result))

    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow

def lucas_kanade_sparse_optical_flow(img1, img2, initial_u, window_size=11, epsilon=1e-6, max_iter=30):
    """
    Refines optical flow using Lucas-Kanade approach starting from the initial motion vectors (u, v),
    utilizing bilinear interpolation and numerical gradient functions.
    
    Parameters:
        img1: np.array - Grayscale image at time t.
        img2: np.array - Grayscale image at time t+1.
        initial_u: float - Initial horizontal motion vector.
        initial_v: float - Initial vertical motion vector.
        window_size: int - Size of the centered window for refinement (default is 11x11).
        epsilon: float - Convergence threshold for motion updates (default is 1e-2).
        max_iter: int - Maximum number of iterations (default is 50).

    Returns:
        refined_u, refined_v: float - Refined motion vectors.
    """
    half_window = window_size // 2
    height, width = img1.shape

    # Define the region of interest for refinement
    x_range = np.arange(half_window, width - half_window)
    y_range = np.arange(half_window, height - half_window)
    X, Y = np.meshgrid(x_range, y_range)  # Create a meshgrid for proper alignment

    # Flatten meshgrid for processing points
    points = np.stack((Y.ravel(), X.ravel()), axis=-1)

    # Initialize motion vectors
    u = initial_u

    # Compute image gradients using numerical_gradient
    gradients = numerical_gradient(img1, points)
    Ix = gradients[:, 1].reshape(height - 2 * half_window, width - 2 * half_window)
    Iy = gradients[:, 0].reshape(height - 2 * half_window, width - 2 * half_window)

    # ep_reshaped = error_patch.reshape(Ix.shape)

    # Compute the A matrix and vector b
    A = np.array([[np.sum(Ix ** 2), np.sum(Ix * Iy)],
                    [np.sum(Ix * Iy), np.sum(Iy ** 2)]])

    # Check if A is invertible
    if np.linalg.det(A) < 1e-6:
        print("A is not invertible, stopping refinement")
        return u # A is not invertible, stop refinement

    A_inv = np.linalg.inv(A)
    img1_patch = int_bilineal(img1, points)#.reshape(img1.shape)

    for _ in range(max_iter):
        # Compute warped points using the current motion (u, v)
        X_warped, Y_warped = np.meshgrid(x_range + u[0], y_range + u[1])  # Create a meshgrid for proper alignment
        warped_points = np.stack((Y_warped.ravel(), X_warped.ravel()), axis=-1)

        # Clip points to stay within image boundaries
        warped_points[:, 0] = np.clip(warped_points[:, 0], 0, height - 2)
        warped_points[:, 1] = np.clip(warped_points[:, 1], 0, width - 2)


        # Interpolate img2 at warped points
        warped_patch = int_bilineal(img1, warped_points)#.reshape(img1.shape)

        # Compute error between img1 and warped img2
        error_patch = (warped_patch - img1_patch).reshape(height - 2 * half_window, width - 2 * half_window)

        # print("error_patch",error_patch.shape)
        # print("warped_patch",warped_patch.shape)
        # print("img1_patch",img1_patch.shape)

        b = np.array([-np.sum(Ix * error_patch), -np.sum(Iy * error_patch)])

        # print ("A", A)
        # print ("b", b)

        # Solve for delta motion
        # delta_u = np.linalg.solve(A, b)
        delta_u = A_inv @ b

        # Update motion
        # u += delta_u
        u = initial_u + delta_u
        # u = delta_u

        # Check for convergence
        if np.linalg.norm([delta_u]) < epsilon:
            break

    return u


def lucas_kanade_sparse_optical_flow2(img1_gray, img2_gray, i_img, j_img, u_seed, patch_half_size=5, epsilon=1e-3, max_iter=50):
    """
    Refines the initial motion (u, v) using the iterative Lucas-Kanade method.
    
    Parameters:
        img1_gray: Grayscale image at time t (numpy array).
        img2_gray: Grayscale image at time t+1 (numpy array).
        i_img, j_img: Coordinates of the point of interest.
        u_seed: Initial motion vector [u, v] from NCC.
        patch_half_size: Half the size of the square patch (default is 5 for an 11x11 window).
        epsilon: Convergence threshold for ||Δu|| (default is 1e-3).
        max_iter: Maximum number of iterations (default is 50).
        
    Returns:
        refined_u: Refined motion vector [u, v].
    """
    # Define the patch region
    patch_size = 2 * patch_half_size + 1
    y, x = np.meshgrid(
        np.arange(-patch_half_size, patch_half_size + 1),
        np.arange(-patch_half_size, patch_half_size + 1),
        indexing='ij'
    )
    x += j_img
    y += i_img
    
    # Compute image gradients on img1_gray
    Ix = sobel(img1_gray, axis=1)  # Gradient in the x-direction
    Iy = sobel(img1_gray, axis=0)  # Gradient in the y-direction
    It = img2_gray - img1_gray     # Temporal difference
    
    # Extract the gradients for the patch
    Ix_patch = Ix[y, x]
    Iy_patch = Iy[y, x]
    
    # Compute the matrix A
    A = np.array([
        [np.sum(Ix_patch**2), np.sum(Ix_patch * Iy_patch)],
        [np.sum(Ix_patch * Iy_patch), np.sum(Iy_patch**2)]
    ])
    
    # Check invertibility of A
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A is not invertible. Cannot refine motion.")
    
    # Initialize motion vector u
    u = np.array(u_seed, dtype=np.float64)
    
    for _ in range(max_iter):
        # Warp coordinates according to the current motion u
        x_warp = x + u[0]
        y_warp = y + u[1]
        
        # Interpolate I2 at the warped coordinates
        I2_patch = map_coordinates(img2_gray, [y_warp.flatten(), x_warp.flatten()], order=1).reshape(patch_size, patch_size)
        
        # Compute error (I1(x) - I2(x + u))
        error = I2_patch - img1_gray[y, x]
        
        # Compute vector b
        b = np.array([
            np.sum(error * Ix_patch),
            np.sum(error * Iy_patch)
        ])
        
        # Solve Δu = A^-1 * b
        delta_u = np.linalg.solve(A, b)
        
        # Update u
        u += delta_u
        
        # Check for convergence
        if np.linalg.norm(delta_u) < epsilon:
            break
    
    return u


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    img1 = read_image("frame10.png")
    img2 = read_image("frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points
    points_selected = np.loadtxt('points_selected.txt')
    points_selected = points_selected.astype(int)

    template_size_half = 5
    searching_area_size: int = 15

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)

    # Initial motion vectors from NCC

    initial_u = seed_optical_flow_sparse[0]

    # initial_u = seed_optical_flow_sparse[:, 0]
    # initial_v = seed_optical_flow_sparse[:, 1]

    # Call the Lucas-Kanade refinement
    refined_u = lucas_kanade_sparse_optical_flow(img1_gray, img2_gray, initial_u)

    print(f"Refined motion vectors: u={refined_u}")
