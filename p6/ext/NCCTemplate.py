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
    """
    Estimate the optical flow using normalized cross-correlation (NCC) for a single point.
    Parameters:
        img1_gray: np.array - Grayscale image at time t.
        img2_gray: np.array - Grayscale image at time t+1.
        i_img: int - Row index of the point.
        j_img: int - Column index of the point.
        patch_half_size: int - Half size of the patch to extract around the point (default is 5).
        searching_area_size: int - Size of the searching area around the point (default is 100).

    Returns:
        i_flow: int - Estimated row displacement.
        j_flow: int - Estimated column displacement.
    """
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

def lucas_kanade_sparse_optical_flow(img1, img2, points, initial_u, window_size=11, epsilon=1e-6, max_iter=50):
    """
    Refines optical flow using Lucas-Kanade approach starting from the initial motion vectors (u, v),
    utilizing bilinear interpolation and numerical gradient functions.
    
    Parameters:
        img1: np.array - Grayscale image at time t.
        img2: np.array - Grayscale image at time t+1.
        points: np.array[:, 2] - List of sparse points to refine
        initial_u: np.array[2] - Initial motion vector.
        window_size: int - Horizontal and vertical size of the centered window for
                           refinement, which must be odd (default is 11).
        epsilon: float - Convergence threshold for motion updates (default is 1e-6).
        max_iter: int - Maximum number of iterations (default is 50).

    Returns:
        refined_u: np.array[2] - Refined motion vector.
    """
    half_window = window_size // 2

    # Compute image gradients using numerical_gradient
    Ix, Iy = np.gradient(img1)

    refined_u = np.zeros_like(initial_u)

    for idx, (x, y) in enumerate(points):
        u = initial_u[idx].T.copy()
        # Extract the patch around the current point
        x_start, x_end = int(x - half_window), int(x + half_window + 1)
        y_start, y_end = int(y - half_window), int(y + half_window + 1)

        Ix_patch = Ix[y_start:y_end, x_start:x_end].flatten()
        Iy_patch = Iy[y_start:y_end, x_start:x_end].flatten()
        I0_patch = img1[y_start:y_end, x_start:x_end].flatten()

        # Compute the A matrix and vector b
        A = np.array([[np.sum(Ix_patch ** 2), np.sum(Ix_patch * Iy_patch)],
                      [np.sum(Ix_patch * Iy_patch), np.sum(Iy_patch ** 2)]])

        # Check if A is invertible
        if np.linalg.det(A) < 1e-6:
            print(f"Skipping point {x, y} due to non-invertible A matrix.")
            refined_u[idx] = u
            continue # A is not invertible, stop refinement

        A_inv = np.linalg.inv(A)

        for _ in range(max_iter):
            # Compute warped points using the current motion (u)
            x_coords, y_coords = np.meshgrid(
                np.arange(x_start, x_end) + u[0],
                np.arange(y_start, y_end) + u[1]
            )

            # Interpolate the warped patch using bilinear interpolation
            warped_points = np.vstack((y_coords.ravel(), x_coords.ravel())).T
            warped_patch = int_bilineal(img2, warped_points)

            # Compute error between img1 and warped img2
            error_patch = warped_patch - I0_patch

            # Compute the b vector
            b = -np.array([np.sum(Iy_patch * error_patch), np.sum(Ix_patch * error_patch)])

            # Solve for delta motion
            delta_u = A_inv @ b

            # Update motion
            u += delta_u

            # Check for convergence
            if np.linalg.norm([delta_u]) < epsilon:
                break

        # Update refined motion vector
        refined_u[idx] = u

    return refined_u

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

    # Call the Lucas-Kanade refinement
    refined_u = lucas_kanade_sparse_optical_flow(img1_gray, img2_gray, points_selected, seed_optical_flow_sparse)

    print(f"Refined motion vectors: u={refined_u}")
