import numpy as np
import cv2
import random
import argparse
from matplotlib import pyplot as plt

def normalize_points(pts):
    """
    Normalize points so that the centroid is at the origin and the average distance to the origin is sqrt(2).
    Args:
        pts: Nx2 array of points.
    Returns:
        pts_normalized: Nx2 array of normalized points.
        T: 3x3 transformation matrix that normalizes the points.
    """
    # Get the centroid of the points.
    centroid = np.mean(pts, axis=0)
    
    # Substract the centroid from the points so that the centroid is at the origin.
    pts_shifted = pts - centroid
    
    # Compute the average distance to the origin.
    avg_dist = np.mean(np.linalg.norm(pts_shifted, axis=1))
    
    # Scale the points so that the average distance is sqrt(2).
    scale = np.sqrt(2) / avg_dist
    pts_normalized = pts_shifted * scale
    
    # Build and return the transformation matrix.
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    
    return pts_normalized, T

def compute_fundamental_matrix(x1, x2):
    """
    Estimate the fundamental matrix using the 8-point algorithm.
    Args:
        x1, x2: Corresponding points in the two images.
    Returns:
        F: The estimated fundamental matrix.
    """
    # Normalie the points.
    x1_normalized, T1 = normalize_points(x1)
    x2_normalized, T2 = normalize_points(x2)
    
    # Build the matrix A.
    N = x1_normalized.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1x = x1_normalized[i, 0]
        y1x = x1_normalized[i, 1]
        x2x = x2_normalized[i, 0]
        y2x = x2_normalized[i, 1]
        A[i] = [x1x*x2x, x1x*y2x, x1x, y1x*x2x, y1x*y2x, y1x, x2x, y2x, 1]
    
    # Solve Af = 0 using SVD.
    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint.
    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0  # Third singular value is 0.
    F_normalized = U @ np.diag(S) @ Vt
    
    F = T2.T @ F_normalized @ T1
    
    # Normalize the fundamental matrix so that the last element is 1.
    return F / F[2, 2]

def point_line_distance(F, p1, p2):
    """
    Compute the distance from point to epipolar line.
    Args:
        F: fundamental matrix.
        p1, p2: points from images (Nx2 array).
    Returns:
        dist: distance from points to epipolar lines.
    """
    p1_homogeneous = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2_homogeneous = np.hstack((p2, np.ones((p2.shape[0], 1))))
    
    # Epipolar lines in the second image for points in the first image
    lines2 = (F @ p1_homogeneous.T).T
    
    # Epipolar lines in the first image for points in the second image
    lines1 = (F.T @ p2_homogeneous.T).T
    
    # Compute distances from points to their corresponding epipolar lines
    dist1 = np.abs(np.sum(lines1 * p1_homogeneous, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    dist2 = np.abs(np.sum(lines2 * p2_homogeneous, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    return dist1 + dist2

def ransac_fundamental_matrix(matches1, matches2, num_iterations=10000, threshold=3):
    """
    Perform RANSAC to estimate the fundamental matrix.
    Args:
        matches1, matches2: Corresponding points in the two images.
        num_iterations: Number of RANSAC iterations.
        threshold: Distance threshold for inliers.
    Returns:
        F: Fundamental matrix.
        inliers: Inliers corresponding to the fundamental matrix.
    """
    num_points = matches1.shape[0]
    best_inliers = []
    best_F = None

    if num_points < 8:
        print("Not enough points to estimate the fundamental matrix (need at least 8).")
        return None, None

    for _ in range(num_iterations):
        # Randomly sample 8 points for the 8-point algorithm
        sample_indices = random.sample(range(num_points), 8)
        sample_p1 = matches1[sample_indices]
        sample_p2 = matches2[sample_indices]
        
        # Estimate fundamental matrix from the sample
        try:
            F = compute_fundamental_matrix(sample_p1, sample_p2)
        except np.linalg.LinAlgError as e:
            print(f"Fundamental matrix computation failed with error: {e}")
            continue
        
        # Compute transfer error for all points
        errors = point_line_distance(F, matches1, matches2)
        
        # Identify inliers
        inliers = np.where(errors < threshold)[0]
        
        # Keep the fundamental matrix with the most inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F

    # Check if we have enough inliers to refine the fundamental matrix
    if best_F is None or len(best_inliers) < 8:
        print("RANSAC failed to find a valid fundamental matrix.")
        print("best_F is ", best_F)
        return None, None
    
    # Refine fundamental matrix using all inliers
    inlier_p1 = matches1[best_inliers]
    inlier_p2 = matches2[best_inliers]
    best_F = compute_fundamental_matrix(inlier_p1, inlier_p2)
    
    return best_F, best_inliers

def draw_epipolar_lines(img1, img2, matches1, matches2, F, refined_matches=None):
    """
    Draw epipolar lines on the images corresponding to the points
    Args:
        img1, img2: Images
        matches1, matches2: Corresponding points in the two images
        F: Fundamental matrix
    """
    def draw_lines(img, lines, pts, colors):
        '''
        Draw the epilines for the points in one image over the other image.
            img - image on which we draw the epilines for the points in img2.
            lines - corresponding epilines.
            pts - corresponding points.
            color - color of the epilines.
        '''
        r, c = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r, pt, color in zip(lines, pts, colors):
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(map(int, pt)), 5, color, -1)
        return img

    # Generate a random colro for each line
    num_lines = matches1.shape[0]
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_lines)]

    # Compute the epipolar lines in both images
    lines1 = cv2.computeCorrespondEpilines(matches2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines = draw_lines(img1, lines1, matches1, colors)

    lines2 = cv2.computeCorrespondEpilines(matches1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines = draw_lines(img2, lines2, matches2, colors)

    if refined_matches is not None:
        print ("Drawing refined matches, ", refined_matches.shape)
        for i in range(refined_matches.shape[0]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # draw a cross on the points
            img1_with_lines = cv2.drawMarker(img1_with_lines, (int(refined_matches[i, 0, 0]), int(refined_matches[i, 0, 1])), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            img2_with_lines = cv2.drawMarker(img2_with_lines, (int(refined_matches[i, 1, 0]), int(refined_matches[i, 1, 1])), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    plt.subplot(121), plt.imshow(img1_with_lines)
    plt.subplot(122), plt.imshow(img2_with_lines)
    plt.show()

import numpy as np
import cv2

def compute_epipolar_lines(F, points):
    """
    Compute the epipolar lines corresponding to points in the other image.
    Args:
        F: Fundamental matrix.
        points: Nx2 array of points in one image.
    Returns:
        Epipolar lines corresponding to points in the other image.
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    lines = (F @ points_h.T).T  # Epipolar lines
    return lines

def guided_matching(matches1, matches2, F, threshold=3.0):
    """
    Perform guided matching using the epipolar geometry
    Args:
        matches1: Nx2 array of points from image 1.
        matches2: Nx2 array of points from image 2.
        F: Fundamental matrix.
        threshold: Distance threshold for epipolar constraint.
    Returns:
        Refined matches after guided matching.
    """
    lines1 = compute_epipolar_lines(F.T, matches2)  # Epipolar lines in image 1 for points in image 2
    lines2 = compute_epipolar_lines(F, matches1)    # Epipolar lines in image 2 for points in image 1

    refined_matches = []
    
    # For each point in image 1, find the closest point in image 2 along the epipolar line
    for i, (p1, p2) in enumerate(zip(matches1, matches2)):
        line1 = lines1[i]
        line2 = lines2[i]
        
        # Distance from p1 to epipolar line in image 1
        dist1 = np.abs(line1[0] * p1[0] + line1[1] * p1[1] + line1[2]) / np.sqrt(line1[0]**2 + line1[1]**2)
        
        # Distance from p2 to epipolar line in image 2
        dist2 = np.abs(line2[0] * p2[0] + line2[1] * p2[1] + line2[2]) / np.sqrt(line2[0]**2 + line2[1]**2)

        # If both distances are below the threshold, keep the match
        if dist1 < threshold and dist2 < threshold:
            refined_matches.append((p1, p2))
    
    return np.array(refined_matches)

# Example usage:
if __name__ == "__main__":
    # Load images and feature matches (for illustration, using SIFT here)

    img1 = cv2.imread('./ext/image1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./ext/image2.png', cv2.IMREAD_GRAYSCALE)

    matches1 = np.array([...])  # Nx2 array of points in image 1
    matches2 = np.array([...])  # Nx2 array of points in image 2

    parser = argparse.ArgumentParser(
        description='Image matching using RANSAC and guided matching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--useSuperGlue', action='store_true',
                        help='Use SuperGlue for feature matching')
    args = parser.parse_args()

    if args.useSuperGlue:
        path = './ext/image1_image2_matches.npz'
        npz = np.load(path) # Load dictionary with super point
        # Create a boolean mask with True for keypoints with a good match, and False for the rest
        mask = npz['matches'] > -1
        
        # Using the boolean mask, select the indexes of matched keypoints from image 2
        idxs = npz['matches'][mask]
        # Using the boolean mask, select the keypoints from image 1 with a good match
        matches1 = npz['keypoints0'][mask]
        matches2 = npz['keypoints1'][idxs]
        descriptors1 = npz['descriptors0']
        descriptors2 = npz['descriptors1']
    else:
        # Detect features using SIFT and match (for demo purposes)
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply NNDR (nearest-neighbor distance ratio) for filtering matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Get matched keypoints
        matches1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
        matches2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

        print(f"Shape of matches1: {matches1.shape}")
        print(f"Shape of matches2: {matches2.shape}")

    # Estimate fundamental matrix using RANSAC
    F, inliers = ransac_fundamental_matrix(matches1, matches2)

    # Perform guided matching using epipolar constraint
    refined_matches = guided_matching(matches1, matches2, F)

    print(f"Number of matches before: {len(matches1)}")
    print(f"Number of refined matches after guided matching: {len(refined_matches)}")

    # Check the shape of F before using it
    if F is not None:
        print(f"Shape of the Fundamental Matrix F: {F.shape}")
    else:
        print("Fundamental matrix estimation failed.")

    # Draw epipolar lines only if F is valid
    if F is None :
        print("Error: Fundamental matrix F is not valid.")
    elif not F.shape == (3, 3):
        print("Error: Fundamental matrix F is not 3x3.")
    else:
        draw_epipolar_lines(img1, img2, matches1[inliers], matches2[inliers], F, refined_matches)

