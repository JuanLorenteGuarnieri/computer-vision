import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

def compute_fundamental_matrix(p1, p2):
    """
    Compute fundamental matrix using 8-point algorithm
    p1: points from image 1 (Nx2 array)
    p2: points from image 2 (Nx2 array)
    """
    A = []
    for i in range(p1.shape[0]):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint by zeroing the smallest singular value
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vh
    
    return F

def point_line_distance(F, p1, p2):
    """
    Compute the distance from point to epipolar line
    F: fundamental matrix
    p1: points from image 1 (Nx2 array)
    p2: points from image 2 (Nx2 array)
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

def ransac_fundamental_matrix(matches1, matches2, num_iterations=2000, threshold=0.01):
    """
    Perform RANSAC to estimate the fundamental matrix.
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
        return None, None
    
    # Refine fundamental matrix using all inliers
    inlier_p1 = matches1[best_inliers]
    inlier_p2 = matches2[best_inliers]
    best_F = compute_fundamental_matrix(inlier_p1, inlier_p2)
    
    return best_F, best_inliers


    # Refine fundamental matrix using all inliers with 8-point algorithm
    if best_F is not None and len(best_inliers) > 8:
        inlier_p1 = matches1[best_inliers]
        inlier_p2 = matches2[best_inliers]
        best_F = compute_fundamental_matrix(inlier_p1, inlier_p2)
    
    return best_F, best_inliers

def draw_epipolar_lines(img1, img2, matches1, matches2, F):
    """
    Draw epipolar lines on the images corresponding to the points
    """
    def draw_lines(img, lines, pts):
        ''' img - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(pt), 5, color, -1)
        return img

    # Compute the epipolar lines in both images
    lines1 = cv2.computeCorrespondEpilines(matches2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_with_lines = draw_lines(img1, lines1, matches1)

    lines2 = cv2.computeCorrespondEpilines(matches1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_with_lines = draw_lines(img2, lines2, matches2)

    plt.subplot(121), plt.imshow(img1_with_lines)
    plt.subplot(122), plt.imshow(img2_with_lines)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load images and feature matches (for illustration, using SIFT here)

    img1 = cv2.imread('./ext/image1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./ext/image2.png', cv2.IMREAD_GRAYSCALE)

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

    # Check the shape of F before using it
    if F is not None:
        print(f"Shape of the Fundamental Matrix F: {F.shape}")
    else:
        print("Fundamental matrix estimation failed.")

    # Draw epipolar lines only if F is valid
    if F is not None and F.shape == (3, 3):
        draw_epipolar_lines(img1, img2, matches1[inliers], matches2[inliers], F)
    else:
        print("Error: Fundamental matrix F is not valid or not 3x3.")

