import numpy as np
import cv2
import random
import argparse
from matplotlib import pyplot as plt

def compute_homography(pts_src, pts_dst):
    """
    findHomography implementation.
    """
    A = []
    
    # Build the matrix A as seen in the slides.
    for i in range(pts_src.shape[0]):
        x, y = pts_src[i, 0], pts_src[i, 1] 
        u, v = pts_dst[i, 0], pts_dst[i, 1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    
    # Get the eigen values.
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    # Normalize the homography matrix.
    H = H / H[2, 2]

    return H

def transfer_error(H, p1, p2):
    """
    Compute the transfer error between points transformed by homography.
    Args:
        H: homography matrix
        p1, p2: points from images (Nx2 arrays).
    Returns:
        Transfer error for each point.
    """
    p1_homogeneous = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2_projected = (H @ p1_homogeneous.T).T
    p2_projected /= p2_projected[:, 2][:, np.newaxis]
    error = np.linalg.norm(p2_projected[:, :2] - p2, axis=1)
    
    return error

def ransac_homography(matches1, matches2, num_iterations=1000, threshold=5):
    """
    Perform RANSAC to estimate a homography matrix.
    Args:
        matches1, matches2: Matched points between two images (Nx2 arrays).
        num_iterations: Number of RANSAC iterations.
        threshold: Transfer error threshold to classify inliers.
    Returns:
        best_H: Homography matrix with the most inliers.
        best_inliers: List of indices of inliers.
    """
    num_points = matches1.shape[0]
    best_inliers = []
    best_H = None

    for _ in range(num_iterations):
        # Randomly sample 4 points
        sample_indices = random.sample(range(num_points), 4)
        sample_p1 = matches1[sample_indices]
        sample_p2 = matches2[sample_indices]
        
        # Estimate homography from the sample
        H = compute_homography(sample_p1, sample_p2)
        
        # Compute transfer error for all points
        errors = transfer_error(H, matches1, matches2)
        
        # Identify inliers
        inliers = np.where(errors < threshold)[0]
        
        # Keep the homography with the most inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    # Refine homography using all inliers
    if best_H is not None and len(best_inliers) > 4:
        inlier_p1 = matches1[best_inliers]
        inlier_p2 = matches2[best_inliers]
        best_H = compute_homography(inlier_p1, inlier_p2)
    
    return best_H, best_inliers

def draw_matches(img1, img2, matches1, matches2, inliers):
    """
    Display matches and inliers between two images.
    Args:
        img1, img2: Input images.
        matches1, matches2: Matched points between two images (Nx2 arrays).
        inliers: List of indices of inliers.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_combined[:h1, :w1] = img1
    img_combined[:h2, w1:] = img2
    
    for i, (pt1, pt2) in enumerate(zip(matches1, matches2)):
        pt1 = tuple(np.round(pt1).astype(int))
        pt2 = tuple(np.round(pt2).astype(int) + np.array([w1, 0]))
        color = (0, 255, 0) if i in inliers else (0, 0, 255)
        cv2.line(img_combined, pt1, pt2, color, 1)
        cv2.circle(img_combined, pt1, 4, color, -1)
        cv2.circle(img_combined, pt2, 4, color, -1)
    
    plt.imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
    plt.show()

# Example usage:
if __name__ == "__main__":

    # Load images and feature matches (for illustration, using SIFT here)
    img1 = cv2.imread('./ext/image1.png')
    img2 = cv2.imread('./ext/image2.png')

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

    # Estimate homography using RANSAC
    H, inliers = ransac_homography(matches1, matches2)

    # Draw matches and inliers
    draw_matches(img1, img2, matches1, matches2, inliers)
