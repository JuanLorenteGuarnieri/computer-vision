#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: SIFT matching
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbors Matching algorithm checking the Distance Ratio.
    A match is accepted only if the distance to the nearest neighbor is less than
    distRatio times the distance to the second nearest neighbor.
    -input:
        desc1: descriptors from image 1 (nDesc1 x 128)
        desc2: descriptors from image 2 (nDesc2 x 128)
        distRatio: distance ratio threshold (0.0 < distRatio < 1.0)
        minDist: minimum distance threshold to accept a match
    -output:
        matches: list of accepted matches with [[indexDesc1, indexDesc2, distance], ...]
    """
    matches = []
    nDesc1 = desc1.shape[0]

    for kDesc1 in range(nDesc1):
        # Compute L2 distance (Euclidean distance) between desc1[kDesc1] and all descriptors in desc2
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))

        # Sort the distances and get the two nearest neighbors
        indexSort = np.argsort(dist)
        d1 = dist[indexSort[0]]  # Distance to nearest neighbor
        d2 = dist[indexSort[1]]  # Distance to second nearest neighbor

        # Apply NNDR: check if d1 is less than distRatio * d2
        if d1 < distRatio * d2 and d1 < minDist:
            # If the match passes the distance ratio test and is below the minimum distance threshold, accept it
            matches.append([kDesc1, indexSort[0], d1])

    return matches


if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.8
    minDist = 1000 # higher -> more FN matches (0-100) | smaller -> more FP matches (>1000)

    '''
    The chessboard image presents a particularly challenging case for feature matching due to image aliasing, a phenomenon
    where high-frequency patterns (like the alternating black and white squares) become indistinguishable at lower resolutions
    or when subjected to perspective distortions. Here's why it becomes difficult to remove false positives in this case:

        Repetitive Patterns: The chessboard is a grid of alternating, identical black and white squares, which results in
        many local areas of the image looking very similar. SIFT, like many feature detectors, relies on local gradients to
        generate descriptors. However, on a chessboard, multiple patches of the image may generate nearly identical descriptors
        due to the repeating nature of the pattern. This leads to many ambiguous matches, even when applying the distance ratio criterion.

        Aliasing and Sampling: When the chessboard is viewed at a lower resolution or at an angle, aliasing occurs, which causes
        neighboring pixels to appear blended. As a result, the SIFT descriptors can be distorted in ways that create matches
        between points that shouldn't correspond. Even small changes in scale or rotation can lead to dramatic changes in the
        appearance of the chessboard, leading to confusion in the feature matching process.

        Keypoint Distribution: The edges between the black and white squares are strong feature locations, and SIFT detects
        many keypoints in these areas. Because so many of the squares' edges look alike, there are often multiple close matches
        for each keypoint. This high density of keypoints, combined with aliasing, increases the likelihood of false matches.
        Reducing the false positive rate without eliminating correct matches can be especially hard when the true matches are so
        visually similar to false ones.
    '''

    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Plot the first 10 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

