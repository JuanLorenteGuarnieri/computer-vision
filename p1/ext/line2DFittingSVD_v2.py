#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: Line fitting with SVD
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.5
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.hstack((0, -l[2] / l[1]))
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.hstack((-l[2] / l[0], 0))
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)


if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # This is the ground truth
    l_GT = np.array([[2], [1], [-1500]])

    plt.figure(1)
    plt.plot([-100, 1800], [0, 0], '--k', linewidth=1)
    plt.plot([0, 0], [-100, 1800], '--k', linewidth=1)
    # Draw the line segment p_l_x to  p_l_y
    drawLine(l_GT, 'g-', 1)
    plt.draw()
    plt.axis('equal')

    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    ## Generating points lying on the line but adding perpendicular Gaussian noise
    l_GTNorm = l_GT/np.sqrt(np.sum(l_GT[0:2]**2, axis=0)) #Normalized the line with respect to the normal norm

    x_l0 = np.vstack((-l_GTNorm[0:2]*l_GTNorm[2],1))  #The closest point of the line to the origin
    plt.plot([0, x_l0[0,0]], [0, x_l0[1,0]], '-r')
    plt.draw()

    # mu = np.arange(-1000, 1000, 250)
    # noiseSigma = 15 #Standard deviation
    # xGT = x_l0 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) @ (l_GTNorm * mu)
    # x = xGT + np.diag([1, 1, 0]) @ np.random.normal(0, noiseSigma, (3, len(mu)))

    xGT = np.loadtxt('./p1/ext/x2DGTLineFittingSVD.txt')
    x = np.loadtxt('./p1/ext/x2DLineFittingSVD.txt')
    plt.plot(xGT[0, :], xGT[1, :], 'b.')
    plt.plot(x[0, :], x[1, :], 'rx')
    plt.draw()
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    ## Fit the least squares solution from points with noise using svd
    # we want to solve the equation x.T @ l = 0
    u, s, vh = np.linalg.svd(x.T) # svd function returns vh which is the tranpose version of V matrix.
    sM = scAlg.diagsvd(s, u.shape[0], vh.shape[0])  # svd function returns the diagonal s values instead of the S matrix. 
    l_ls = vh[-1, :]
    
        
    # Task 3.1: Fit the least squares solution using only the 2 extreme points from x
    x_extreme = x[:, [0, -1]]  # Only select the first and last points

    u, s, vh = np.linalg.svd(x_extreme.T)  # Compute SVD
    l_ls_extreme = vh[-1, :]  # Line coefficients from the last row of vh
    
    # Task 3.2: Fit the least squares solution using the original 5 perfect points xGT
    u_gt, s_gt, vh_gt = np.linalg.svd(xGT.T)  # Compute SVD using xGT
    l_ls_gt = vh_gt[-1, :]  # Line coefficients from the last row of vh


    # Notice that the input matrix A of the svd has been decomposed such that A = u @ sM @ vh 

    drawLine(l_ls, 'r--', 1)
    drawLine(l_ls_extreme, 'b--', 2)
    drawLine(l_ls_gt, 'g--', 2)

    plt.draw()
    plt.waitforbuttonpress()

    print("Singular values for 2 extreme points:", s)
    print("Singular values for ground truth points:", s_gt)


    ## Project the points on the line using SVD
    s[2] = 0  # If all the points are lying on the line s[2] = 0, therefore we impose it
    xProjectedOnTheLine = (u @ scAlg.diagsvd(s, u.shape[0], vh.shape[0]) @ vh).T
    xProjectedOnTheLine /= xProjectedOnTheLine[2, :]

    plt.plot(xProjectedOnTheLine[0,:], xProjectedOnTheLine[1, :], 'bx')
    plt.show()
    print('End')

    # Task 3.3: Interpretation of Singular Values
    # Singular values 𝑠 provide insight into the geometric properties of the points:
    # In both cases, the matrix is decomposed into matrices 𝑈, 𝑆, and 𝑉 such that the input matrix 𝐴 can be represented as:
    #             𝐴=𝑈𝑆𝑉^𝑇
    # Where:
    #   𝑈 contains the left singular vectors.
    #   𝑆 is a diagonal matrix containing singular values.
    #   𝑉 contains the right singular vectors.
    # When using 2 extreme points:
    #   The size of the matrices 𝑈, 𝑆, and 𝑉:
    #     𝑈 is 2×2 because we only have 2 points.
    #     𝑆 is 2×2, and 𝑉 is 2×3.
    #   Singular values interpretation:
    #     One singular value is large (representing the main direction of the points), and one is very small,
    #     close to zero (indicating that the points are nearly collinear).
    # When using 5 ground-truth points:
    #   The size of the matrices 𝑈, 𝑆, and 𝑉:
    #     𝑈 is 5×5, 𝑆 is 5×5, and 𝑉 is 5×3.
    #   Singular values interpretation:
    #     One singular value will be large (capturing the direction of the line), while the rest will be close to zero since the points lie perfectly on the line.

    # Task 3.4: Interpretation of Singular Values
    # By setting the third singular value to zero, we are enforcing that all points lie exactly on the line.
    # This essentially removes any noise or deviation from the line, forcing the points to lie on the best-fit line.
    #
    # The re-composed matrix using the modified singular values represents the points projected onto the line, i.e.,
    # the closest points on the line that the noisy points correspond to.