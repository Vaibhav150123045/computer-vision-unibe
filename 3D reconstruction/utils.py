import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import sparse
from scipy.sparse.linalg import lsmr

def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    #
    # Output: T 3x3 transformation matrix of points

    # TO DO TASK:
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid
    mean = np.mean(x[:, :2], axis=0)
    std = np.std(x[:, :2])
    T = np.array([
            [1 / std, 0, -mean[0] / std],
            [0, 1 / std, -mean[1] / std],
            [0, 0, 1]
        ])

    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm
    Inputs:
                    x1      3xN     homogeneous coordinates of matched points in view 1
                    x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
                    F       3x3     fundamental matrix
    """
    N = x1.shape[1]

    if normalize:
        # Construct transformation matrices to normalize the coordinates
        # TODO
        T_x1 = get_normalization_matrix(x1.T)
        T_x2 = get_normalization_matrix(x2.T)

        # Normalize inputs
        # TODO
        x1 = (T_x1 @ x1).T
        x2 = (T_x2 @ x2).T


    # Construct matrix A encoding the constraints on x1 and x2
    # TODO
    N = x1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        X1 = x1[i]
        X2 = x2[i]
        A[i] = [X2[0] * X1[0], X2[0] * X1[1], X2[0],
                X2[1] * X1[0], X2[1] * X1[1], X2[1],
                X1[0], X1[1], 1]

    # Solve for f using SVD
    # TODO
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce that rank(F)=2
    # TODO
    U, S, V = np.linalg.svd(F)
    S[2] = 0  # Set the smallest singular value to 0
    F = U @ np.diag(S) @ V

    if normalize:
        # Transform F back
        # TODO
        F = T_x2.T @ F @ T_x1

    return F


def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    """

    # The epipole is the null space of F (F * e = 0)
    # TODO

    # Perform Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(F)
    
    # The null space is given by the last column of V (transpose of Vt)
    e = Vt[-1]
    e /= e[2]

    return e


def plot_epipolar_line(im, F, x, e):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    m, n = im.shape[:2]
    # TODO
    
    epiline_coeff = np.dot(F, x) # line = epiline_coeff[0]*x + epiline_coeff[1]*y + epiline_coeff[2]*z = 0
    a = epiline_coeff[0]; b = epiline_coeff[1]; c=epiline_coeff[2]
    # Define x range for the plot
    x = np.linspace(n, 5, 100)

    # Calculate corresponding y values
    if b != 0:
        y = -(a * x + c) / b
        plt.plot(x, y)
    else:
        # Vertical line (b = 0)
        x_vertical = -c / a
        plt.axvline(x=x_vertical, color='r')


def ransac(leftPoints, rightPoints, good_threshold, max_iterations=1000):
    """
    RANSAC-based estimation of the Fundamental Matrix.
    Parameters:
    - leftPoints: Nx2 array of points from the first image.
    - rightPoints: Nx2 array of points from the second image.
    - good_threshold: Distance threshold to consider a point as an inlier.
    - max_iterations: Number of RANSAC iterations.
    
    Returns:
    - F: Estimated Fundamental Matrix.
    - inliers: Boolean array indicating inlier matches.
    """
    best_F = None
    best_inliers = []
    best_inlier_count = 0

    leftPoints = leftPoints.T
    rightPoints = rightPoints.T
    
    num_points = leftPoints.shape[0]
    
    for _ in range(max_iterations):
        # Randomly select 8 points
        sample_indices = np.random.choice(num_points, 8, replace=False)
        sample_left = leftPoints[sample_indices]
        sample_right = rightPoints[sample_indices]
        
        # Compute Fundamental Matrix for the sample
        F = eight_points_algorithm(sample_left.T, sample_right.T, normalize=True)
        
        # Calculate Sampson distance for all points
        # ones = np.ones((num_points, 1))
        # left_homogeneous = np.hstack((leftPoints, ones))
        # right_homogeneous = np.hstack((rightPoints, ones))
        
        # x2^T * F * x1
        errors = np.abs(np.sum(rightPoints @ F * leftPoints, axis=1))
        
        # Count inliers based on the threshold
        inliers = errors < good_threshold
        inlier_count = np.sum(inliers)
        
        # Update the best model if this one has more inliers
        if inlier_count > best_inlier_count:
            best_F = F
            best_inliers = inliers
            best_inlier_count = inlier_count
    
    return best_F, best_inliers


def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # TODO: Compute possible rotations and translations
    # Perform SVD on E
    U, _, Vt = np.linalg.svd(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # Compute possible rotations and translations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Ensure rotations have determinant +1
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Translations
    t1 = t.reshape(-1, 1)
    t2 = -t1

    # Four possibilities
    Pr = [np.concatenate((R1, t1), axis=1),
          np.concatenate((R1, t2), axis=1),
          np.concatenate((R2, t1), axis=1),
          np.concatenate((R2, t2), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = sparse.csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d
