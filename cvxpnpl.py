import numpy as np
import cvxpy as cvx

__version__ = "0.1.2"


def _point_constraints(pts_2d, pts_3d, K):
    """Build point constraints from 2D-3D correspondences.

    Arguments:
    pts_2d -- n x 2 np.array of 2D pixels
    pts_3d -- n x 3 np.array of 3D points
    K -- 3 x 3 np.array with the camera intrinsics
    """
    n = len(pts_3d)

    # Expand arguments
    # points in 2D
    px, py, pz = np.linalg.solve(K, np.vstack((pts_2d.T, np.ones(n))))

    # points in 3D
    Px, Py, Pz = pts_3d.T

    # Point Constraints
    Pxpx = Px * px
    Pxpy = Px * py
    Pxpz = Px * pz
    Pypx = Py * px
    Pypy = Py * py
    Pypz = Py * pz
    Pzpx = Pz * px
    Pzpy = Pz * py
    Pzpz = Pz * pz

    c11 = np.zeros(n)
    c12 = -Pxpz
    c13 = Pxpy
    c14 = np.zeros(n)
    c15 = -Pypz
    c16 = Pypy
    c17 = np.zeros(n)
    c18 = -Pzpz
    c19 = Pzpy

    c21 = Pxpz
    c22 = np.zeros(n)
    c23 = -Pxpx
    c24 = Pypz
    c25 = np.zeros(n)
    c26 = -Pypx
    c27 = Pzpz
    c28 = np.zeros(n)
    c29 = -Pzpx

    c31 = -Pxpy
    c32 = Pxpx
    c33 = np.zeros(n)
    c34 = -Pypy
    c35 = Pypx
    c36 = np.zeros(n)
    c37 = -Pzpy
    c38 = Pzpx
    c39 = np.zeros(n)

    n11 = np.zeros(n)
    n12 = -pz
    n13 = py

    n21 = pz
    n22 = np.zeros(n)
    n23 = -px

    n31 = -py
    n32 = px
    n33 = np.zeros(n)

    ## Compose block matrices for the equation system
    c1 = np.stack((c11, c12, c13, c14, c15, c16, c17, c18, c19), axis=1)
    c2 = np.stack((c21, c22, c23, c24, c25, c26, c27, c28, c29), axis=1)
    c3 = np.stack((c31, c32, c33, c34, c35, c36, c37, c38, c39), axis=1)

    n1 = np.stack((n11, n12, n13), axis=1)
    n2 = np.stack((n21, n22, n23), axis=1)
    n3 = np.stack((n31, n32, n33), axis=1)

    return (c1, c2, c3), (n1, n2, n3)


def _line_constraints(line_2d, line_3d, K):
    """Build line constraints from 2D-3D correspondences.

    Arguments:
    line_2d -- n x 2 x 2 np.array organized as (line, pt, dim). Each line is defined
    by sampling 2 points from it. Each point is a pixel in 2D.
    line_3d -- A 2-element tuple organized as (points, directions). Each line is
    parameterized by a 3D point and a 3D direction. The first element of the tuple
    is a n x 3 np.array of 3D points and second is a n x 3 np.array of 3D directions.
    K -- 3 x 3 np.array with the camera intrinsics.
    """
    n = len(line_2d)

    # line in 2D
    # has two sampled points.
    line_2d_c = np.linalg.solve(
        K, np.vstack((line_2d.reshape((2 * n, 2)).T, np.ones((1, 2 * n))))
    ).T
    line_2d_c = line_2d_c.reshape((n, 2, 3))

    # row wise cross product
    n_li = np.cross(line_2d_c[:, 0, :], line_2d_c[:, 1, :])

    # Normalize for stability
    n_li /= np.linalg.norm(n_li, axis=1)[:, None]
    nx, ny, nz = n_li.T

    # line in 3D
    PL, vL = line_3d
    PLx, PLy, PLz = PL.T
    vLx, vLy, vLz = vL.T

    # Line constraints - direction
    cl11 = vLx * nx
    cl12 = vLx * ny
    cl13 = vLx * nz
    cl14 = vLy * nx
    cl15 = vLy * ny
    cl16 = vLy * nz
    cl17 = vLz * nx
    cl18 = vLz * ny
    cl19 = vLz * nz

    # Line constraints - point
    cl21 = PLx * nx
    cl22 = PLx * ny
    cl23 = PLx * nz
    cl24 = PLy * nx
    cl25 = PLy * ny
    cl26 = PLy * nz
    cl27 = PLz * nx
    cl28 = PLz * ny
    cl29 = PLz * nz

    ## Compose block matrices for the equation system
    cl1 = np.stack((cl11, cl12, cl13, cl14, cl15, cl16, cl17, cl18, cl19), axis=1)
    cl2 = np.stack((cl21, cl22, cl23, cl24, cl25, cl26, cl27, cl28, cl29), axis=1)

    return (cl1, cl2), n_li


def _shor(A, eps=1e-9, max_iters=2500, verbose=False):
    """Solves the relaxed optimization problem with respect to Z, given the homogeneous
    linear system of the form A r = 0.

    Arguments:
    A -- the matrix defining the homogeneous linear system
    eps -- requested precision to the convex solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional information to the console
    """
    # Construct Q
    Q = np.block([[A.T @ A, np.zeros((9, 1))], [np.zeros((1, 9)), 0]])

    # Variables
    Z = cvx.Variable(shape=(10, 10), symmetric=True)

    # Constraints
    Pr = np.empty((6, 10, 10))
    Pc = np.empty((6, 10, 10))
    Pdet = np.empty((9, 10, 10))

    # Rows and cols
    k = 0
    for i in range(3):
        for j in range(i, 3):

            E_ij = np.zeros((3, 3))
            E_ij[i, j] = 1

            P = np.block(
                [
                    [np.kron(np.eye(3), E_ij.T), np.zeros((9, 1))],
                    [np.zeros(9), -int(i == j)],
                ]
            )
            Pr[k] = 0.5 * (P + P.T)

            P = np.block(
                [
                    [np.kron(E_ij, np.eye(3)), np.zeros((9, 1))],
                    [np.zeros(9), -int(i == j)],
                ]
            )
            Pc[k] = 0.5 * (P + P.T)
            k += 1

    # Determinant
    m = 0
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for l in range(3):

            E_ji = np.zeros((3, 3))
            E_ji[j, i] = 1

            e_l = np.zeros(3)
            e_l[l] = 1

            e_k = np.zeros(3)
            e_k[k] = 1

            S = np.array(
                [[0, -e_l[2], e_l[1]], [e_l[2], 0, -e_l[0]], [-e_l[1], e_l[0], 0]]
            )

            P = np.block(
                [[np.kron(E_ji, S), np.zeros((9, 1))], [-np.kron(e_k, e_l), 0]]
            )

            Pdet[m] = 0.5 * (P + P.T)
            m += 1

    # Compose all constraints
    constraints = [Z[9, 9] == 1, Z >> 0]
    for Pgroup in (Pr, Pc, Pdet):
        for P in Pgroup:
            constraints.append(cvx.trace(P * Z) == 0)

    # Invoke problem
    cost = cvx.Minimize(cvx.trace(Q * Z))
    prob = cvx.Problem(cost, constraints)
    prob.solve(eps=eps, max_iters=max_iters, verbose=verbose)
    return Z.value


def _re6q3(A):
    """Solves the E6Q3 problem i.e., the intersection of 6 quadrics with 3 unknowns.

    Arguments:
    A -- The N x 10 matrix with N >= 6. The columns are ordered as
    [a^2, b^2, c^2, ab, ac, bc, a, b, c, 1].
    """
    B = A[:, :6]
    C = A[:, 6:]
    Ap = np.block([np.eye(6), np.linalg.solve(B.T @ B, B.T @ C)])

    # Extract D
    D = -Ap[[1, 2, 5]][:, 6:]
    d0, d1, d2 = D
    d00, d01, d02, d03 = d0
    d10, d11, d12, d13 = d1
    d20, d21, d22, d23 = d2

    # Compose the polynomial coefficients
    # fmt: off
    p0 = d01**3*d11*d12*d22*d23 - d01**3*d11*d13*d22**2 + d01**3*d11*d23**2 - d01**2*d02*d11**2*d22*d23 + d01**2*d02*d11*d12**2*d23 - d01**2*d02*d11*d12*d13*d22 - d01**2*d02*d11*d12*d21*d23 + 2*d01**2*d02*d11*d13*d21*d22 + 2*d01**2*d02*d11*d13*d23 + d01**2*d03*d11**2*d22**2 - d01**2*d03*d11*d12*d21*d22 + d01**2*d03*d11*d12*d23 - 2*d01**2*d03*d11*d13*d22 - 2*d01**2*d03*d11*d21*d23 - 2*d01**2*d11*d12*d22**2*d23 + 2*d01**2*d11*d13*d22**3 - 2*d01**2*d11*d22*d23**2 + d01**2*d12**2*d21*d22*d23 - d01**2*d12*d13*d21*d22**2 + d01**2*d12*d13*d22*d23 - d01**2*d12*d21**2*d22*d23 + d01**2*d12*d21*d23**2 - d01**2*d13**2*d22**2 + d01**2*d13*d21**2*d22**2 + d01**2*d13*d23**2 - d01**2*d21**2*d23**2 - 2*d01*d02**2*d11**2*d12*d23 + d01*d02**2*d11**2*d13*d22 + d01*d02**2*d11**2*d21*d23 + d01*d02**2*d11*d12*d13*d21 + d01*d02**2*d11*d13**2 - d01*d02**2*d11*d13*d21**2 + d01*d02*d03*d11**2*d12*d22 - d01*d02*d03*d11**2*d21*d22 - 3*d01*d02*d03*d11**2*d23 - d01*d02*d03*d11*d12**2*d21 - d01*d02*d03*d11*d12*d13 + d01*d02*d03*d11*d12*d21**2 + 2*d01*d02*d11**2*d22**2*d23 - d01*d02*d11*d12**2*d22*d23 + d01*d02*d11*d12*d13*d22**2 + 2*d01*d02*d11*d12*d21*d22*d23 + d01*d02*d11*d12*d23**2 - 5*d01*d02*d11*d13*d21*d22**2 - 5*d01*d02*d11*d13*d22*d23 + d01*d02*d11*d21**2*d22*d23 + d01*d02*d11*d21*d23**2 + d01*d02*d12**3*d21*d23 - d01*d02*d12**2*d13*d21*d22 + d01*d02*d12**2*d13*d23 - 2*d01*d02*d12**2*d21**2*d23 - d01*d02*d12*d13**2*d22 + 3*d01*d02*d12*d13*d21**2*d22 + d01*d02*d12*d13*d21*d23 + d01*d02*d12*d21**3*d23 + 2*d01*d02*d13**2*d21*d22 + 2*d01*d02*d13**2*d23 - 2*d01*d02*d13*d21**3*d22 - 2*d01*d02*d13*d21**2*d23 + 2*d01*d03**2*d11**2*d22 - d01*d03**2*d11*d12*d21 - d01*d03**2*d11*d13 + d01*d03**2*d11*d21**2 - 2*d01*d03*d11**2*d22**3 + 3*d01*d03*d11*d12*d21*d22**2 + d01*d03*d11*d12*d22*d23 + 2*d01*d03*d11*d13*d22**2 - d01*d03*d11*d21**2*d22**2 + 4*d01*d03*d11*d21*d22*d23 + 3*d01*d03*d11*d23**2 - d01*d03*d12**2*d21**2*d22 + d01*d03*d12**2*d21*d23 - 3*d01*d03*d12*d13*d21*d22 + d01*d03*d12*d13*d23 + d01*d03*d12*d21**3*d22 - 3*d01*d03*d12*d21**2*d23 - 2*d01*d03*d13**2*d22 + 2*d01*d03*d13*d21**2*d22 - 2*d01*d03*d13*d21*d23 + 2*d01*d03*d21**3*d23 + d01*d11*d12*d22**3*d23 - d01*d11*d13*d22**4 + d01*d11*d22**2*d23**2 - d01*d12**2*d21*d22**2*d23 + d01*d12**2*d22*d23**2 + d01*d12*d13*d21*d22**3 - 3*d01*d12*d13*d22**2*d23 - 3*d01*d12*d21*d22*d23**2 + d01*d12*d23**3 + 2*d01*d13**2*d22**3 + 2*d01*d13*d21*d22**2*d23 - 2*d01*d13*d22*d23**2 - 2*d01*d21*d23**3 + d02**3*d11**3*d23 - d02**3*d11**2*d13*d21 - d02**2*d03*d11**3*d22 + d02**2*d03*d11**2*d12*d21 - d02**2*d03*d11**2*d13 + d02**2*d11**2*d12*d22*d23 - 3*d02**2*d11**2*d21*d22*d23 - 3*d02**2*d11**2*d23**2 - d02**2*d11*d12**2*d21*d23 - d02**2*d11*d12*d13*d21*d22 - 3*d02**2*d11*d12*d13*d23 + 2*d02**2*d11*d12*d21**2*d23 + d02**2*d11*d13**2*d22 + 3*d02**2*d11*d13*d21**2*d22 + 5*d02**2*d11*d13*d21*d23 - d02**2*d11*d21**3*d23 + d02**2*d12**2*d13*d21**2 + 2*d02**2*d12*d13**2*d21 - 2*d02**2*d12*d13*d21**3 + d02**2*d13**3 - 2*d02**2*d13**2*d21**2 + d02**2*d13*d21**4 + d02*d03**2*d11**2*d12 + d02*d03**2*d11**2*d21 - d02*d03*d11**2*d12*d22**2 + 3*d02*d03*d11**2*d21*d22**2 + 5*d02*d03*d11**2*d22*d23 + 2*d02*d03*d11*d12**2*d21*d22 + 2*d02*d03*d11*d12**2*d23 - 5*d02*d03*d11*d12*d21**2*d22 - 5*d02*d03*d11*d12*d21*d23 - d02*d03*d11*d13*d21*d22 - d02*d03*d11*d13*d23 + d02*d03*d11*d21**3*d22 + d02*d03*d11*d21**2*d23 - d02*d03*d12**3*d21**2 - 2*d02*d03*d12**2*d13*d21 + 2*d02*d03*d12**2*d21**3 - d02*d03*d12*d13**2 + 2*d02*d03*d12*d13*d21**2 - d02*d03*d12*d21**4 - d02*d11**2*d22**3*d23 + d02*d11*d12*d21*d22**2*d23 + d02*d11*d12*d22*d23**2 + d02*d11*d13*d21*d22**3 + d02*d11*d13*d22**2*d23 + 3*d02*d11*d21*d22*d23**2 + 3*d02*d11*d23**3 + d02*d12**3*d23**2 - 2*d02*d12**2*d13*d22*d23 - 2*d02*d12**2*d21*d23**2 + d02*d12*d13**2*d22**2 - d02*d12*d13*d21**2*d22**2 + 4*d02*d12*d13*d21*d22*d23 + 3*d02*d12*d13*d23**2 + d02*d12*d21**2*d23**2 - 4*d02*d13**2*d21*d22**2 - 4*d02*d13**2*d22*d23 - 4*d02*d13*d21**2*d22*d23 - 4*d02*d13*d21*d23**2 + d03**3*d11**2 - 2*d03**2*d11**2*d22**2 + 2*d03**2*d11*d12*d21*d22 + 2*d03**2*d11*d12*d23 - 4*d03**2*d11*d21**2*d22 - 4*d03**2*d11*d21*d23 - d03**2*d12**2*d21**2 - 2*d03**2*d12*d13*d21 + 2*d03**2*d12*d21**3 - d03**2*d13**2 + 2*d03**2*d13*d21**2 - d03**2*d21**4 + d03*d11**2*d22**4 - 2*d03*d11*d12*d21*d22**3 - 2*d03*d11*d12*d22**2*d23 - 4*d03*d11*d21*d22**2*d23 - 4*d03*d11*d22*d23**2 + d03*d12**2*d21**2*d22**2 + d03*d12**2*d23**2 + 2*d03*d12*d13*d21*d22**2 - 2*d03*d12*d13*d22*d23 + 2*d03*d12*d21**2*d22*d23 - 2*d03*d12*d21*d23**2 + 2*d03*d13**2*d22**2 + 2*d03*d13*d21**2*d22**2 + 8*d03*d13*d21*d22*d23 + 2*d03*d13*d23**2 + 2*d03*d21**2*d23**2 - d12**2*d22**2*d23**2 + 2*d12*d13*d22**3*d23 - 2*d12*d22*d23**3 - d13**2*d22**4 + 2*d13*d22**2*d23**2 - d23**4
    p1 = d00*d01**2*d11**2*d22**2 - d00*d01**2*d11*d12*d21*d22 + d00*d01**2*d11*d12*d23 - 2*d00*d01**2*d11*d13*d22 - 2*d00*d01**2*d11*d21*d23 + d00*d01*d02*d11**2*d12*d22 - d00*d01*d02*d11**2*d21*d22 - 3*d00*d01*d02*d11**2*d23 - d00*d01*d02*d11*d12**2*d21 - d00*d01*d02*d11*d12*d13 + d00*d01*d02*d11*d12*d21**2 + 4*d00*d01*d03*d11**2*d22 - 2*d00*d01*d03*d11*d12*d21 - 2*d00*d01*d03*d11*d13 + 2*d00*d01*d03*d11*d21**2 - 2*d00*d01*d11**2*d22**3 + 3*d00*d01*d11*d12*d21*d22**2 + d00*d01*d11*d12*d22*d23 + 2*d00*d01*d11*d13*d22**2 - d00*d01*d11*d21**2*d22**2 + 4*d00*d01*d11*d21*d22*d23 + 3*d00*d01*d11*d23**2 - d00*d01*d12**2*d21**2*d22 + d00*d01*d12**2*d21*d23 - 3*d00*d01*d12*d13*d21*d22 + d00*d01*d12*d13*d23 + d00*d01*d12*d21**3*d22 - 3*d00*d01*d12*d21**2*d23 - 2*d00*d01*d13**2*d22 + 2*d00*d01*d13*d21**2*d22 - 2*d00*d01*d13*d21*d23 + 2*d00*d01*d21**3*d23 - d00*d02**2*d11**3*d22 + d00*d02**2*d11**2*d12*d21 - d00*d02**2*d11**2*d13 + 2*d00*d02*d03*d11**2*d12 + 2*d00*d02*d03*d11**2*d21 - d00*d02*d11**2*d12*d22**2 + 3*d00*d02*d11**2*d21*d22**2 + 5*d00*d02*d11**2*d22*d23 + 2*d00*d02*d11*d12**2*d21*d22 + 2*d00*d02*d11*d12**2*d23 - 5*d00*d02*d11*d12*d21**2*d22 - 5*d00*d02*d11*d12*d21*d23 - d00*d02*d11*d13*d21*d22 - d00*d02*d11*d13*d23 + d00*d02*d11*d21**3*d22 + d00*d02*d11*d21**2*d23 - d00*d02*d12**3*d21**2 - 2*d00*d02*d12**2*d13*d21 + 2*d00*d02*d12**2*d21**3 - d00*d02*d12*d13**2 + 2*d00*d02*d12*d13*d21**2 - d00*d02*d12*d21**4 + 3*d00*d03**2*d11**2 - 4*d00*d03*d11**2*d22**2 + 4*d00*d03*d11*d12*d21*d22 + 4*d00*d03*d11*d12*d23 - 8*d00*d03*d11*d21**2*d22 - 8*d00*d03*d11*d21*d23 - 2*d00*d03*d12**2*d21**2 - 4*d00*d03*d12*d13*d21 + 4*d00*d03*d12*d21**3 - 2*d00*d03*d13**2 + 4*d00*d03*d13*d21**2 - 2*d00*d03*d21**4 + d00*d11**2*d22**4 - 2*d00*d11*d12*d21*d22**3 - 2*d00*d11*d12*d22**2*d23 - 4*d00*d11*d21*d22**2*d23 - 4*d00*d11*d22*d23**2 + d00*d12**2*d21**2*d22**2 + d00*d12**2*d23**2 + 2*d00*d12*d13*d21*d22**2 - 2*d00*d12*d13*d22*d23 + 2*d00*d12*d21**2*d22*d23 - 2*d00*d12*d21*d23**2 + 2*d00*d13**2*d22**2 + 2*d00*d13*d21**2*d22**2 + 8*d00*d13*d21*d22*d23 + 2*d00*d13*d23**2 + 2*d00*d21**2*d23**2 - d01**3*d10*d11*d22**2 + d01**3*d11*d12*d20*d22 + 2*d01**3*d11*d20*d23 - d01**2*d02*d10*d11*d12*d22 + 2*d01**2*d02*d10*d11*d21*d22 + 2*d01**2*d02*d10*d11*d23 - d01**2*d02*d11**2*d20*d22 + d01**2*d02*d11*d12**2*d20 - d01**2*d02*d11*d12*d20*d21 + 2*d01**2*d02*d11*d13*d20 - 2*d01**2*d03*d10*d11*d22 + d01**2*d03*d11*d12*d20 - 2*d01**2*d03*d11*d20*d21 + 2*d01**2*d10*d11*d22**3 - d01**2*d10*d12*d21*d22**2 + d01**2*d10*d12*d22*d23 - 2*d01**2*d10*d13*d22**2 + d01**2*d10*d21**2*d22**2 + d01**2*d10*d23**2 - 2*d01**2*d11*d12*d20*d22**2 - 4*d01**2*d11*d20*d22*d23 + d01**2*d12**2*d20*d21*d22 + d01**2*d12*d13*d20*d22 - d01**2*d12*d20*d21**2*d22 + 2*d01**2*d12*d20*d21*d23 + 2*d01**2*d13*d20*d23 - 2*d01**2*d20*d21**2*d23 + d01*d02**2*d10*d11**2*d22 + d01*d02**2*d10*d11*d12*d21 + 2*d01*d02**2*d10*d11*d13 - d01*d02**2*d10*d11*d21**2 - 2*d01*d02**2*d11**2*d12*d20 + d01*d02**2*d11**2*d20*d21 - d01*d02*d03*d10*d11*d12 - 3*d01*d02*d03*d11**2*d20 + d01*d02*d10*d11*d12*d22**2 - 5*d01*d02*d10*d11*d21*d22**2 - 5*d01*d02*d10*d11*d22*d23 - d01*d02*d10*d12**2*d21*d22 + d01*d02*d10*d12**2*d23 - 2*d01*d02*d10*d12*d13*d22 + 3*d01*d02*d10*d12*d21**2*d22 + d01*d02*d10*d12*d21*d23 + 4*d01*d02*d10*d13*d21*d22 + 4*d01*d02*d10*d13*d23 - 2*d01*d02*d10*d21**3*d22 - 2*d01*d02*d10*d21**2*d23 + 2*d01*d02*d11**2*d20*d22**2 - d01*d02*d11*d12**2*d20*d22 + 2*d01*d02*d11*d12*d20*d21*d22 + 2*d01*d02*d11*d12*d20*d23 - 5*d01*d02*d11*d13*d20*d22 + d01*d02*d11*d20*d21**2*d22 + 2*d01*d02*d11*d20*d21*d23 + d01*d02*d12**3*d20*d21 + d01*d02*d12**2*d13*d20 - 2*d01*d02*d12**2*d20*d21**2 + d01*d02*d12*d13*d20*d21 + d01*d02*d12*d20*d21**3 + 2*d01*d02*d13**2*d20 - 2*d01*d02*d13*d20*d21**2 - d01*d03**2*d10*d11 + 2*d01*d03*d10*d11*d22**2 - 3*d01*d03*d10*d12*d21*d22 + d01*d03*d10*d12*d23 - 4*d01*d03*d10*d13*d22 + 2*d01*d03*d10*d21**2*d22 - 2*d01*d03*d10*d21*d23 + d01*d03*d11*d12*d20*d22 + 4*d01*d03*d11*d20*d21*d22 + 6*d01*d03*d11*d20*d23 + d01*d03*d12**2*d20*d21 + d01*d03*d12*d13*d20 - 3*d01*d03*d12*d20*d21**2 - 2*d01*d03*d13*d20*d21 + 2*d01*d03*d20*d21**3 - d01*d10*d11*d22**4 + d01*d10*d12*d21*d22**3 - 3*d01*d10*d12*d22**2*d23 + 4*d01*d10*d13*d22**3 + 2*d01*d10*d21*d22**2*d23 - 2*d01*d10*d22*d23**2 + d01*d11*d12*d20*d22**3 + 2*d01*d11*d20*d22**2*d23 - d01*d12**2*d20*d21*d22**2 + 2*d01*d12**2*d20*d22*d23 - 3*d01*d12*d13*d20*d22**2 - 6*d01*d12*d20*d21*d22*d23 + 3*d01*d12*d20*d23**2 + 2*d01*d13*d20*d21*d22**2 - 4*d01*d13*d20*d22*d23 - 6*d01*d20*d21*d23**2 - d02**3*d10*d11**2*d21 + d02**3*d11**3*d20 - d02**2*d03*d10*d11**2 - d02**2*d10*d11*d12*d21*d22 - 3*d02**2*d10*d11*d12*d23 + 2*d02**2*d10*d11*d13*d22 + 3*d02**2*d10*d11*d21**2*d22 + 5*d02**2*d10*d11*d21*d23 + d02**2*d10*d12**2*d21**2 + 4*d02**2*d10*d12*d13*d21 - 2*d02**2*d10*d12*d21**3 + 3*d02**2*d10*d13**2 - 4*d02**2*d10*d13*d21**2 + d02**2*d10*d21**4 + d02**2*d11**2*d12*d20*d22 - 3*d02**2*d11**2*d20*d21*d22 - 6*d02**2*d11**2*d20*d23 - d02**2*d11*d12**2*d20*d21 - 3*d02**2*d11*d12*d13*d20 + 2*d02**2*d11*d12*d20*d21**2 + 5*d02**2*d11*d13*d20*d21 - d02**2*d11*d20*d21**3 - d02*d03*d10*d11*d21*d22 - d02*d03*d10*d11*d23 - 2*d02*d03*d10*d12**2*d21 - 2*d02*d03*d10*d12*d13 + 2*d02*d03*d10*d12*d21**2 + 5*d02*d03*d11**2*d20*d22 + 2*d02*d03*d11*d12**2*d20 - 5*d02*d03*d11*d12*d20*d21 - d02*d03*d11*d13*d20 + d02*d03*d11*d20*d21**2 + d02*d10*d11*d21*d22**3 + d02*d10*d11*d22**2*d23 - 2*d02*d10*d12**2*d22*d23 + 2*d02*d10*d12*d13*d22**2 - d02*d10*d12*d21**2*d22**2 + 4*d02*d10*d12*d21*d22*d23 + 3*d02*d10*d12*d23**2 - 8*d02*d10*d13*d21*d22**2 - 8*d02*d10*d13*d22*d23 - 4*d02*d10*d21**2*d22*d23 - 4*d02*d10*d21*d23**2 - d02*d11**2*d20*d22**3 + d02*d11*d12*d20*d21*d22**2 + 2*d02*d11*d12*d20*d22*d23 + d02*d11*d13*d20*d22**2 + 6*d02*d11*d20*d21*d22*d23 + 9*d02*d11*d20*d23**2 + 2*d02*d12**3*d20*d23 - 2*d02*d12**2*d13*d20*d22 - 4*d02*d12**2*d20*d21*d23 + 4*d02*d12*d13*d20*d21*d22 + 6*d02*d12*d13*d20*d23 + 2*d02*d12*d20*d21**2*d23 - 4*d02*d13**2*d20*d22 - 4*d02*d13*d20*d21**2*d22 - 8*d02*d13*d20*d21*d23 - 2*d03**2*d10*d12*d21 - 2*d03**2*d10*d13 + 2*d03**2*d10*d21**2 + 2*d03**2*d11*d12*d20 - 4*d03**2*d11*d20*d21 + 2*d03*d10*d12*d21*d22**2 - 2*d03*d10*d12*d22*d23 + 4*d03*d10*d13*d22**2 + 2*d03*d10*d21**2*d22**2 + 8*d03*d10*d21*d22*d23 + 2*d03*d10*d23**2 - 2*d03*d11*d12*d20*d22**2 - 4*d03*d11*d20*d21*d22**2 - 8*d03*d11*d20*d22*d23 + 2*d03*d12**2*d20*d23 - 2*d03*d12*d13*d20*d22 + 2*d03*d12*d20*d21**2*d22 - 4*d03*d12*d20*d21*d23 + 8*d03*d13*d20*d21*d22 + 4*d03*d13*d20*d23 + 4*d03*d20*d21**2*d23 + 2*d10*d12*d22**3*d23 - 2*d10*d13*d22**4 + 2*d10*d22**2*d23**2 - 2*d12**2*d20*d22**2*d23 + 2*d12*d13*d20*d22**3 - 6*d12*d20*d22*d23**2 + 4*d13*d20*d22**2*d23 - 4*d20*d23**3
    p2 = 2*d00**2*d01*d11**2*d22 - d00**2*d01*d11*d12*d21 - d00**2*d01*d11*d13 + d00**2*d01*d11*d21**2 + d00**2*d02*d11**2*d12 + d00**2*d02*d11**2*d21 + 3*d00**2*d03*d11**2 - 2*d00**2*d11**2*d22**2 + 2*d00**2*d11*d12*d21*d22 + 2*d00**2*d11*d12*d23 - 4*d00**2*d11*d21**2*d22 - 4*d00**2*d11*d21*d23 - d00**2*d12**2*d21**2 - 2*d00**2*d12*d13*d21 + 2*d00**2*d12*d21**3 - d00**2*d13**2 + 2*d00**2*d13*d21**2 - d00**2*d21**4 - 2*d00*d01**2*d10*d11*d22 + d00*d01**2*d11*d12*d20 - 2*d00*d01**2*d11*d20*d21 - d00*d01*d02*d10*d11*d12 - 3*d00*d01*d02*d11**2*d20 - 2*d00*d01*d03*d10*d11 + 2*d00*d01*d10*d11*d22**2 - 3*d00*d01*d10*d12*d21*d22 + d00*d01*d10*d12*d23 - 4*d00*d01*d10*d13*d22 + 2*d00*d01*d10*d21**2*d22 - 2*d00*d01*d10*d21*d23 + d00*d01*d11*d12*d20*d22 + 4*d00*d01*d11*d20*d21*d22 + 6*d00*d01*d11*d20*d23 + d00*d01*d12**2*d20*d21 + d00*d01*d12*d13*d20 - 3*d00*d01*d12*d20*d21**2 - 2*d00*d01*d13*d20*d21 + 2*d00*d01*d20*d21**3 - d00*d02**2*d10*d11**2 - d00*d02*d10*d11*d21*d22 - d00*d02*d10*d11*d23 - 2*d00*d02*d10*d12**2*d21 - 2*d00*d02*d10*d12*d13 + 2*d00*d02*d10*d12*d21**2 + 5*d00*d02*d11**2*d20*d22 + 2*d00*d02*d11*d12**2*d20 - 5*d00*d02*d11*d12*d20*d21 - d00*d02*d11*d13*d20 + d00*d02*d11*d20*d21**2 - 4*d00*d03*d10*d12*d21 - 4*d00*d03*d10*d13 + 4*d00*d03*d10*d21**2 + 4*d00*d03*d11*d12*d20 - 8*d00*d03*d11*d20*d21 + 2*d00*d10*d12*d21*d22**2 - 2*d00*d10*d12*d22*d23 + 4*d00*d10*d13*d22**2 + 2*d00*d10*d21**2*d22**2 + 8*d00*d10*d21*d22*d23 + 2*d00*d10*d23**2 - 2*d00*d11*d12*d20*d22**2 - 4*d00*d11*d20*d21*d22**2 - 8*d00*d11*d20*d22*d23 + 2*d00*d12**2*d20*d23 - 2*d00*d12*d13*d20*d22 + 2*d00*d12*d20*d21**2*d22 - 4*d00*d12*d20*d21*d23 + 8*d00*d13*d20*d21*d22 + 4*d00*d13*d20*d23 + 4*d00*d20*d21**2*d23 + d01**3*d11*d20**2 + 2*d01**2*d02*d10*d11*d20 - d01**2*d10**2*d22**2 + d01**2*d10*d12*d20*d22 + 2*d01**2*d10*d20*d23 - 2*d01**2*d11*d20**2*d22 + d01**2*d12*d20**2*d21 + d01**2*d13*d20**2 - d01**2*d20**2*d21**2 + d01*d02**2*d10**2*d11 - d01*d02*d10**2*d12*d22 + 2*d01*d02*d10**2*d21*d22 + 2*d01*d02*d10**2*d23 - 5*d01*d02*d10*d11*d20*d22 + d01*d02*d10*d12**2*d20 + d01*d02*d10*d12*d20*d21 + 4*d01*d02*d10*d13*d20 - 2*d01*d02*d10*d20*d21**2 + d01*d02*d11*d12*d20**2 + d01*d02*d11*d20**2*d21 - 2*d01*d03*d10**2*d22 + d01*d03*d10*d12*d20 - 2*d01*d03*d10*d20*d21 + 3*d01*d03*d11*d20**2 + 2*d01*d10**2*d22**3 - 3*d01*d10*d12*d20*d22**2 + 2*d01*d10*d20*d21*d22**2 - 4*d01*d10*d20*d22*d23 + d01*d11*d20**2*d22**2 + d01*d12**2*d20**2*d22 - 3*d01*d12*d20**2*d21*d22 + 3*d01*d12*d20**2*d23 - 2*d01*d13*d20**2*d22 - 6*d01*d20**2*d21*d23 + d02**2*d10**2*d11*d22 + 2*d02**2*d10**2*d12*d21 + 3*d02**2*d10**2*d13 - 2*d02**2*d10**2*d21**2 - 3*d02**2*d10*d11*d12*d20 + 5*d02**2*d10*d11*d20*d21 - 3*d02**2*d11**2*d20**2 - d02*d03*d10**2*d12 - d02*d03*d10*d11*d20 + d02*d10**2*d12*d22**2 - 4*d02*d10**2*d21*d22**2 - 4*d02*d10**2*d22*d23 + d02*d10*d11*d20*d22**2 - 2*d02*d10*d12**2*d20*d22 + 4*d02*d10*d12*d20*d21*d22 + 6*d02*d10*d12*d20*d23 - 8*d02*d10*d13*d20*d22 - 4*d02*d10*d20*d21**2*d22 - 8*d02*d10*d20*d21*d23 + d02*d11*d12*d20**2*d22 + 3*d02*d11*d20**2*d21*d22 + 9*d02*d11*d20**2*d23 + d02*d12**3*d20**2 - 2*d02*d12**2*d20**2*d21 + 3*d02*d12*d13*d20**2 + d02*d12*d20**2*d21**2 - 4*d02*d13*d20**2*d21 - d03**2*d10**2 + 2*d03*d10**2*d22**2 - 2*d03*d10*d12*d20*d22 + 8*d03*d10*d20*d21*d22 + 4*d03*d10*d20*d23 - 4*d03*d11*d20**2*d22 + d03*d12**2*d20**2 - 2*d03*d12*d20**2*d21 + 2*d03*d13*d20**2 + 2*d03*d20**2*d21**2 - d10**2*d22**4 + 2*d10*d12*d20*d22**3 + 4*d10*d20*d22**2*d23 - d12**2*d20**2*d22**2 - 6*d12*d20**2*d22*d23 + 2*d13*d20**2*d22**2 - 6*d20**2*d23**2
    p3 = d00**3*d11**2 - d00**2*d01*d10*d11 - 2*d00**2*d10*d12*d21 - 2*d00**2*d10*d13 + 2*d00**2*d10*d21**2 + 2*d00**2*d11*d12*d20 - 4*d00**2*d11*d20*d21 - 2*d00*d01*d10**2*d22 + d00*d01*d10*d12*d20 - 2*d00*d01*d10*d20*d21 + 3*d00*d01*d11*d20**2 - d00*d02*d10**2*d12 - d00*d02*d10*d11*d20 - 2*d00*d03*d10**2 + 2*d00*d10**2*d22**2 - 2*d00*d10*d12*d20*d22 + 8*d00*d10*d20*d21*d22 + 4*d00*d10*d20*d23 - 4*d00*d11*d20**2*d22 + d00*d12**2*d20**2 - 2*d00*d12*d20**2*d21 + 2*d00*d13*d20**2 + 2*d00*d20**2*d21**2 + d01**2*d10*d20**2 + 2*d01*d02*d10**2*d20 - 2*d01*d10*d20**2*d22 + d01*d12*d20**3 - 2*d01*d20**3*d21 + d02**2*d10**3 - 4*d02*d10**2*d20*d22 + 3*d02*d10*d12*d20**2 - 4*d02*d10*d20**2*d21 + 3*d02*d11*d20**3 + 2*d03*d10*d20**2 + 2*d10*d20**2*d22**2 - 2*d12*d20**3*d22 - 4*d20**3*d23
    p4 = -d00**2*d10**2 + 2*d00*d10*d20**2 - d20**4
    # fmt: on

    # Compute the polynomial roots
    roots = np.roots((p4, p3, p2, p1, p0))
    a = np.real(roots)

    # Construct M from identities
    # fmt: off
    m00 = -a*d20 + d02*d11 - d21*d22 - d23
    m01 = a*d00 + d01*d22 + d02*d12 - d02*d21 + d03 - d22**2
    m02 = a*(-d00*d21 + d01*d20 + d02*d10 - d20*d22) + d01*d23 + d02*d13 - d03*d21 - d22*d23

    m10 = a*d10 + d01*d11 - d11*d22 + d12*d21 + d13 - d21**2
    m11 = -a*d20 + d02*d11 - d21*d22 - d23
    m12 = a*(d00*d11 - d10*d22 + d12*d20 - d20*d21) + d03*d11 + d12*d23 - d13*d22 - d21*d23

    m20 = a*(d00*d11 + d01*d10 - 2*d20*d21) + d01**2*d11 + d01*d12*d21 + d01*d13 - d01*d21**2 + d02*d11*d12 + d02*d11*d21 + d03*d11 - d11*d22**2 - 2*d21**2*d22 - 2*d21*d23
    m21 = a*(d00*d12 + d02*d10 - 2*d20*d22) + d01*d02*d11 + d01*d12*d22 + d02*d11*d22 + d02*d12**2 + d02*d13 - d02*d21**2 + d03*d12 - d12*d22**2 - 2*d21*d22**2 - 2*d22*d23
    m22 = a**2*(d00*d10 - d20**2) + a*(d00*d01*d11 + d00*d13 - d00*d21**2 + d01*d12*d20 + d02*d10*d12 + d02*d11*d20 + d03*d10 - d10*d22**2 - 2*d20*d21*d22 - 2*d20*d23) + d01*d03*d11 + d01*d12*d23 + d02*d11*d23 + d02*d12*d13 + d03*d13 - d03*d21**2 - d13*d22**2 - 2*d21*d22*d23 - d23**2
    # fmt: on

    M = np.block(
        [
            [m00[:, None, None], m01[:, None, None], m02[:, None, None]],
            [m10[:, None, None], m11[:, None, None], m12[:, None, None]],
            [m20[:, None, None], m21[:, None, None], m22[:, None, None]],
        ]
    )

    # Extract b and c
    bc = -np.linalg.solve(
        M[:, :, :2].transpose((0, 2, 1)) @ M[:, :, :2],
        M[:, :, :2].transpose((0, 2, 1)) @ M[:, :, 2, None],
    )
    b, c = bc.reshape((-1, 2)).T

    return a, b, c


def _constraint_ortho_det(vecs, rank):
    """Estimate poses possible rotation solutions from enforcing geometric constraints
    on the solution space.

    Arguments:
    vecs -- the eigenvectors from Z, stacked in columns
    rank -- the rank of Z
    """
    # We can only handle up to rank 4.
    # rank ceils into the next
    _rank = min(int(np.ceil(rank / 2) * 2), 4)

    # Marginalize
    V = vecs[:, -_rank:].T
    v0 = V[-1] / V[-1, -1]
    V = np.vstack((V[:-1] - np.outer(V[:-1, -1], v0), v0)).T[:-1]

    # Row and Col constraints norm and orthogonality constraints
    k = 0
    Pc = np.zeros((6, _rank, _rank))
    Pr = np.zeros((6, _rank, _rank))
    for i in range(3):
        for j in range(i, 3):

            e_i = np.zeros(3)
            e_i[i] = 1

            e_j = np.zeros(3)
            e_j[j] = 1

            Vci = np.kron(e_i, np.eye(3)) @ V
            Vcj = np.kron(e_j, np.eye(3)) @ V

            Vri = np.kron(np.eye(3), e_i) @ V
            Vrj = np.kron(np.eye(3), e_j) @ V

            K = np.diag(np.zeros(_rank))
            K[-1, -1] = int(i == j)

            P = Vci.T @ Vcj - K
            Pc[k] = 0.5 * (P + P.T)

            P = Vri.T @ Vrj - K
            Pr[k] = 0.5 * (P + P.T)
            k += 1

    # Determinant constraints
    m = 0
    Pdet = np.zeros((9, _rank, _rank))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for l in range(3):

            e_i = np.zeros(3)
            e_i[i] = 1

            e_j = np.zeros(3)
            e_j[j] = 1

            e_k = np.zeros(3)
            e_k[k] = 1

            e_l = np.zeros(3)
            e_l[l] = 1

            S = np.array(
                [[0, -e_l[2], e_l[1]], [e_l[2], 0, -e_l[0]], [-e_l[1], e_l[0], 0]]
            )

            Vci = np.kron(e_i, np.eye(3)) @ V
            Vcj = np.kron(e_j, np.eye(3)) @ V
            Vck = np.kron(e_k, np.eye(3)) @ V

            P = Vcj.T @ S @ Vci - np.block(
                [[np.zeros((_rank - 1, _rank))], [e_l @ Vck]]
            )

            Pdet[m] = 0.5 * (P + P.T)
            m += 1

    # Construct equation system
    P = np.concatenate((Pc, Pr, Pdet), axis=0)
    alpha_d = None
    if _rank == 2:
        A = np.array([P[:, 0, 0], 2 * P[:, 0, 1], P[:, 1, 1]]).T  # a**2, a, 1

        # Condense everything and solve the general quadratic formula
        coeffs = np.mean(A, axis=0)
        a = np.empty(2)

        root = np.real(np.sqrt(coeffs[1] * coeffs[1] - 4 * coeffs[0] * coeffs[2]))
        a[0] = (-coeffs[1] + root) / (2 * coeffs[0])
        a[1] = (-coeffs[1] - root) / (2 * coeffs[0])
        alpha_d = np.stack([a, np.ones(2)], axis=1)

    elif _rank == 4:
        A = np.array(
            [
                P[:, 0, 0],  # a**2
                P[:, 1, 1],  # b**2
                P[:, 2, 2],  # c**2
                2 * P[:, 0, 1],  # ab
                2 * P[:, 0, 2],  # ac
                2 * P[:, 1, 2],  # bc
                2 * P[:, 0, 3],  # a
                2 * P[:, 1, 3],  # b
                2 * P[:, 2, 3],  # c
                P[:, 3, 3],  # 1
            ]
        ).T

        # Solve the 6Q3 system
        a, b, c = _re6q3(A)

        # Return all candidates
        m = len(a)
        alpha_d = np.stack([a, b, c, np.ones(m)], axis=1)

    else:
        raise NotImplementedError

    return alpha_d @ V.T


def _solve_relaxation(A, B, eps=1e-9, max_iters=2500, verbose=False):
    """Given the linear system formed by the problem's geometric constraints,
    computes all possible poses.

    Arguments:
    A -- the matrix defining the homogeneous linear system formed from the problem's
    geometric constraints, such that A r = 0.
    B -- an auxiliary matrix which allows to retrieve the translation vector
    from an optimal rotation t = - B @ r.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional information to the console
    """
    # Solve the QCQP using shor's relaxation
    Z = _shor(A, eps=eps, max_iters=max_iters, verbose=verbose)
    vals, vecs = np.linalg.eigh(Z)

    # check for rank
    rank = np.sum(vals > 1e-3)
    r_c = None
    if rank == 1:
        r_c = (vecs[:-1, -1] / vecs[-1, -1])[None, :]
    else:
        r_c = _constraint_ortho_det(vecs, rank)

    poses = []
    for r in r_c:
        # Retrieve rotation and translation
        U, _, Vh = np.linalg.svd(r.reshape((3, 3)).T)
        R = U @ Vh
        t = -B @ R.ravel("F")
        poses.append((R, t))
    return poses


def pnp(pts_2d, pts_3d, K, eps=1e-9, max_iters=2500, verbose=False):
    """Compute object poses from point 2D-3D correspondences.

    Arguments:
    pts_2d -- n x 2 np.array of 2D pixels
    pts_3d -- n x 3 np.array of 3D points
    K -- 3 x 3 np.array with the camera intrinsics
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    (C1, C2, C3), (N1, N2, N3) = _point_constraints(pts_2d, pts_3d, K)

    # Compose block matrices
    C = np.vstack((C1, C2, C3))
    N = np.vstack((N1, N2, N3))

    B = np.linalg.solve(N.T @ N, N.T) @ C
    A = C - N @ B

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


def pnl(line_2d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False):
    """Compute object poses from line 2D-3D correspondences.

    Arguments:
    line_2d -- n x 2 x 2 np.array organized as (line, pt, dim). Each line is defined
    by sampling 2 points from it. Each point is a pixel in 2D.
    line_3d -- A 2-element tuple organized as (points, directions). Each line is
    parameterized by a 3D point and a 3D direction. The first element of the tuple
    is a n x 3 np.array of 3D points and second is a n x 3 np.array of 3D directions.
    K -- 3 x 3 np.array with the camera intrinsics.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    (C1, C2), N = _line_constraints(line_2d, line_3d, K)

    # Compose block matrices
    B = np.linalg.solve(N.T @ N, N.T) @ C2
    A = np.vstack((C1, C2 - N @ B))

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation(A, B, eps=eps, max_iters=max_iters, verbose=verbose)


def pnpl(pts_2d, line_2d, pts_3d, line_3d, K, eps=1e-9, max_iters=2500, verbose=False):
    """Compute object poses from point and line 2D-3D correspondences.

    Arguments:
    pts_2d -- n x 2 np.array of 2D pixels
    line_2d -- n x 2 x 2 np.array organized as (line, pt, dim). Each line is defined
    by sampling 2 points from it. Each point is a pixel in 2D.
    pts_3d -- n x 3 np.array of 3D points
    line_3d -- A 2-element tuple organized as (points, directions). Each line is
    parameterized by a 3D point and a 3D direction. The first element of the tuple
    is a n x 3 np.array of 3D points and second is a n x 3 np.array of 3D directions.
    K -- 3 x 3 np.array with the camera intrinsics.
    eps -- numerical precision of the solver
    max_iters -- maximum number of iterations the solver is allowed to perform
    verbose -- print additional solver information to the console
    """
    # Extract point constraints
    (Cp1, Cp2, Cp3), (Np1, Np2, Np3) = _point_constraints(
        pts_2d=pts_2d.reshape((-1, 2)), pts_3d=pts_3d.reshape((-1, 3)), K=K
    )

    # Extract line constraints
    (Cl1, Cl2), Nl = _line_constraints(line_2d.reshape((-1, 2, 2)), line_3d, K)

    # Compose block matrices
    C = np.vstack((Cp1, Cp2, Cp3, Cl2))
    N = np.vstack((Np1, Np2, Np3, Nl))

    # Compose block matrices
    B = np.linalg.solve(N.T @ N, N.T) @ C
    A = np.vstack((Cl1, C - N @ B))

    # Solve the QCQP using shor's relaxation
    return _solve_relaxation(A, B, eps=eps, max_iters=max_iters, verbose=verbose)
