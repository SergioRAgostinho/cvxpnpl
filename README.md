# CvxPnPL

A convex Perspective-n-Points-and-Lines method.

**Title:** CvxPnPL: A Unified Convex Solution to the Absolute Pose Estimation Problem from Point and Line Correspondences

**Abstract:** We present a new convex method to estimate 3D pose from mixed combinations of 2D-3D point and line correspondences, the Perspective-n-Points-and-Lines problem (PnPL). We merge the contributions of each point and line into a unified Quadratic Constrained Quadratic Problem (QCQP) and then relax it into a Semi Definite Program (SDP) through Shor's relaxation. This makes it possible to gracefully handle mixed configurations of points and lines. Furthermore, the proposed relaxation allows us to recover a finite number of solutions under ambiguous configurations. In such cases, the 3D pose candidates are found by further enforcing geometric constraints on the solution space and then retrieving such poses from the intersections of multiple quadrics. Experiments provide results in line with the best performing state of the art methods while providing the flexibility of solving for an arbitrary number of points and lines.

**URL to the Paper:** TBA

**License:** Apache 2.0

## Installing

The easiest way to install the package is through pip
```
pip install cvxpnpl
```

Alternatively, you can clone this repo and invoke from its root folder
```
python setup.py install
```

## Examples

The library exposes 3 public functions: `pnp`, `pnl` and `pnpl`. You can find a couple of examples showing how to use each in the [examples folder](https://github.com/SergioRAgostinho/cvxpnpl/blob/master/examples).

## SDP Solver, BLAS and LAPACK

cvxpnpl makes use of [cvxpy](https://www.cvxpy.org/) as its opaque convex solver. However, cvxpy is only an abstraction layer and invokes [SCS](https://github.com/cvxgrp/scs) to obtain a solution to the underlying SDP problem. SCS requires BLAS and LAPACK which can be painful to set up for Windows users. With that in mind, I **really recommend that you install cvxpy through their anaconda channel**, as it will abstract away all of this dependency setup.
