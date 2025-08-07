"""
This code is taken and adapted from Mallasto's notebook on WGP.

https://colab.research.google.com/drive/1gCpSOoZ6odPmDQxYwki4koYqciaolRQw#scrollTo=Kx5ruxXOipeB
"""

from typing import Callable
import numpy as np
from scipy.linalg import logm, expm, sqrtm, inv, norm
from functools import reduce


class Manifold:
    # The tangent space will always be handled as R^d. For example, in the
    # case of positive definite matrices, when providing the exponential map
    # the map should be R^d -> space of symmetric matrices -> space of PD matrices

    def __init__(self, dimension, innerProduct, distance, exponential, logarithm):
        # [dimension] - Dimension of the manifold
        # [innerProduct] - Riemannian metric on the manifold
        # [distance] - Riemannian length distance
        # [exponential] - Riemannian exponential map
        # [logarithm] - Riemannian logarithm map
        self.dim = dimension
        self.g = innerProduct
        self.d = distance
        self.Exp = exponential
        self.Log = logarithm

    def norm(self, p, v):
        return np.sqrt(self.g(p, v, v))

    def FrechetMean(self, pop, weights=None, err=10 ** (-6)):
        # Compute the Frechet Mean of population [pop] with weights [weights].
        print("Computing the Frechet mean and printing the loss")
        N = len(pop)
        if weights is None:
            w = np.ones(N) / N
        else:
            w = weights

        p0 = pop[0]
        tmp = np.inf
        while tmp > err:
            logmaps = [w[i] * self.Log(p0, pop[i]) for i in range(N)]
            dx = reduce(lambda x, y: x + y, logmaps)
            tmp = self.norm(p0, dx)
            print(tmp)
            p0 = self.Exp(p0, dx)

        return p0


class PD(Manifold):
    # Manifold of positive definite matrices
    errorMargin = 10 ** (-6)

    def __init__(self, NonManiDim, metric):

        ManiDim = NonManiDim * (NonManiDim + 1) // 2
        self.elementDim = NonManiDim

        if metric == "LogEuclidean":
            Manifold.__init__(
                self,
                ManiDim,
                lambda p, v, u: self.gLE(p, v, u),
                lambda p, q: self.dLE(p, q),
                lambda p, v: self.ExpLE(p, v),
                lambda p, q: self.LogLE(p, q),
            )
            self.metric = "LogEuclidean"

        elif metric == "AffineInvariant":
            Manifold.__init__(
                self,
                ManiDim,
                lambda p, v, u: self.gAI(p, v, u),
                lambda p, q: self.dAI(p, q),
                lambda p, v: self.ExpAI(p, v),
                lambda p, q: self.LogAI(p, q),
            )
            self.metric = "AffineInvariant"

        else:
            print("Provide a metric: LogEuclidean, AffineInvariant or Wasserstein")

    # *********AFFINE-INVARIANT***********
    # Pennec, X., Fillard, P., & Ayache, N. (2006).
    # A Riemannian framework for tensor computing.
    # International Journal of Computer Vision, 66(1), 41-66.

    def gAI(self, p, v, u):
        v = self.VecToMat(v)
        u = self.VecToMat(u)

        # invsqrtp = inv(sqrtm(p))
        invp = inv(p)

        return (invp.dot(v).dot(invp).dot(u)).trace()

    def dAI(self, p, q):
        invsqrtp = inv(sqrtm(p))

        return norm(logm(invsqrtp.dot(q).dot(invsqrtp)), "fro")

    def ExpAI(self, p, v):
        if self.norm(p, v) < self.errorMargin:
            return p

        v = self.VecToMat(v)

        sqrtp = sqrtm(p)
        invsqrtp = inv(sqrtp)

        return sqrtp.dot(expm(invsqrtp.dot(v).dot(invsqrtp))).dot(sqrtp)

    def LogAI(self, p, q):
        if self.dAI(p, q) < self.errorMargin:
            return np.zeros(self.dim)

        sqrtp = sqrtm(p)
        invsqrtp = inv(sqrtp)

        v = sqrtp.dot(logm(invsqrtp.dot(q).dot(invsqrtp))).dot(sqrtp)

        return self.MatToVec(v)

    # *********LOG-EUCLIDEAN***********
    # Arsigny, V., Fillard, P., Pennec, X., & Ayache, N. (2006).
    # Log‐Euclidean metrics for fast and simple calculus on diffusion tensors.
    # Magnetic resonance in medicine, 56(2), 411-421.

    def gLE(self, p, v, u):
        v = self.VecToMat(v)
        u = self.VecToMat(u)

        return ((v.T).dot(u)).trace()

    def dLE(self, p, q):

        return norm(logm(p) - logm(q), "fro")

    def ExpLE(self, p, v):
        if self.norm(p, v) < self.errorMargin:
            return p
        if len(v.shape) > 1:
            v = np.array([self.VecToMat(v_i) for v_i in v])
            return expm(logm(p).reshape(1, self.elementDim, self.elementDim) + v)

        v = self.VecToMat(v)
        return expm(logm(p) + v)

    def LogLE(self, p, q):
        # if self.dLE(p, q) < self.errorMargin:
        #     return np.zeros(self.dim)

        v = logm(q) - logm(p)

        return self.MatToVec(v)

    # *********UTILITY FUNCTIONS***********
    def VecToMat(self, v):
        n = self.elementDim
        M = np.zeros([n, n])
        k = 0
        if len(v.shape) > 1:
            return v
        for i in range(n):
            for j in range(i, n):
                M[i, j] = v[k]
                M[j, i] = M[i, j]
                k += 1

        return M

    def MatToVec(self, M):
        n = self.dim
        m = self.elementDim
        v = np.zeros(n)
        k = 0

        for i in range(m):
            for j in range(i, m):
                v[k] = M[i, j]
                k += 1

        return v

    def tangent_space_regression(
        self, X, Y, initial_guess=None
    ) -> Callable[[np.ndarray], np.ndarray]:
        # Geodesic regression on the manifold
        # X: independent variables (e.g., time points)
        # Y: dependent variables (e.g., diffusion tensors)
        from sklearn.linear_model import LinearRegression

        frechet_mean = self.FrechetMean(Y)

        tangent_space_Y = np.array([self.LogLE(frechet_mean, y) for y in Y])

        reg = LinearRegression()
        reg.fit(X.reshape(-1, 1), tangent_space_Y)

        def predict(x):
            return self.ExpLE(frechet_mean, reg.predict(x.reshape(-1, 1)))

        return predict
