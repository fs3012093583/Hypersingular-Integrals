"""Legendre product quadrature: original paper Eq.13-16, generalized to even m.

The weights act on network VALUES. No spatial network differentiation is used.
H_n(t)=PV integral P_n(s)/(s-t) ds; its Taylor coefficient of order m-1
is the m-th finite-part modified moment. All differentiation here concerns
known Legendre/Hilbert functions, and is evaluated offline by a recurrence.
"""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss


def legendre_values(x, n):
    x = np.asarray(x, dtype=np.longdouble)
    p = np.ones((len(x), n), dtype=np.longdouble)
    if n > 1:
        p[:, 1] = x
    for k in range(2, n):
        p[:, k] = ((2*k-1)*x*p[:, k-1]-(k-1)*p[:, k-2])/k
    return p


def modified_moments(t, m, n):
    t = np.asarray(t, dtype=np.longdouble)
    h0 = np.empty((len(t), m), dtype=np.longdouble)
    h0[:, 0] = np.log1p(-t)-np.log1p(t)
    for r in range(1, m):
        h0[:, r] = (-(1-t)**(-r)+(-1)**r*(1+t)**(-r))/r
    out = np.empty((len(t), n), dtype=np.longdouble)
    out[:, 0] = h0[:, -1]
    def times_t(jet):
        result = t[:, None]*jet
        result[:, 1:] += jet[:, :-1]
        return result
    if n == 1:
        return out
    h1 = times_t(h0)
    h1[:, 0] += 2
    out[:, 1] = h1[:, -1]
    for k in range(2, n):
        h2 = ((2*k-1)*times_t(h1)-(k-1)*h0)/k
        out[:, k] = h2[:, -1]
        h0, h1 = h1, h2
    return out


def matrix(m, t, nodes=192):
    x, w = leggauss(nodes)
    mu = modified_moments(t, m, nodes)
    p = legendre_values(x, nodes)
    weights = (mu*((2*np.arange(nodes)+1)/2).astype(np.longdouble)) @ p.T
    weights *= w.astype(np.longdouble)[None, :]
    return x, np.asarray(weights, dtype=np.float64)
