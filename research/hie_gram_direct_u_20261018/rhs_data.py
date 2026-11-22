"""Known-RHS endpoint finite-part moments, independent of the online map.

The regular remainder and endpoint poles of f are integrated separately.
An exact solution is used ONLY in verification_reference(), never build_b().
All polynomial moment sums are evaluated with arbitrary precision before
conversion to float64. No scipy dependency is required on the training host.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss


def norm(ell, k):
    return mp.pi*mp.power(2, 1-2*ell)*mp.gamma(k+2*ell)/(mp.factorial(k)*(k+ell)*mp.gamma(ell)**2)


def polys(ell, modes):
    out = [[mp.mpf(1)]]
    if modes > 1:
        out.append([mp.mpf(0), mp.mpf(2*ell)])
    for k in range(2, modes):
        c = [mp.mpf(0)]*(k+1)
        for j, v in enumerate(out[-1]):
            c[j+1] += mp.mpf(2*(k+ell-1))/k*v
        for j, v in enumerate(out[-2]):
            c[j] -= mp.mpf(k+2*ell-2)/k*v
        out.append(c)
    return [[v/mp.sqrt(norm(ell, k)) for v in c] for k, c in enumerate(out)]


def basis(x, modes, ell):
    x = np.asarray(x)
    out = np.empty((len(x), modes))
    out[:, 0] = 1
    if modes > 1:
        out[:, 1] = 2*ell*x
    for k in range(2, modes):
        out[:, k] = (2*(k+ell-1)*x*out[:, k-1]-(k+2*ell-2)*out[:, k-2])/k
    ns = np.exp([math.log(math.pi)+(1-2*ell)*math.log(2)+math.lgamma(k+2*ell)
                 -math.lgamma(k+1)-math.log(k+ell)-2*math.lgamma(ell) for k in range(modes)])
    return out/np.sqrt(ns)


def exact(x, family):
    return np.exp(x) if family == 'exp' else np.exp(.5*x)+.2*np.sin(6*np.pi*x)


def exponential_coeff(z, degree):
    c = [mp.mpc(1)]
    for j in range(1, degree):
        c.append(c[-1]*z/j)
    return c


def hilbert_regular_coeff(c):
    # H[e^(zt)] = e^(zt)*J1(t) + P_z(t).
    return [mp.fsum(c[d]*2/(d-j) for d in range(j+1, len(c)) if (d-j) % 2)
            for j in range(len(c)-1)]


def regular_moments(ell, count):
    # mu_j = integral t^j*omega; nu_j = integral t^j*omega*J1.
    mu = [mp.beta(mp.mpf(j+1)/2, mp.mpf(ell)+mp.mpf('.5')) if j % 2 == 0
          else mp.mpf(0) for j in range(count)]
    nu = [mp.mpf(0)]*count
    for n in range(count-1):
        nu[n+1] = (n*(nu[n-1] if n else 0)-2*mu[n])/(n+2*ell+1)
    return mu, nu


def endpoint_moments(poly, ell, m):
    # Analytic continuation of beta integral after t=2*x-1.
    c = [mp.mpf(0)]*len(poly)
    for d, v in enumerate(poly):
        for j in range(d+1):
            c[j] += v*math.comb(d, j)*2**j*(-1)**(d-j)
    a = mp.mpf(ell)+mp.mpf('.5')
    kp = [mp.power(2, 2*ell-q)*mp.fsum(v*mp.beta(a+j, a-q) for j, v in enumerate(c))
          for q in range(1, m)]
    # phi_k has parity (-1)^k; omega is even.
    km = [(-1)**(len(poly)-1)*v for v in kp]
    return kp, km


def build_b(m, z, phi, mu, nu, coeff, pregular):
    """Integrate known f = z^(m-1) H[e^(zt)]/(m-1)! + poles.

    This function does NOT use c_m*lambda_k, ordinary solution moments,
    the online transposed residual, or verification_reference().
    """
    n = m-1
    factor = z**n/mp.factorial(n)
    ap = [-mp.factorial(q-1)/mp.factorial(n)*z**(n-q)*mp.exp(z) for q in range(1, m)]
    am = [(-1)**q*mp.factorial(q-1)/mp.factorial(n)*z**(n-q)*mp.exp(-z) for q in range(1, m)]
    regular, poles, total = [], [], []
    for poly in phi:
        lm = [mp.fsum(v*nu[d+j] for j, v in enumerate(poly)) for d in range(len(coeff))]
        pm = [mp.fsum(v*mu[d+j] for j, v in enumerate(poly)) for d in range(len(pregular))]
        r = factor*(mp.fsum(a*b for a, b in zip(coeff, lm))+
                    mp.fsum(a*b for a, b in zip(pregular, pm)))
        kp, km = endpoint_moments(poly, m//2, m)
        p = mp.fsum(a*b+c*d for a, b, c, d in zip(ap, kp, am, km))
        regular.append(r)
        poles.append(p)
        total.append(r+p)
    return total, regular, poles


def verification_reference(m, phi, coeff):
    # ONLY for an independent check of the endpoint finite-part integration.
    out = []
    for k, poly in enumerate(phi):
        integral = mp.fsum(a*v*2/(d+j+1) for d, a in enumerate(coeff)
                          for j, v in enumerate(poly) if (d+j) % 2 == 0)
        out.append((-1)**(m//2)*mp.pi*math.comb(k+m-1, m-1)*integral)
    return out


def rhs_point(m, z, t, pregular):
    f = mp.exp(z*t)*mp.log((1-t)/(1+t))+mp.polyval(list(reversed(pregular)), t)
    for r in range(2, m+1):
        f = (z*f-mp.exp(z)/(1-t)**(r-1)+mp.exp(-z)/(-1-t)**(r-1))/(r-1)
    return f


def point_reference(m, z, t, degree=320):
    # Direct Taylor-subtracted integral around t, not the Hilbert recurrence.
    j1 = mp.log((1-t)/(1+t))
    value = z**(m-1)/mp.factorial(m-1)*j1
    for j in range(m-1):
        r = m-j
        jr = ((1-t)**(1-r)-(-1-t)**(1-r))/(1-r)
        value += z**j/mp.factorial(j)*jr
    value += mp.fsum(z**j/mp.factorial(j)*((1-t)**(j-m+1)-(-1-t)**(j-m+1))/(j-m+1)
                     for j in range(m, degree))
    return mp.exp(z*t)*value


def build(output, dps=110, degree=256, modes=64, orders=(2, 4, 6)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    with mp.workdps(dps):
        series_start = time.perf_counter()
        zs = [mp.mpf(1), mp.mpf('.5'), 6*mp.pi*1j]
        series = [(exponential_coeff(z, degree), None) for z in zs]
        series = [(c, hilbert_regular_coeff(c)) for c, _ in series]
        series_seconds = time.perf_counter()-series_start
        for m in orders:
            start = time.perf_counter()
            phi = polys(m//2, modes)
            mu, nu = regular_moments(m//2, degree+modes)
            bs, rs, ps, refs = [], [], [], []
            b_seconds, reference_seconds = 0., 0.
            for z, (c, p) in zip(zs, series):
                tt = time.perf_counter()
                b, r, pole = build_b(m, z, phi, mu, nu, c, p)
                b_seconds += time.perf_counter()-tt
                bs.append(b); rs.append(r); ps.append(pole)
                tt = time.perf_counter()
                refs.append(verification_reference(m, phi, c))
                reference_seconds += time.perf_counter()-tt
            setup = time.perf_counter()-start
            for family in ('exp', 'multiscale'):
                def combine(v):
                    return [mp.re(a) for a in v[0]] if family == 'exp' else [mp.re(a)+mp.mpf('.2')*mp.im(b) for a, b in zip(v[1], v[2])]
                b, reg, pole, reference = map(combine, (bs, rs, ps, refs))
                defect = max(abs(a-b)/max(1, abs(b)) for a, b in zip(b, reference))
                assert defect < mp.mpf('1e-50'), (m, family, str(defect))
                arrays = dict(b=np.array(b, float), b_regular=np.array(reg, float), b_poles=np.array(pole, float),
                              verification_only_b=np.array(reference, float))
                point_defect = mp.mpf(0)
                point_start = time.perf_counter()
                for q in (192, 384):
                    x, w = leggauss(q)
                    values = []
                    for xx in x:
                        t = mp.mpf(float(xx))
                        if family == 'exp':
                            value = mp.re(rhs_point(m, zs[0], t, series[0][1]))
                        else:
                            value = mp.re(rhs_point(m, zs[1], t, series[1][1]))+mp.mpf('.2')*mp.im(rhs_point(m, zs[2], t, series[2][1]))
                        values.append(value)
                    arrays.update({f'x{q}': x, f'w{q}': w, f'f{q}': np.array(values, float)})
                point_seconds = time.perf_counter()-point_start
                point_validation_start = time.perf_counter()
                for z, (_, p) in zip(zs, series):
                    for t in map(mp.mpf, ('-.9999', '-.83', '.17', '.91', '.9999')):
                        a, b0 = rhs_point(m, z, t, p), point_reference(m, z, t)
                        point_defect = max(point_defect, abs(a-b0)/max(1, abs(b0)))
                assert point_defect < mp.mpf('1e-50')
                np.savez(output/f'm{m}_{family}.npz', **arrays)
                row = dict(m=m, family=family, modes=modes, moment_pairing_defect=float(defect),
                           strong_rhs_defect=float(point_defect), setup_seconds=setup,
                           b_construction_seconds_shared_all_complex_components=b_seconds,
                           point_construction_seconds=point_seconds,
                           series_construction_seconds_shared_all_orders=series_seconds,
                           verification_seconds_excluded=reference_seconds+time.perf_counter()-point_validation_start,
                           b_source='known f: integrable Hilbert remainder plus analytic endpoint finite-part beta moments',
                           verification_only_b_used_for_training=False)
                checks.append(row)
                print(json.dumps(row), flush=True)
    report = dict(status='passed', dps=dps, degree=degree, modes=modes, checks=checks,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  solution='raw u, no endpoint factor', quadrature='ordinary Gauss-Legendre')
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    build(args.output)
