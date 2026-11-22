"""Mapped MC benchmark and independent known-f endpoint finite-part moments."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'research/hie_gram_direct_u_20261018'
sys.path.insert(0, str(OLD))
import rhs_data as old

POLY = (-mp.mpf(1)/8, -mp.mpf(1)/8, mp.mpf(1)/8, mp.mpf(1)/8)


def exact(x, family):
    if family == 'exp':
        return np.exp(x)
    if family == 'polynomial':
        return (x+1)**2*(x-1)/8
    raise ValueError(family)


def derivative_coeff(j):
    return [POLY[k]*math.comb(k, j) for k in range(j, 4)] if j < 4 else [mp.mpf(0)]


def jq(q, t):
    if q == 1:
        return mp.log((1-t)/(1+t))
    return ((1-t)**(1-q)-(-1-t)**(1-q))/(1-q)


def polynomial_rhs(m, t):
    return mp.fsum(mp.polyval(list(reversed(derivative_coeff(j))), t)*jq(m-j, t)
                   for j in range(4))


def convolution(a, b):
    out = [mp.mpf(0)]*(len(a)+len(b)-1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i+j] += ai*bj
    return out


def endpoint(poly, ell, power):
    shifted = [mp.mpf(0)]*len(poly)
    for d, v in enumerate(poly):
        for j in range(d+1):
            shifted[j] += v*math.comb(d, j)*2**j*(-1)**(d-j)
    a = mp.mpf(ell)+mp.mpf('.5')
    plus = mp.power(2, 2*ell-power)*mp.fsum(v*mp.beta(a+j, a-power) for j, v in enumerate(shifted))
    minus = mp.power(2, 2*ell-power)*mp.fsum(v*mp.beta(a+j-power, a) for j, v in enumerate(shifted))
    return plus, minus


def known_f_moments(m, phi):
    """Integrate f=sum p_j(t) J_(m-j)(t); no adjoint-image multiplier here."""
    mu, nu = old.regular_moments(m//2, len(phi)+8)
    total, regular, poles = [], [], []
    for test in phi:
        reg, pole = mp.mpf(0), mp.mpf(0)
        for j in range(4):
            poly = convolution(test, derivative_coeff(j))
            q = m-j
            if q == 1:
                reg += mp.fsum(a*nu[k] for k, a in enumerate(poly))
            elif q == 0:
                reg += 2*mp.fsum(a*mu[k] for k, a in enumerate(poly))
            elif q == -1:
                reg -= 2*mp.fsum(a*mu[k+1] for k, a in enumerate(poly))
            else:
                kp, km = endpoint(poly, m//2, q-1)
                pole += -(kp-(-1)**(q-1)*km)/(q-1)
        regular.append(reg); poles.append(pole); total.append(reg+pole)
    return total, regular, poles


def build(output, original, dps=150):
    output, original = Path(output), Path(original)
    output.mkdir(parents=True, exist_ok=False)
    original_validation = json.loads((original/'validation.json').read_text())
    checks = []
    for m in (2, 4, 6):
        shutil.copy2(original/f'm{m}_exp.npz', output/f'm{m}_exp.npz')
        checks += [r for r in original_validation['checks'] if r['m'] == m and r['family'] == 'exp']
        start = time.perf_counter()
        with mp.workdps(dps):
            phi = old.polys(m//2, 64)
            b, regular, poles = known_f_moments(m, phi)
            # Analytic-u projection is used solely as a verification control.
            reference = old.verification_reference(m, phi, POLY)
            defect = max(abs(a-bb)/max(1, abs(bb)) for a, bb in zip(b, reference))
            assert defect < mp.mpf('1e-70'), (m, str(defect))
            arrays = dict(b=np.asarray(b, float), b_regular=np.asarray(regular, float), b_poles=np.asarray(poles, float),
                          verification_only_b=np.asarray(reference, float))
            for q in (192, 384):
                x, w = leggauss(q)
                arrays.update({f'x{q}': x, f'w{q}': w,
                               f'f{q}': np.asarray([polynomial_rhs(m, mp.mpf(float(t))) for t in x], float)})
            if m == 2:
                for t in map(mp.mpf, ('-.99', '-.3', '.17', '.91')):
                    explicit = (3*t+2)/4+(3*t*t+2*t-1)/8*mp.log((1-t)/(1+t))
                    assert abs(explicit-polynomial_rhs(m, t)) < mp.mpf('1e-90')
            np.savez(output/f'm{m}_polynomial.npz', **arrays)
            row = dict(m=m, family='polynomial', modes=64, setup_seconds=time.perf_counter()-start,
                       moment_pairing_defect=float(defect), b_source='known f polynomial J_q split and endpoint beta finite parts',
                       verification_only_b_used_for_training=False)
            checks.append(row)
            print(json.dumps(row), flush=True)
    report = dict(status='passed', dps=dps, checks=checks, raw_u=True,
                  original_source='MC-Nonlocal-PINNs arXiv:2212.12984 Eq.4.3; x=(t+1)/2',
                  exact_polynomial='(t+1)^2*(t-1)/8', extra_endpoint_loss=False)
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--dps', type=int, default=150)
    args = p.parse_args()
    build(args.output, args.original, args.dps)
