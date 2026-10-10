"""Numerical precision of a ratio of posterior means, not posterior inference.

Separate fits can share simulation seeds.  A conservative first-order MCSE
bound therefore uses the sum of the absolute gradient contributions; it does
not assume zero cross-model Monte Carlo covariance or pair posterior draws.
"""
from __future__ import annotations

import math
import numbers

MAX_MCSE_BOUND_PP = 1.0


def attenuation_precision(adjusted_mean, core_mean, adjusted_mcse, core_mcse):
    """Compare effect magnitudes and estimate their Monte Carlo precision.

    The bound uses the supplied mean MCSEs and a first-order delta
    approximation. It is not a confidence or credible interval and does not
    guarantee identical last digits across independently rounded simulations.
    """
    values = (adjusted_mean, core_mean, adjusted_mcse, core_mcse)
    if any(isinstance(v, bool) or not isinstance(v, numbers.Real)
           or not math.isfinite(v) for v in values):
        raise ValueError('Attenuation precision requires finite real means and MCSEs.')
    if adjusted_mcse < 0 or core_mcse < 0:
        raise ValueError('Posterior mean MCSEs cannot be negative.')
    if core_mean == 0:
        return {'attenuation_percent': None, 'attenuation_status': 'undefined_zero_core',
                'attenuation_mcse_bound_pp': None, 'attenuation_mcse_independent_pp': None,
                'attenuation_precision_passed': False}
    if adjusted_mean == 0:
        # The delta-method MCSE is undefined at the kink in abs(a).
        return {'attenuation_percent': 100.0, 'attenuation_status': 'zero_adjusted_mean',
                'attenuation_mcse_bound_pp': None, 'attenuation_mcse_independent_pp': None,
                'attenuation_precision_passed': False}
    estimate = 100.0 * (1.0 - abs(adjusted_mean) / abs(core_mean))
    a = 100.0 * adjusted_mcse / abs(core_mean)
    c = 100.0 * (abs(adjusted_mean) / abs(core_mean)) * (core_mcse / abs(core_mean))
    bound = a + c
    independent = math.hypot(a, c)
    if not all(math.isfinite(v) for v in (estimate, bound, independent)):
        raise ValueError('Attenuation precision is nonfinite.')
    same_sign = (adjusted_mean > 0) == (core_mean > 0)
    return {'attenuation_percent': float(estimate),
            'attenuation_status': 'estimated' if same_sign else 'sign_reversal',
            'attenuation_mcse_bound_pp': float(bound),
            'attenuation_mcse_independent_pp': float(independent),
            'attenuation_precision_passed': bool(bound <= MAX_MCSE_BOUND_PP)}
