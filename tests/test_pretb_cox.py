"""Independent equation and failure checks for the selected Cox engine."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from statsmodels.duration.hazard_regression import PHReg

from dchs_lung_function.models import pretb_cox as cox


def contract(frame, target='resistance'):
    frame = frame.copy()
    return {'frame': frame, 'terms': list(cox.TERMS),
            'design': frame.loc[:, cox.TERMS].to_numpy(float),
            'audit': {'family': 'pretb_cox', 'column': target, 'carry_forward_years': 2.,
                      'n_children': frame.pid_child.nunique(), 'n_intervals': len(frame),
                      'events': int(frame.event.sum()),
                      'person_years': float((frame.stop-frame.start).sum())}}


def sample():
    rng = np.random.default_rng(746)
    rows = []
    for i in range(120):
        x = rng.normal()
        male, site = rng.integers(0, 2, 2)
        rows.append((f'synthetic-{i}', 0., .5, 0, x, male, site))
        event = int(rng.random() < .65)
        stop = float(rng.choice([1.5, 2., 2.5, 3.])) if event else 3.
        rows.append((f'synthetic-{i}', 1., stop, event, x+.2*rng.normal(), male, site))
    # Internal intervals without any event-age membership expose the selected
    # source residual bookkeeping, while the coefficient likelihood stays Cox.
    for i in range(8):
        rows.append((f'gap-{i}', 1.6, 1.8, 0, rng.normal(), i % 2, (i//2) % 2))
    return pd.DataFrame(rows, columns=['pid_child', 'start', 'stop', 'event', *cox.TERMS])


def equations(frame, beta):
    x = frame.loc[:, cox.TERMS].to_numpy(float)
    eta = x@beta
    weight = np.exp(eta)
    start, stop, event = frame.start.to_numpy(), frame.stop.to_numpy(), frame.event.to_numpy(bool)
    times = np.unique(stop[event])
    likelihood, score, information = 0., np.zeros(3), np.zeros((3, 3))
    for time in times:
        risk = (start <= time) & (stop >= time)
        dead = event & (stop == time)
        d = int(dead.sum())
        likelihood += eta[dead].sum()
        score += x[dead].sum(axis=0)
        for k in range(d):
            w = weight * (risk.astype(float)-k/d*dead)
            denom = w.sum()
            mean = np.sum(w[:, None]*x, axis=0)/denom
            likelihood -= np.log(denom)
            score -= mean
            information += (x.T*w)@x/denom-np.outer(mean, mean)
    return likelihood, score, information


def closed_form_scores(frame, beta, source_membership=True):
    x = frame.loc[:, cox.TERMS].to_numpy(float)
    start, stop, event = frame.start.to_numpy(), frame.stop.to_numpy(), frame.event.to_numpy(bool)
    times = np.unique(stop[event])
    true_risk = (start[None, :] <= times[:, None]) & (stop[None, :] >= times[:, None])
    retained = (start <= times[-1]) & (stop >= times[0])
    empty = ~true_risk.any(axis=0)
    active = true_risk | (retained[None, :] & empty[None, :] & (times[:, None] <= stop[None, :])) if source_membership else true_risk
    weight = np.exp(x@beta)
    scores = np.zeros_like(x)
    for j, time in enumerate(times):
        risk, support = true_risk[j], active[j]
        dead = event & (stop == time)
        denominator = weight[risk].sum()
        mean = (x[risk]*weight[risk, None]).sum(axis=0)/denominator
        martingale = dead-weight*dead.sum()/denominator
        scores[support] += (x[support]-mean)*martingale[support, None]
    return scores, active


def test_efron_likelihood_score_and_information_are_explicit():
    frame = sample()
    model = PHReg(frame.stop, frame.loc[:, cox.TERMS], status=frame.event, entry=frame.start, ties='efron')
    beta = np.array([.23, -.31, .12])
    loglike, score, info = equations(frame, beta)
    np.testing.assert_allclose(model.loglike(beta), loglike, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(model.score(beta), score, rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(-model.hessian(beta), info, rtol=1e-12, atol=1e-11)
    # Efron ties are material in this example.
    breslow = PHReg(frame.stop, frame.loc[:, cox.TERMS], status=frame.event, entry=frame.start, ties='breslow')
    assert abs(model.loglike(beta)-breslow.loglike(beta)) > 1


def test_selected_source_score_membership_and_sandwich():
    frame = sample()
    fit = cox.fit_cox(contract(frame), 'resistance')
    assert fit.metadata['accepted'], fit.metadata
    beta = fit.coefficients.to_numpy()
    _, gradient, info = equations(frame, beta)
    scores, support = closed_form_scores(frame, beta)
    bread = np.linalg.inv(info)
    codes = pd.Categorical(frame.pid_child).codes
    grouped = np.zeros((frame.pid_child.nunique(), 3))
    np.add.at(grouped, codes, scores)
    n, g = len(frame), len(grouped)
    robust = bread@(grouped.T@grouped)@bread*g/(g-1)*(n-1)/(n-3)
    np.testing.assert_allclose(fit.model_covariance, bread, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(fit.covariance, robust, rtol=1e-11, atol=1e-12)
    assert np.max(abs(gradient)) < 1e-7
    assert fit.metadata['risk_sets']['source_score_membership_difference_cells'] > 0
    assert fit.metadata['risk_sets']['source_undefined_score_intervals'] == 120
    direct, _ = closed_form_scores(frame, beta, source_membership=False)
    assert np.max(abs(direct-scores)) > .01
    # Tied-event martingale residuals do not sum to the Efron score.
    assert np.max(abs(scores.sum(axis=0)-gradient)) > .01
    assert fit.metadata['risk_sets']['source_score_membership_sha256'] == cox._bit_hash(support)
    json.dumps(fit.metadata, allow_nan=False)


def test_original_fit_and_retained_optimizer_have_same_estimates():
    frame = sample()
    original = PHReg(frame.stop, frame.loc[:, cox.TERMS], status=frame.event, entry=frame.start, ties='efron').fit()
    fit = cox.fit_cox(contract(frame), 'resistance')
    np.testing.assert_allclose(fit.coefficients, original.params, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fit.model_covariance, original.cov_params(), rtol=1e-13, atol=1e-14)
    assert fit.metadata['optimizer']['converged']
    assert fit.metadata['optimizer']['iterations'] > 0


def test_closed_entry_boundary_includes_both_intervals():
    frame = pd.DataFrame([
        ('a', 0., 1., 0, .1, 0, 0), ('a', 1., 2., 0, .2, 0, 0),
        ('b', 0., 1., 1, .3, 1, 1),
    ], columns=['pid_child', 'start', 'stop', 'event', *cox.TERMS])
    audit = cox.risk_set_audit(frame)
    time = audit['event_times'][0]
    assert time == {'event_age_years': 1., 'events': 1, 'risk_rows': 3, 'risk_children': 2,
                    'entry_at_event_rows': 1, 'entry_at_event_children': 1,
                    'additional_same_child_risk_rows': 1}


def test_only_library_undefined_scores_are_zeroed():
    supported = np.array([True, False, True])
    values = np.array([[1., 2., 3.], [np.nan]*3, [4., 5., 6.]])
    np.testing.assert_array_equal(cox._clean_source_scores(values, supported)[1], 0.)
    for row in ([np.nan, 0., 0.], [np.inf, np.inf, np.inf], [0., 0., 0.]):
        bad = values.copy(); bad[1] = row
        with unittest.TestCase().assertRaises(ArithmeticError):
            cox._clean_source_scores(bad, supported)
    bad = values.copy(); bad[0, 0] = np.nan
    with unittest.TestCase().assertRaises(ArithmeticError):
        cox._clean_source_scores(bad, supported)


def test_low_event_source_guard_preserves_exact_small_counts():
    frame = sample().iloc[:4].copy()
    frame['event'] = 0
    frame.iloc[-1, frame.columns.get_loc('event')] = 1
    fit = cox.fit_cox(contract(frame), 'resistance')
    assert fit.metadata['status'] == 'non_estimable' and fit.metadata['accepted']
    assert fit.metadata['counts']['n_children'] == 2
    assert fit.metadata['counts']['n_subsequent_tb'] == 1
    assert len(fit.estimates) == 3
    assert fit.estimates.coefficient.isna().all() and fit.estimates.p_value.isna().all()
    assert fit.covariance.isna().all().all()


def test_singular_design_is_failure_and_does_not_drop_adjustment():
    frame = sample(); frame['mbekweni'] = frame['male']
    fit = cox.fit_cox(contract(frame), 'resistance')
    assert not fit.metadata['accepted'] and fit.metadata['status'] == 'fit_failed'
    assert fit.metadata['terms'] == list(cox.TERMS)
    assert fit.estimates.hazard_ratio.isna().all()


def test_nonconverged_fit_is_explicit_failure():
    from statsmodels.base.model import LikelihoodModel
    original = LikelihoodModel.fit
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        result.mle_retvals['converged'] = False
        result.mle_retvals['warnflag'] = 1
        return result
    with patch.object(LikelihoodModel, 'fit', fail):
        fit = cox.fit_cox(contract(sample()), 'resistance')
    assert fit.metadata['reason'] == 'optimizer_did_not_converge'
    assert not fit.metadata['accepted'] and fit.coefficients.isna().all()


def test_contract_validation_is_strict_and_input_is_unchanged():
    prepared = contract(sample())
    original = deepcopy(prepared)
    cox.fit_cox(prepared, 'resistance')
    pd.testing.assert_frame_equal(prepared['frame'], original['frame'])
    np.testing.assert_array_equal(prepared['design'], original['design'])
    with unittest.TestCase().assertRaisesRegex(ValueError, 'target'):
        cox.validate_contract(prepared, 'compliance')
    prepared['design'][0, 0] += 1
    with unittest.TestCase().assertRaisesRegex(ValueError, 'matrix'):
        cox.validate_contract(prepared, 'resistance')


def test_full_selection_inventory_and_finite_only_bh():
    frame = sample().iloc[:4].copy(); frame['event'] = 0
    contracts = {cox.contract_key(t): contract(frame, t) for t in cox.DEFAULT_TARGETS}
    result = cox.fit_selected(contracts)
    assert result['metadata']['accepted']
    tables = result['tables']
    assert tuple(tables['estimates']) == cox.ESTIMATE_COLUMNS
    assert tuple(tables['coefficients']) == cox.COEFFICIENT_COLUMNS
    assert tuple(tables['covariance']) == cox.COVARIANCE_COLUMNS
    assert [len(tables[k]) for k in ('estimates', 'coefficients', 'covariance')] == [17, 51, 306]
    assert tables['estimates'].target.tolist() == list(cox.DEFAULT_TARGETS)
    assert tables['estimates'].fdr_q_value.isna().all()
    np.testing.assert_allclose(cox.benjamini_hochberg([.01, .04, .04, np.nan, .9]),
                               [.04, .04*4/3, .04*4/3, np.nan, .9], equal_nan=True)
    with unittest.TestCase().assertRaisesRegex(ValueError, '17'):
        cox.fit_selected({})


def test_child_sandwich_retains_cross_interval_score_products():
    frame = sample()
    first = frame.start.eq(0.)
    second = frame.start.eq(1.)
    frame.loc[first, 'stop'] = 1.5
    frame.loc[second, 'start'] = 1.5
    frame.loc[second & frame.stop.eq(1.5), 'stop'] = 1.8
    added = pd.DataFrame([
        ('early-a', 0., 1.2, 1, .2, 0, 1),
        ('early-b', 0., 1.4, 1, -.3, 1, 0),
    ], columns=frame.columns)
    frame = pd.concat([frame, added], ignore_index=True)
    fit = cox.fit_cox(contract(frame), 'resistance')
    assert fit.metadata['accepted']
    scores, _ = closed_form_scores(frame, fit.coefficients.to_numpy())
    bread = fit.model_covariance.to_numpy()
    correction = fit.metadata['covariance_correction']
    row_sandwich = bread@(scores.T@scores)@bread*correction
    grouped = np.zeros((frame.pid_child.nunique(), 3))
    np.add.at(grouped, pd.Categorical(frame.pid_child).codes, scores)
    child_sandwich = bread@(grouped.T@grouped)@bread*correction
    np.testing.assert_allclose(fit.covariance, child_sandwich, rtol=1e-11, atol=1e-12)
    assert np.max(abs(child_sandwich-row_sandwich)) > 1e-4


def test_bad_runtime_rejects_unverified_score_residual_implementation():
    with patch.object(cox.statsmodels, '__version__', '0.0.0'):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, '0.14.5'):
            cox.fit_cox(contract(sample()), 'resistance')


class PreTBCoxTests(unittest.TestCase):
    pass


for _name, _test in list(globals().items()):
    if _name.startswith("test_") and callable(_test):
        setattr(PreTBCoxTests, _name, staticmethod(_test))


del _name, _test


if __name__ == "__main__":
    unittest.main()
