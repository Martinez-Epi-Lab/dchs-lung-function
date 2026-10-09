"""Diagnostic gates and interval-limit MCSE checks on fabricated MCMC arrays."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.diagnostics import compare_contrasts, evaluate_diagnostics


def fabricated_draws(chains=4, draws=1200):
    rng = np.random.default_rng(8031)
    posterior = {'beta': rng.normal(size=(chains, draws)),
                 'mechanics': rng.normal(size=(chains, draws, 5)),
                 'lci_raw': rng.normal(size=(chains, draws, 3))}
    posterior['lci_process'] = .8*posterior['lci_raw']
    stats = {'diverging': np.zeros((chains, draws), dtype=bool),
             'energy': rng.normal(size=(chains, draws)),
             'reached_max_treedepth': np.zeros((chains, draws), dtype=bool)}
    return posterior, stats


class DiagnosticTests(unittest.TestCase):
    def test_independent_draws_pass_and_account_for_every_child_scalar(self):
        posterior, stats = fabricated_draws()
        result = evaluate_diagnostics(posterior, stats, {})
        self.assertTrue(result['passed'], result['reasons'])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['metrics']['posterior_variables'], 4)
        self.assertEqual(result['metrics']['posterior_scalars'], 12)
        self.assertEqual(result['metrics']['by_variable']['mechanics']['scalars'], 5)
        self.assertEqual(result['metrics']['chains'], 4)
        self.assertEqual(result['metrics']['draws_per_chain'], 1200)
        self.assertEqual(result['metrics']['divergences'], 0)
        json.dumps(result, allow_nan=False)

    def test_one_failed_child_cannot_hide_behind_good_global_parameters(self):
        posterior, stats = fabricated_draws()
        posterior['mechanics'][0, :, 4] += 4.
        result = evaluate_diagnostics(posterior, stats, {})
        self.assertFalse(result['passed'])
        self.assertGreater(result['metrics']['by_variable']['mechanics']['max_rhat'], 1.01)
        self.assertLessEqual(result['metrics']['by_variable']['beta']['max_rhat'], 1.01)
        self.assertTrue(any('R-hat' in reason for reason in result['reasons']))

    def test_bulk_and_tail_ess_are_independent_required_gates(self):
        posterior, stats = fabricated_draws()
        for key, reason in [('min_bulk_ess', 'Bulk ESS'), ('min_tail_ess', 'Tail ESS')]:
            result = evaluate_diagnostics(posterior, stats, {key: 100000})
            self.assertFalse(result['passed'])
            self.assertTrue(any(reason in text for text in result['reasons']))

    def test_low_energy_mixing_divergence_and_tree_depth_each_fail(self):
        posterior, stats = fabricated_draws()
        stats['energy'] = np.cumsum(stats['energy'], axis=1)
        result = evaluate_diagnostics(posterior, stats, {})
        self.assertFalse(result['passed'])
        self.assertLess(result['metrics']['min_bfmi'], .3)
        for key, metric in [('diverging', 'divergences'), ('reached_max_treedepth', 'treedepth_hits')]:
            posterior, stats = fabricated_draws(); stats[key][2, 100] = True
            result = evaluate_diagnostics(posterior, stats, {})
            self.assertFalse(result['passed'])
            self.assertEqual(result['metrics'][metric], 1)

    def test_missing_nonfinite_and_mismatched_arrays_fail_closed(self):
        for change in ['missing_stat', 'nan_posterior', 'nan_stat', 'stat_shape',
                       'posterior_shape', 'nonbinary_stat', 'empty_dimension', 'empty_posterior']:
            posterior, stats = fabricated_draws()
            if change == 'missing_stat': del stats['reached_max_treedepth']
            elif change == 'nan_posterior': posterior['mechanics'][3, 500, 2] = np.nan
            elif change == 'nan_stat': stats['energy'][0, 0] = np.inf
            elif change == 'stat_shape': stats['energy'] = stats['energy'].T
            elif change == 'posterior_shape': posterior['mechanics'] = posterior['mechanics'][:, :-1]
            elif change == 'nonbinary_stat': stats['diverging'] = np.full((4, 1200), 2)
            elif change == 'empty_dimension': posterior['mechanics'] = np.empty((4, 1200, 0))
            else: posterior = {}
            result = evaluate_diagnostics(posterior, stats, {})
            with self.subTest(change=change):
                self.assertFalse(result['passed']); self.assertTrue(result['reasons'])
                json.dumps(result, allow_nan=False)

    def test_four_chains_and_protocol_draw_minimum(self):
        posterior, stats = fabricated_draws(chains=3)
        self.assertFalse(evaluate_diagnostics(posterior, stats, {})['passed'])
        posterior, stats = fabricated_draws(draws=1199)
        self.assertFalse(evaluate_diagnostics(posterior, stats, {})['passed'])
        result = evaluate_diagnostics(posterior, stats, {'min_draws': 1000})
        self.assertTrue(result['passed'], result['reasons'])

    def test_constant_posterior_and_constant_energy_do_not_produce_json_nan(self):
        for what in ['posterior', 'energy']:
            posterior, stats = fabricated_draws()
            if what == 'posterior': posterior['beta'][:] = 0.
            else: stats['energy'][:] = 1.
            result = evaluate_diagnostics(posterior, stats, {})
            self.assertFalse(result['passed']); json.dumps(result, allow_nan=False)

    def test_invalid_or_relaxed_criteria_fail_and_source_alias_is_supported(self):
        posterior, stats = fabricated_draws()
        for criteria in [{'max_rhat': 1.2}, {'min_bulk_ess': 399}, {'min_tail_ess': 399},
                         {'min_bfmi': .1}, {'max_divergences': 1}, {'max_treedepth_hits': 1},
                         {'min_chains': 2}, {'min_draws': 3}, {'max_rhat': np.nan},
                         {'unrecognized': 3}, {'min_chains': True}]:
            result = evaluate_diagnostics(posterior, stats, criteria)
            with self.subTest(criteria=criteria):
                self.assertFalse(result['passed']); json.dumps(result, allow_nan=False)
        self.assertTrue(evaluate_diagnostics(posterior, stats, {'divergences': 0, 'scope': 'all'})['passed'])


class ContrastComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(9045)
        cls.reference = {'early': rng.normal(size=(4, 1200)),
                         'late': rng.normal(-.3, .5, size=(4, 1200))}
        cls.current = {name: value.copy() for name, value in cls.reference.items()}

    def test_mean_and_quantiles_use_separate_mcse_and_combined_threshold(self):
        import arviz as az
        current = deepcopy(self.current); current['early'] += .015
        result = compare_contrasts(current, self.reference)
        self.assertEqual(len(result), 6)
        self.assertTrue(result.passed.all())
        self.assertEqual(set(result.statistic), {'mean', 'q025', 'q975'})
        for _, row in result.iterrows():
            probability = {'q025': .025, 'q975': .975}.get(row.statistic)
            kwargs = {'method': 'mean'} if probability is None else {'method': 'quantile', 'prob': probability}
            expected_a = float(az.mcse(current[row.contrast], **kwargs))
            expected_b = float(az.mcse(self.reference[row.contrast], **kwargs))
            self.assertAlmostEqual(row.mcse_current, expected_a, places=14)
            self.assertAlmostEqual(row.mcse_reference, expected_b, places=14)
            self.assertAlmostEqual(row.threshold, 4*np.hypot(expected_a, expected_b), places=14)

    def test_changed_mean_and_changed_interval_limits_are_detected(self):
        shifted = {name: value+1. for name, value in self.current.items()}
        self.assertFalse(compare_contrasts(shifted, self.reference).passed.any())
        changed = deepcopy(self.current)
        # Keep the mean exactly fixed while widening the posterior interval.
        changed['early'] = 1.8*(changed['early']-changed['early'].mean())+changed['early'].mean()
        result = compare_contrasts(changed, self.reference)
        early = result.loc[result.contrast.eq('early')].set_index('statistic')
        self.assertTrue(bool(early.loc['mean', 'passed']))
        self.assertFalse(bool(early.loc['q025', 'passed']))
        self.assertFalse(bool(early.loc['q975', 'passed']))

    def test_distinct_run_shapes_are_allowed_but_malformed_inputs_rejected(self):
        longer = {name: np.concatenate([value, value], axis=1) for name, value in self.current.items()}
        result = compare_contrasts(longer, self.reference)
        self.assertEqual(set(result.current_draws), {2400})
        self.assertEqual(set(result.reference_draws), {1200})
        bad_inputs = [
            {'early': self.current['early']},
            {**self.current, 'early': self.current['early'].reshape(-1)},
            {**self.current, 'early': self.current['early'][None, ...]},
            {**self.current, 'early': self.current['early'][:, :-1]},
            {**self.current, 'early': np.full((4, 1200), np.nan)},
        ]
        for current in bad_inputs:
            with self.assertRaises(ValueError): compare_contrasts(current, self.reference)
        for multiplier in [0., -1., np.inf, np.nan, True]:
            with self.assertRaises(ValueError): compare_contrasts(self.current, self.reference, multiplier)


if __name__ == '__main__':
    unittest.main()
