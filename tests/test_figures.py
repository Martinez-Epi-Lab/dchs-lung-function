"""Independent synthetic uncertainty and figure-publication safeguards.

All interval records in these tests are invented. Numerical expectations are
calculated independently from the accepted synthetic Efron fit; no study data
or participant-level outputs are read or written by the suite.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import quote
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dchs_lung_function.analysis import pretb_uncertainty as uncertainty
from dchs_lung_function.models import pretb_cox as cox
from dchs_lung_function import pretb_execution
from dchs_lung_function.execution import ArtifactWriter, file_hash, write_json
from dchs_lung_function.pipeline import ValidationError
from test_pretb_cox import sample, contract
from dchs_lung_function.reporting import figure_data as figure_data
from dchs_lung_function.reporting.results import _schema, _semantics
from dchs_lung_function.reporting.formatting import ReportingError

ROOT = Path(__file__).resolve().parents[1]


def independent_scores(frame, beta):
    """Scalar event/interval loops, without the production membership matrix."""
    rows = list(frame.itertuples(index=False))
    x = frame.loc[:, cox.TERMS].to_numpy(float)
    scores = np.zeros_like(x)
    times = sorted({row.stop for row in rows if row.event == 1})
    membership = np.zeros((len(times), len(rows)), dtype=bool)
    breslow_gradient = np.zeros(3)
    for j, age in enumerate(times):
        selected = [i for i, row in enumerate(rows) if row.start <= age <= row.stop]
        deaths = [i for i, row in enumerate(rows) if row.event == 1 and row.stop == age]
        weights = {i: np.exp(float(x[i] @ beta)) for i in selected}
        denominator = sum(weights.values())
        mean = sum(weights[i] * x[i] for i in selected) / denominator
        for i in selected:
            membership[j, i] = True
            scores[i] += (x[i] - mean) * (int(i in deaths) - len(deaths) * weights[i] / denominator)
        breslow_gradient += sum(x[i] for i in deaths) - len(deaths) * mean
    grouped = np.array([scores[np.asarray(frame.pid_child == child)].sum(axis=0)
                        for child in sorted(set(frame.pid_child))])
    return scores, grouped, membership, breslow_gradient


def independent_sandwich(frame, beta, bread):
    scores, grouped, membership, gradient = independent_scores(frame, beta)
    n, g = len(frame), frame.pid_child.nunique()
    covariance = bread @ (grouped.T @ grouped) @ bread * g / (g - 1) * (n - 1) / (n - 3)
    return covariance, scores, grouped, membership, gradient


class CorrectedCoxNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = sample()
        cls.prepared = contract(cls.frame)
        cls.fit = cox.fit_cox(cls.prepared, 'resistance')
        assert cls.fit.metadata['accepted']
        cls.beta = cls.fit.coefficients.to_numpy()
        cls.bread = cls.fit.model_covariance.to_numpy()
        cls.corrected, cls.evidence = uncertainty.correct_covariance(cls.prepared, 'resistance', cls.beta, cls.bread)

    def call(self, prepared=None, beta=None, bread=None):
        return uncertainty.correct_covariance(self.prepared if prepared is None else prepared, 'resistance',
            self.beta if beta is None else beta, self.bread if bread is None else bread)

    def test_full_covariance_including_offdiagonals_uses_actual_risk_child_scores(self):
        expected, _, grouped, risk, _ = independent_sandwich(self.frame, self.beta, self.bread)
        np.testing.assert_allclose(self.corrected, expected, rtol=2e-12, atol=2e-13)
        self.assertGreater(np.max(np.abs(self.corrected - self.fit.covariance.to_numpy())), 1e-6)
        np.testing.assert_allclose(self.evidence['score_cluster_sum'], grouped.sum(axis=0), rtol=1e-12, atol=1e-12)
        expected_hash = hashlib.sha256(np.packbits(risk, bitorder='little').tobytes()).hexdigest()
        self.assertEqual(self.evidence['risk_membership_sha256'], expected_hash)

    def test_ties_keep_breslow_residual_score_and_accepted_efron_bread(self):
        _, grouped, _, breslow = independent_scores(self.frame, self.beta)
        np.testing.assert_allclose(grouped.sum(axis=0), breslow, rtol=1e-12, atol=1e-12)
        self.assertGreater(np.max(np.abs(breslow)), .01)
        self.assertLess(self.evidence['max_absolute_efron_score'], 1e-5)
        spec = uncertainty.specification()
        self.assertIn('Efron', spec['likelihood'])
        self.assertIn('Breslow', spec['score_method'])
        self.assertEqual(spec['confidence_multiplier'], 1.96)

    def test_irrelevant_extreme_covariates_cannot_change_weights_or_covariance(self):
        _, _, risk, _ = independent_scores(self.frame, self.beta)
        frame = self.frame.copy()
        # A huge irrelevant predictor must not set the exponential shift.
        frame.loc[~risk.any(axis=0), 'lf_adverse_sd'] = np.copysign(1e10, self.beta[0])
        actual, evidence = self.call(contract(frame))
        np.testing.assert_array_equal(actual, self.corrected)
        self.assertEqual(evidence['risk_membership_sha256'], self.evidence['risk_membership_sha256'])
        self.assertEqual(evidence['removed_legacy_scored_event_free_intervals'], 8)

    def test_wholly_unsupported_intervals_have_exactly_zero_scores(self):
        _, scores, _, risk, _ = independent_sandwich(self.frame, self.beta, self.bread)
        np.testing.assert_array_equal(scores[~risk.any(axis=0)], 0.)
        self.assertEqual(self.evidence['intervals_outside_all_event_risk_sets'], int((~risk.any(axis=0)).sum()))
        self.assertTrue(self.evidence['outside_risk_scores_exactly_zero'])
        self.assertGreater(self.evidence['removed_legacy_score_memberships'], 0)

    def test_shared_child_cross_interval_products_are_retained(self):
        # Introduce two risk-supported intervals for one child at a retained
        # inclusive boundary, rather than the fixture's unsupported first visits.
        frame = self.frame.copy()
        child = frame.loc[(frame.start == 1) & (frame.stop > 2), 'pid_child'].iloc[0]
        pair = frame.index[frame.pid_child == child]
        frame.loc[pair[0], 'stop'] = 1.5
        frame.loc[pair[1], 'start'] = 1.5
        prepared = contract(frame)
        fit = cox.fit_cox(prepared, 'resistance')
        self.assertTrue(fit.metadata['accepted'])
        beta, bread = fit.coefficients.to_numpy(), fit.model_covariance.to_numpy()
        actual, evidence = uncertainty.correct_covariance(prepared, 'resistance', beta, bread)
        expected, scores, _, risk, _ = independent_sandwich(frame, beta, bread)
        n, g = len(frame), frame.pid_child.nunique()
        independent_rows = bread @ (scores.T @ scores) @ bread * g/(g-1)*(n-1)/(n-3)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
        self.assertGreater(np.max(np.abs(actual - independent_rows)), 1e-6)
        self.assertEqual(evidence['entry_at_event_memberships'], 1)
        self.assertEqual(int(risk[:, pair].sum()), int(risk[:, pair[0]].sum()+risk[:, pair[1]].sum()))
        self.assertTrue(risk[0, pair].all())

    def test_small_estimable_group_counts_are_exact_and_unsuppressed(self):
        frame = sample().iloc[18:36].copy()
        prepared = contract(frame)
        fit = cox.fit_cox(prepared, 'resistance')
        self.assertTrue(fit.metadata['accepted'])
        actual, evidence = uncertainty.correct_covariance(prepared, 'resistance', fit.coefficients, fit.model_covariance)
        self.assertEqual(evidence['counts']['n_children'], 9)
        self.assertEqual(evidence['counts']['n_subsequent_tb'], 7)
        self.assertEqual(evidence['counts']['n_intervals'], 18)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(evidence['finite_sample_correction'], 9/8*17/15)

    def test_input_arrays_and_contract_remain_unchanged_without_fitting(self):
        prepared, beta, bread = deepcopy(self.prepared), self.beta.copy(), self.bread.copy()
        before = deepcopy(prepared)
        with mock.patch.object(cox, 'fit_cox', side_effect=AssertionError('must not refit')):
            uncertainty.correct_covariance(prepared, 'resistance', beta, bread)
        pd.testing.assert_frame_equal(prepared['frame'], before['frame'])
        np.testing.assert_array_equal(prepared['design'], before['design'])
        np.testing.assert_array_equal(beta, self.beta)
        np.testing.assert_array_equal(bread, self.bread)

    def test_stale_accepted_beta_or_bread_is_rejected(self):
        for beta, bread in [(self.beta + .1, self.bread), (self.beta, self.bread * 1.01)]:
            with self.subTest(beta_changed=bool(np.any(beta != self.beta))), self.assertRaises(ValueError):
                self.call(beta=beta, bread=bread)

    def test_invalid_bread_shapes_finiteness_symmetry_and_definiteness_rejected(self):
        asym = self.bread.copy(); asym[0, 1] += .01
        for bad in (np.eye(2), np.full((3,3), np.nan), np.diag([1.,1.,-1.]), np.zeros((3,3)), asym):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.call(bread=bad)

    def test_nonfinite_or_wrong_size_coefficients_rejected(self):
        for bad in (np.array([1.,2.]), np.array([np.inf,0.,0.]), np.array([np.nan,0.,0.])):
            with self.subTest(size=bad.size), self.assertRaises(ValueError):
                self.call(beta=bad)

    def test_minimum_event_rule_does_not_silently_create_estimates(self):
        frame = sample().iloc[:4].copy(); frame['event'] = 0
        frame.iloc[-1, frame.columns.get_loc('event')] = 1
        with self.assertRaises(ValueError):
            self.call(contract(frame))

    def test_contract_matrix_ordering_counts_and_cluster_validity_are_checked(self):
        changes = []
        c = deepcopy(self.prepared); c['design'][0,0] += 1; changes.append(c)
        c = deepcopy(self.prepared); c['audit']['n_children'] += 1; changes.append(c)
        c = deepcopy(self.prepared); c['frame'].loc[0, 'pid_child'] = ' '; changes.append(c)
        c = deepcopy(self.prepared); c['frame'] = c['frame'].iloc[::-1]; c['design'] = c['design'][::-1]; changes.append(c)
        for c in changes:
            with self.assertRaises(ValueError):
                self.call(c)

    def test_aggregate_evidence_contains_no_cluster_labels_or_record_arrays(self):
        serialized = json.dumps(self.evidence, allow_nan=False)
        self.assertFalse(any(str(label) in serialized for label in self.frame.pid_child.unique()))
        self.assertNotIn('pid_child', serialized)
        self.assertEqual(len(self.evidence['score_cluster_sum']), 3)
        self.assertEqual(self.evidence['risk_membership_shape'][1], len(self.frame))


class CorrectedCoxPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = sample()
        cls.contract_fixture = {}
        for i, target in enumerate(cox.DEFAULT_TARGETS):
            frame = base.copy()
            frame['lf_adverse_sd'] += .03 * i * np.sin(np.arange(len(frame)))
            if i < 2:
                frame = base.iloc[:4].copy(); frame['event'] = 0
                if i:
                    frame.iloc[-1, frame.columns.get_loc('event')] = 1
            cls.contract_fixture[cox.contract_key(target)] = contract(frame, target)
        cls.result_fixture = cox.fit_selected(cls.contract_fixture)
        assert cls.result_fixture['metadata']['accepted']

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name); self.prep = self.base/'preparation'; self.prep.mkdir()
        self.contracts = deepcopy(self.contract_fixture)
        manifest = ArtifactWriter(self.prep).finish({'M03':{'pft':{'samples':self.contracts}}})
        prep_identity = {'through':'M03','fixture':'wholly synthetic'}
        write_json(self.prep/'receipt.json', {'through':'M03','status':'prepared','identity':prep_identity,
            'fingerprint':pretb_execution.digest(prep_identity),
            'artifact_manifest_sha256':file_hash(self.prep/'artifacts.json')})
        descriptor = pretb_execution.contract_descriptor(self.prep)
        fingerprint, identity = pretb_execution.fit_identity(ROOT, descriptor)
        self.source = self.base/'accepted'/fingerprint; self.source.mkdir(parents=True)
        self.original = deepcopy(self.result_fixture)
        for name, frame in self.original['tables'].items():
            frame.to_csv(self.source/(name+'.csv'), index=False, float_format='%.17g')
        summary = pretb_execution._summary(self.original, fingerprint)
        write_json(self.source/'diagnostics.json', self.original['metadata'])
        write_json(self.source/'summary.json', summary)
        self.receipt = {'schema_version':1,'fingerprint':fingerprint,'identity':identity,'summary':summary,
            'files':{n:file_hash(self.source/n) for n in uncertainty.RESULT_FILES}}
        write_json(self.source/'receipt.json', self.receipt)

    def run_correction(self):
        with mock.patch.object(cox, 'fit_selected', side_effect=AssertionError('no new fitting')):
            return uncertainty.calculate(self.prep, self.source)

    def repair_hash(self, name):
        self.receipt['files'][name] = file_hash(self.source/name)
        write_json(self.source/'receipt.json', self.receipt)

    def test_complete_correction_changes_only_declared_uncertainty_and_no_source_bytes(self):
        before = {p.name:file_hash(p) for p in self.source.iterdir()}
        result = self.run_correction()
        self.assertEqual(result.metadata['rows'], {'estimates':17,'coefficients':51,'covariance':306})
        self.assertEqual(result.metadata['model_fits_performed'], 0)
        for name, permitted in [('estimates', {'robust_se','ci_low','ci_high','p_value','fdr_q_value'}),
                                ('coefficients', {'robust_se','ci_low','ci_high','p_value'})]:
            old = uncertainty._read_tables(self.source)[name]
            columns = [c for c in old if c not in permitted]
            pd.testing.assert_frame_equal(getattr(result,name)[columns], old[columns], check_exact=True)
        old = uncertainty._read_tables(self.source)['covariance']; mask = old.covariance_type.eq('model')
        pd.testing.assert_frame_equal(result.covariance.loc[mask], old.loc[mask], check_exact=True)
        self.assertEqual({p.name:file_hash(p) for p in self.source.iterdir()}, before)

    def test_full_coefficients_se_intervals_p_and_target_bh_are_recomputed(self):
        result = self.run_correction()
        for target in cox.DEFAULT_TARGETS[2:]:
            prepared = self.contracts[cox.contract_key(target)]
            coefficients = result.coefficients.loc[result.coefficients.target.eq(target)].set_index('term').loc[list(cox.TERMS)]
            matrix = result.covariance.loc[result.covariance.target.eq(target) & result.covariance.covariance_type.eq('model')]
            bread = matrix.pivot(index='row_term',columns='column_term',values='value').loc[list(cox.TERMS),list(cox.TERMS)].to_numpy()
            beta = coefficients.coefficient.to_numpy()
            covariance, *_ = independent_sandwich(prepared['frame'], beta, bread)
            expected_se = np.sqrt(np.diag(covariance))
            np.testing.assert_allclose(coefficients.robust_se, expected_se, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(coefficients.ci_low, np.exp(beta-1.96*expected_se), rtol=1e-11)
            np.testing.assert_allclose(coefficients.ci_high, np.exp(beta+1.96*expected_se), rtol=1e-11)
            np.testing.assert_allclose(coefficients.p_value, 2*norm.sf(abs(beta/expected_se)), rtol=1e-11)
            est = result.estimates.loc[result.estimates.target.eq(target)].iloc[0]
            self.assertEqual(est.robust_se, coefficients.robust_se.iloc[0])
            self.assertEqual(est.p_value, coefficients.p_value.iloc[0])
        finite = result.estimates.p_value.dropna().sort_values()
        expected = [min(1., min(len(finite)*finite.iloc[j]/(j+1) for j in range(i,len(finite)))) for i in range(len(finite))]
        np.testing.assert_allclose(result.estimates.loc[finite.index,'fdr_q_value'], expected, rtol=1e-13)
        self.assertEqual(result.metadata['multiplicity']['finite_tests'], 15)

    def test_nonestimable_zero_and_one_event_rows_keep_small_counts_and_nulls(self):
        result = self.run_correction()
        for i in (0,1):
            row = result.estimates.iloc[i]
            self.assertEqual((row.n_children,row.n_intervals,row.n_subsequent_tb), (2,4,i))
            self.assertEqual(row.status, 'non_estimable')
            self.assertTrue(row[['coefficient','robust_se','hr_per_adverse_sd','ci_low','ci_high','p_value','fdr_q_value']].isna().all())
            self.assertFalse(result.metadata['models'][i]['correction_performed'])
        serialized = json.dumps(result.metadata, allow_nan=False)
        self.assertNotIn('synthetic-', serialized); self.assertNotIn('gap-', serialized)
        self.assertNotIn('pid_child', serialized)

    def test_each_missing_or_tampered_artifact_is_rejected(self):
        for name in uncertainty.RESULT_FILES:
            path = self.source/name; original = path.read_bytes()
            with self.subTest(name=name, mutation='missing'):
                path.unlink()
                with self.assertRaises(ValueError): self.run_correction()
                path.write_bytes(original)
            with self.subTest(name=name, mutation='tampered'):
                path.write_bytes(original+b'\nchanged')
                with self.assertRaises(ValueError): self.run_correction()
                path.write_bytes(original)

    def test_repaired_hash_changed_hazard_ratio_is_rejected(self):
        table = pd.read_csv(self.source/'estimates.csv'); table.loc[2,'hr_per_adverse_sd'] *= 1.1
        table.to_csv(self.source/'estimates.csv',index=False,float_format='%.17g'); self.repair_hash('estimates.csv')
        with self.assertRaises((ValueError,ValidationError)): self.run_correction()

    def test_repaired_hash_changed_covariance_or_q_value_is_rejected(self):
        for name,column in [('covariance','value'),('estimates','fdr_q_value')]:
            path = self.source/(name+'.csv'); original = path.read_bytes()
            table = pd.read_csv(path); index = table.index[table[column].notna()][0]
            table.loc[index,column] += .01; table.to_csv(path,index=False,float_format='%.17g')
            self.repair_hash(path.name)
            with self.subTest(name=name), self.assertRaises((ValueError,ValidationError)): self.run_correction()
            path.write_bytes(original); self.repair_hash(path.name)

    def test_nonestimable_reason_cannot_hide_genuine_failure(self):
        table = pd.read_csv(self.source/'estimates.csv',keep_default_na=False)
        table.loc[0,'reason'] = 'numerical_fit_failure'
        table.to_csv(self.source/'estimates.csv',index=False,float_format='%.17g'); self.repair_hash('estimates.csv')
        with self.assertRaises((ValueError,ValidationError)): self.run_correction()

    def test_unexpected_file_symlink_file_and_symlink_publication_rejected(self):
        extra = self.source/'records.csv'; extra.write_text('must reject\n')
        with self.assertRaises(ValueError): self.run_correction()
        extra.unlink()
        alias = self.base/'alias'; alias.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError): uncertainty.calculate(self.prep,alias)
        original = self.source/'summary.json'; target = self.base/'summary.json'; original.rename(target); original.symlink_to(target)
        with self.assertRaises(ValueError): self.run_correction()

    def test_preparation_bytes_or_receipt_selection_changed_rejected(self):
        self.receipt['identity']['selection'] = list(cox.DEFAULT_TARGETS[:-1])
        write_json(self.source/'receipt.json',self.receipt)
        with self.assertRaises(ValueError): self.run_correction()

    def test_unaccepted_or_inconsistent_summary_cannot_authorize_correction(self):
        path = self.source/'summary.json'; original = path.read_bytes()
        for name,value in [('accepted',False),('status','fit_failed'),('accepted',1)]:
            summary = json.loads(original); summary[name] = value; write_json(path,summary)
            self.receipt['summary'] = summary; self.repair_hash(path.name)
            with self.subTest(name=name,value=value), self.assertRaises(ValueError): self.run_correction()

    def test_midcalculation_source_change_blocks_completion(self):
        actual = uncertainty.correct_covariance
        changed = False
        def mutate(*args):
            nonlocal changed
            result = actual(*args)
            if not changed:
                path = self.source/'summary.json'; path.write_bytes(path.read_bytes()+b'\n'); changed = True
            return result
        with mock.patch.object(uncertainty,'correct_covariance',side_effect=mutate), self.assertRaisesRegex(ValueError,'changed'):
            self.run_correction()


def fabricated_figure_registry():
    """Full declared schemas with invented aggregates; no archived data file."""
    specification = json.loads((ROOT/'config/publication_figures.json').read_text())
    schemas = json.loads((ROOT/'config/reporting_schemas.json').read_text())
    records = {}
    def add(source, where, values=None):
        source = source.replace('pretb_corrected/', 'pretb/')
        columns = schemas['schemas'][schemas['sources'][source]]
        row = {c:None for c in columns}
        row.update(where)
        for k in _schema(source):
            if row[k] is None:
                row[k] = 'invented-' + json.dumps(where, sort_keys=True)
        selector = {k:row[k] for k in _schema(source)}
        key = source+'#'+'&'.join(k+'='+quote(json.dumps(v,ensure_ascii=False,separators=(',',':')),safe='') for k,v in selector.items())
        if key in records:
            row.update(records[key]['values'])
        if 'status' in row:
            row['status'] = 'estimated'
        for flag in ('accepted','estimable'):
            if flag in row:
                row[flag] = True
        if 'unit' in row:
            row['unit'] = 'synthetic aggregate unit'
        row.update(values or {})
        row = {k:(v.item() if isinstance(v,np.generic) else v) for k,v in row.items()}
        assert set(row) == set(columns), (source, set(row)-set(columns))
        records[key] = {'key':key,'source':source,'selector':selector,'values':row,
            'status':row.get('status') or 'available','semantics':_semantics(source,row),
            'provenance':{'fingerprint':'b'*64,'fit_fingerprint':'a'*64,'receipt_sha256':'c'*64}}
        return records[key]
    def text(item, unavailable=False):
        values = {f:(None if unavailable else 7) for f in item['fields']}
        return add(item['source'], item['where'], values)
    for figure in specification['figures']:
        for b in figure['caption']['bindings'].values():
            text(b)
        for panel in figure['panels']:
            if panel['kind'] == 'forest':
                for row in panel['rows']:
                    point, low, high = (2.,1.5,2.5) if panel['value_axis']['scale']=='log' else (-.015,-.125,.095)
                    fields = row['fields']
                    values = {fields['point']:point, fields['lower']:low, fields['upper']:high}
                    values.update({f:3 for f in row.get('support_fields',{}).values()})
                    if row['source']=='pretb_corrected/estimates':
                        values.update(coefficient=np.log(point),n_children=9,n_intervals=18,n_subsequent_tb=3,person_years=17.5)
                    add(row['source'],row['where'],values)
            elif panel['kind'] == 'trajectory':
                for group in panel['series_order']:
                    for age in np.linspace(5,10,101):
                        values = {panel['fields']['x']:float(age),panel['fields']['point']:80.+age,
                                  panel['fields']['lower']:79.+age,panel['fields']['upper']:81.+age}
                        add(panel['source'],{**panel['where'],panel['series_key']:group,'age_years':float(age)},values)
            elif panel['kind']=='schedule':
                for i, row in enumerate(panel['row_order']):
                    for j,column in enumerate(panel['column_order']):
                        count = (i+j)%10
                        add(panel['source'],{**panel['where'],panel['row_key']:row,panel['column_key']:column},
                            {panel['fields']['records']:count,panel['fields']['children']:count,
                             panel['fields']['represented']:bool(count)})
            elif panel['kind']=='flow':
                for node in panel['nodes']:
                    for b in node['bindings'].values():
                        text(b,node['id'] in panel['unavailable_nodes'])
    # Required sources not selected by these figures still have complete header
    # and key inventories. Deliberate extra rows test selection, not row order.
    present = {r['source'] for r in records.values()}
    for source in schemas['sources']:
        if source not in present:
            add(source,{k:'unused' for k in _schema(source)})
    counter = 0
    while len(records)<3196:
        add('primary/ratio_pre/estimates',{'contrast':'unused-'+str(counter)})
        counter += 1
    exported = {'schema_version':1,'sources':sorted(schemas['sources']),
                'record_count':len(records),'records':list(records.values())}
    corrected = pd.DataFrame([deepcopy(r['values']) for r in records.values() if r['source']=='pretb/estimates'])
    metadata = {'accepted':True,'status':'accepted','point_estimates_preserved_exactly':True,
                'model_covariance_preserved_exactly':True,'model_fits_performed':0,
                'source_fingerprint':'b'*64,'remaining_limitations':uncertainty.specification()['remaining_conventions']}
    correction = uncertainty.CorrectedPreTBCovariance(corrected,pd.DataFrame(),pd.DataFrame(),metadata)
    return exported, specification, correction


class FigureDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fabricated_figure_registry()

    def setUp(self):
        self.registry,self.spec,self.correction = deepcopy(self.fixture)

    def build(self):
        return figure_data.build_data(self.registry,self.spec,self.correction)

    def selected(self, source, **where):
        return next(r for r in self.registry['records'] if r['source']==source and all(r['values'][k]==v for k,v in where.items()))

    def test_full_nine_figure_seventeen_panel_coverage_and_defensive_copy(self):
        before = deepcopy(self.fixture)
        data = self.build()
        self.assertEqual(data['coverage'], {'figures':9,'panels':17,'forest_estimates':56,
            'trajectory_points':808,'schedule_cells':44,'flow_nodes':20,'flow_count_bindings':28,'caption_count_bindings':12})
        self.assertEqual(tuple(f['id'] for f in data['figures']), figure_data.FIGURE_IDS)
        self.assertEqual(self.registry,before[0]); self.assertEqual(self.spec,before[1])
        self.assertFalse(data['scientific_values_modified_by_rendering'])

    def test_exact_zero_and_all_small_schedule_counts_retained(self):
        cells = self.build()['figures'][4]['panels'][0]['cells']
        self.assertEqual({c['records'] for c in cells}, set(range(10)))
        for row in cells:
            self.assertEqual(row['records'],row['children'])
            self.assertEqual(row['represented'],row['records']>0)
        forest = self.build()['figures'][5]['panels'][0]['rows']
        self.assertTrue(all(row['support']=={'children':9,'events':3} for row in forest))
        self.assertTrue(all('9' in row['support_display'] and '3' in row['support_display'] for row in forest))

    def test_unavailable_flow_is_explicit_and_never_zero_imputed(self):
        panel = self.build()['figures'][3]['panels'][0]
        missing = [n for n in panel['nodes'] if n['status']=='unavailable']
        self.assertEqual({n['id'] for n in missing},set(panel['unavailable_nodes']))
        self.assertEqual(len(missing),4)
        for node in missing:
            self.assertIn('Not available',node['text'])
            self.assertTrue(any(None in b['values'] for b in node['bindings'].values()))

    def test_caption_and_point_bindings_preserve_full_precision_source_values(self):
        data = self.build(); row = data['figures'][1]['panels'][0]['rows'][0]
        self.assertEqual(row['point'],-.015); self.assertEqual(row['lower'],-.125)
        self.assertEqual(row['bindings']['values'],{'point':-.015,'lower':-.125,'upper':.095})
        self.assertIn('-0.02',row['display'])
        self.assertEqual(len(data['figures'][1]['caption']['bindings']),6)
        csv_text = figure_data.points_csv(data)
        self.assertIn('-.015'.lstrip('-'),csv_text)
        self.assertNotIn('pid_child',csv_text)
        self.assertIn('source_field',csv_text.splitlines()[0])

    def test_corrected_hazard_source_is_separate_and_keeps_selected_point_identity(self):
        view = figure_data.AggregateView(self.registry,self.correction)
        old = view.select('pretb/estimates',{'target':'resistance'})
        new = view.select('pretb_corrected/estimates',{'target':'resistance'})
        self.assertEqual(old['values']['hr_per_adverse_sd'],new['values']['hr_per_adverse_sd'])
        self.assertNotEqual(old['key'],new['key'])
        self.assertEqual(new['provenance']['source_result_key'],old['key'])
        self.assertIn('actual-risk-membership',new['semantics']['intervals'][0]['type'])

    def test_corrected_point_count_or_selection_changes_are_rejected(self):
        for column in ('coefficient','hr_per_adverse_sd','n_children','n_intervals','n_subsequent_tb','person_years'):
            before=self.correction.estimates.copy();self.correction.estimates.loc[0,column]+=1
            with self.subTest(column=column),self.assertRaises(ReportingError):self.build()
            self.correction.estimates=before
        self.correction.estimates=self.correction.estimates.iloc[:-1]
        with self.assertRaises(ReportingError):self.build()

    def test_registry_exact_source_key_column_count_and_finite_values_required(self):
        for mode in ('source','key','column','count','nested','infinity','privacy','selector','meaning'):
            self.registry=deepcopy(self.fixture[0]);r=self.registry['records'][0]
            if mode=='source':self.registry['sources'][0]='unapproved/source'
            elif mode=='key':r['key']='wrong'
            elif mode=='column':r['values']['extra']=0
            elif mode=='count':self.registry['record_count']-=1
            elif mode=='nested':r['values'][next(iter(r['values']))]=['invented private label']
            elif mode=='infinity':r['values'][next(iter(r['values']))]=float('inf')
            elif mode=='privacy':r['values']['pid_child']='invented'
            elif mode=='selector':r['selector'][next(iter(r['selector']))]='wrong'
            else:
                r=self.selected('primary/ratio_pre/estimates',contrast='TB <5 weighted')
                r['semantics']['intervals']=[]
            with self.subTest(mode=mode),self.assertRaises(ReportingError):self.build()

    def test_ambiguous_missing_unknown_and_boolean_selectors_rejected(self):
        view = figure_data.AggregateView(self.registry,self.correction)
        for source,where in [('primary/ratio_pre/estimates',{}),('unknown',{'x':1}),
            ('primary/ratio_pre/estimates',{'contrast':'absent'}),
            ('trajectory/predictions',{'threshold':True,'age_years':5,'group':'No recorded TB'}),
            ('primary/ratio_pre/estimates',{'unknown':1})]:
            with self.subTest(source=source,where=where),self.assertRaises(ReportingError):view.select(source,where)
        selected=view.select('primary/ratio_pre/estimates',{'contrast':'TB <5 weighted'})
        selected['values']['mean']=999
        self.assertEqual(view.select('primary/ratio_pre/estimates',{'contrast':'TB <5 weighted'})['values']['mean'],-.015)

    def test_estimator_interval_and_recurrent_process_labels_cannot_be_swapped(self):
        for mode in ('mean_to_median','credible_to_confidence','rr_to_irr','interval_fields'):
            self.spec=deepcopy(self.fixture[1]);panel=self.spec['figures'][1]['panels'][0]
            if mode=='mean_to_median':panel['rows'][0]['fields']['point']='median'
            elif mode=='credible_to_confidence':panel['interval']='95% confidence interval'
            elif mode=='rr_to_irr':self.spec['figures'][2]['panels'][0]['estimate']='posterior median conditional incidence rate ratio'
            else:panel['rows'][0]['fields']['lower']='mean'
            with self.subTest(mode=mode),self.assertRaises(ReportingError):self.build()

    def test_logarithmic_intervals_reject_zero_negative_reversed_or_partial_values(self):
        for values in ([0.,1.,2.],[1.,-1.,2.],[1.,2.,.5],[1.,None,2.],[True,.5,2.]):
            self.registry=deepcopy(self.fixture[0]);r=self.selected('feno/estimates',target='main_12months')
            r['values'].update(dict(zip(('geometric_mean_ratio','ci_low','ci_high'),values)))
            with self.subTest(values=values),self.assertRaises(ReportingError):self.build()

    def test_explicit_nonestimability_retains_row_without_fabricated_position(self):
        r=self.selected('feno/estimates',target='supplement_6to10weeks')
        r['values'].update(geometric_mean_ratio=None,ci_low=None,ci_high=None,status='non_estimable')
        if 'estimable' in r['values']:r['values']['estimable']=False
        r['status']='non_estimable'
        data=self.build();row=data['figures'][6]['panels'][0]['rows'][0]
        self.assertEqual(row['status'],'not_estimable');self.assertIsNone(row['point'])
        self.assertEqual(row['display'],'Not estimable')
        r['status']='available';r['values']['status']='estimated'
        with self.assertRaises(ReportingError):self.build()

    def test_schedule_counts_must_be_integer_nonnegative_and_consistent(self):
        fields=('records_with_numeric_value','children_with_measurement','represented')
        for values in ((1.5,1,True),(-1,0,False),(1,2,True),(0,0,True),(1,1,1),(True,1,True)):
            self.registry=deepcopy(self.fixture[0]);r=next(r for r in self.registry['records'] if r['source']=='preparation/modality_schedule')
            r['values'].update(dict(zip(fields,values)));r['semantics']=_semantics(r['source'],r['values'])
            with self.subTest(values=values),self.assertRaises(ReportingError):self.build()

    def test_trajectory_common_grid_and_group_inventory_are_required(self):
        r=self.selected('trajectory/predictions',target='fev1_fvc',threshold=1,age_years=5.,group='No recorded TB')
        r['values']['age_years']=5.01;r['selector']['age_years']=5.01
        r['key']=r['source']+'#'+'&'.join(k+'='+quote(json.dumps(v,separators=(',',':')),safe='') for k,v in r['selector'].items())
        with self.assertRaises(ReportingError):self.build()

    def test_coverage_cannot_be_satisfied_with_omitted_figures_panels_or_rows(self):
        for mode in ('figure','panel','row','order','duplicate'):
            self.spec=deepcopy(self.fixture[1])
            if mode=='figure':self.spec['figures'].pop()
            elif mode=='panel':self.spec['figures'][0]['panels'].pop()
            elif mode=='row':self.spec['figures'][1]['panels'][0]['rows'].pop()
            elif mode=='order':self.spec['figures'].reverse()
            else:self.spec['figures'][0]['panels'][1]['id']='A'
            with self.subTest(mode=mode),self.assertRaises(ReportingError):self.build()

    def test_caption_placeholders_allow_only_declared_simple_names(self):
        view=figure_data.AggregateView(self.registry,self.correction)
        binding={'source':'preparation/flow','where':{'node':'liveborn_children_enrolled'},'fields':['count'],'style':'integer','digits':0}
        for template in ('{count.__class__}','{count[0]}','{count!r}','{count:>5}','{unknown}'):
            with self.subTest(template=template),self.assertRaises(ReportingError):
                figure_data.template({'template':template,'bindings':{'count':binding}},view)

    def test_nested_undeclared_spec_fields_cannot_carry_private_records(self):
        for location in ('panel','axis','style','row','node'):
            self.spec=deepcopy(self.fixture[1])
            p=self.spec['figures'][1]['panels'][0]
            target={'panel':p,'axis':p['value_axis'],'style':next(iter(p['series_styles'].values())),
                    'row':p['rows'][0],'node':self.spec['figures'][3]['panels'][0]['nodes'][0]}[location]
            target['participant_records']=['invented-private']
            with self.subTest(location=location),self.assertRaises(ReportingError):self.build()

    def test_correction_provenance_and_failed_source_status_are_rejected(self):
        self.correction.metadata['source_fingerprint']='f'*64
        with self.assertRaises(ReportingError):self.build()
        self.correction=deepcopy(self.fixture[2])
        r=self.selected('feno/estimates',target='main_12months')
        r['status']='fit_failed'
        with self.assertRaises(ReportingError):self.build()
        r['values']['status']='fit_failed'
        with self.assertRaises(ReportingError):self.build()

    def test_malformed_registry_and_spec_fail_with_controlled_errors(self):
        for mode in ('record','records','figures','panel'):
            self.registry,self.spec,self.correction=deepcopy(self.fixture)
            if mode=='record':self.registry['records'][0]=None
            elif mode=='records':self.registry['records']=None
            elif mode=='figures':self.spec['figures']=[None]
            else:self.spec['figures'][0]['panels']=[None]
            with self.subTest(mode=mode),self.assertRaises(ReportingError):self.build()


class FigureExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.figure_fixture=fabricated_figure_registry()
        cls.data_fixture=figure_data.build_data(*cls.figure_fixture)

    def setUp(self):
        from dchs_lung_function import figures_execution as ex, execution, pipeline
        from dchs_lung_function.reporting import figures
        from test_reporting import fake_font
        self.ex,self.preparation,self.pipeline,self.renderer=ex,execution,pipeline,figures
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.base=Path(temporary.name).resolve();self.root=self.base/'source'
        for name in ex.source_hashes(ROOT):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        self.work,self.output=self.base/'work',self.base/'output';self.work.mkdir();self.output.mkdir()
        self.runtime=self.base/'document-python';self.runtime.write_text('# fabricated never invoked\n');self.runtime.chmod(0o700)
        self.tool=self.base/'pdffonts';self.tool.write_text('# fabricated never invoked\n');self.tool.chmod(0o700)
        self.fonts={n:self.base/(n+'.ttf') for n in ('arial','arial_bold')}
        for n,p in self.fonts.items():p.write_bytes(fake_font('Bold' if n=='arial_bold' else 'Regular'))
        self.paths=[self.base/n for n in ('paths.json','analysis.json','primary.json','mechanics.json',
            'supporting.json','wheeze.json','sensitivity.json','reporting.json','figures.json')]
        self.local={'schema_version':1,'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)},
                    'fonts':{k:str(v) for k,v in self.fonts.items()}}
        write_json(self.paths[0],self.local)
        for p in self.paths[1:7]:write_json(p,{'synthetic':True})
        write_json(self.paths[7],{'schema_version':1,'document_python':str(self.runtime)})
        write_json(self.paths[8],{'schema_version':1,'pdf_fonts_tool':str(self.tool),
                                'pretb_uncertainty':'corrected_actual_risk_membership'})
        self.doc_runtime={'python':'synthetic','implementation':'cpython','system':'synthetic','machine':'synthetic',
                          'zlib':'synthetic','executable_sha256':file_hash(self.runtime)}
        self.figure_runtime={'synthetic':True,'inspector':file_hash(self.tool)}
        self.prep_fp='d'*64;self.prep=self.work/'preparation_runs'/self.prep_fp;self.prep.mkdir(parents=True)
        self.upstream_file=self.prep/'invented-contract.json';write_json(self.upstream_file,{'invented':True})
        self.manifest={'files':{self.upstream_file.name:file_hash(self.upstream_file)}}
        write_json(self.prep/'artifacts.json',self.manifest);write_json(self.prep/'receipt.json',{'fingerprint':self.prep_fp})
        self.source_identity={'schema_version':1,'synthetic':'accepted Cox source'}
        self.source_fp=ex.digest(self.source_identity);self.source=self.output/'pretb_cox'/self.source_fp;self.source.mkdir(parents=True)
        source_summary={'accepted':True,'status':'accepted'}
        for n in pretb_execution.RESULT_FILES:
            if n=='summary.json':write_json(self.source/n,source_summary)
            else:(self.source/n).write_text('{}\n' if n.endswith('.json') else 'synthetic\n')
        write_json(self.source/'receipt.json',{'schema_version':1,'identity':self.source_identity,'fingerprint':self.source_fp,
            'summary':source_summary,'files':{n:file_hash(self.source/n) for n in pretb_execution.RESULT_FILES}})
        self.registry,self.spec,self.correction=deepcopy(self.figure_fixture)
        self.data=deepcopy(self.data_fixture)
        self.coverage={'numbered_tables':8,'physical_tables':10,'numeric_cells':304,'scientific_cells':261,'bound_context_cells':3,
            'cell_classifications':{'estimate':186,'count':75,'header':26,'context':17},'sentences':42,'value_tokens':96,'qualitative_bindings':52}
        upstream={'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,
            'mechanics_models_verified':6,'shared_factor_rows_verified':6,'supporting_joint_models_verified':7,
            'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'wheeze_models_verified':2,'feno_models_verified':11}
        report_identity={'schema_version':1,'source':ex.source_hashes(self.root),'preparation_fingerprint':self.prep_fp,
            'document_runtime':self.doc_runtime,'references':{},'fonts':{k:file_hash(v) for k,v in self.fonts.items()},
            'source_bindings':[{'family':'pretb','fingerprint':self.source_fp,'receipt_sha256':file_hash(self.source/'receipt.json')}]}
        report_fp=ex.digest(report_identity);self.publication=self.output/'manuscript_reporting'/report_fp;self.publication.mkdir(parents=True)
        summary={'accepted':True,'status':'accepted','through':'M13','fingerprint':report_fp,'preparation_fingerprint':self.prep_fp,
            'registry_records':3196,'aggregate_sources':143,'new_model_fits':0,'coverage':self.coverage,
            'preceding_modules_verified':12,'source_publications_verified':45,'upstream_models':upstream}
        self.prior={**summary,'resumed':True}
        for n in ex.REPORTING_FILES:
            if n=='registry.json':write_json(self.publication/n,self.registry)
            elif n=='summary.json':write_json(self.publication/n,summary)
            else:(self.publication/n).write_text('{}\n' if n.endswith('.json') else 'fabricated\n')
        write_json(self.publication/'receipt.json',{'schema_version':1,'status':'accepted','identity':report_identity,
            'fingerprint':report_fp,'summary':summary,'files':{n:file_hash(self.publication/n) for n in ex.REPORTING_FILES},'completed_at':'synthetic'})
        self.calls=[]
        def upstream_call(*args,**kwargs):self.calls.append('M13');return deepcopy(self.prior)
        def calculate(*args):self.calls.append('correction');self.assertIn('M13',self.calls);return deepcopy(self.correction)
        def render(data,destination,fonts,**kwargs):
            self.calls.append('render');self.assertIn('correction',self.calls)
            self.assertEqual(kwargs,{'dpi':600})
            for n in ex.IMAGE_FILES:(destination/n).write_bytes(('FABRICATED IMAGE BYTES '+n).encode())
            return {'files':{n:file_hash(destination/n) for n in ex.IMAGE_FILES}}
        self.render=render
        patches=[mock.patch.object(ex,'run_reporting',side_effect=upstream_call),
            mock.patch.object(ex,'document_runtime',return_value=self.doc_runtime),
            mock.patch.object(ex,'font_tool_identity',return_value={'version':'pdffonts version synthetic'}),
            mock.patch.object(ex,'figure_runtime',return_value=self.figure_runtime),
            mock.patch.object(ex,'verify_artifacts',return_value=self.manifest),
            mock.patch.object(ex,'build_data',return_value=self.data),
            mock.patch.object(pretb_execution,'contract_descriptor',return_value={'contracts':{},'preparation_fingerprint':self.prep_fp}),
            mock.patch.object(pretb_execution,'fit_identity',return_value=(self.source_fp,self.source_identity)),
            mock.patch.object(uncertainty,'calculate',side_effect=calculate),
            mock.patch.object(execution,'run_preparation',return_value={'fingerprint':self.prep_fp}),
            mock.patch.object(figures,'render_figures',side_effect=render),
            mock.patch.object(ex,'verify_pdf_fonts',return_value={'status':'passed','only_actual_Arial':True,'pdf_count':10})]
        for patch in patches:patch.start();self.addCleanup(patch.stop)

    def run_figures(self,**kwargs):return self.ex.run_figures(self.root,*self.paths,**kwargs)
    def destination(self,r):return self.output/'publication_figures'/r['fingerprint']
    def clean(self):
        self.assertEqual(list(self.output.rglob('.staging-*')),[])
        self.assertEqual(list(self.base.rglob('.preparation.lock')),[])
    def no_pointer(self):self.assertFalse((self.work/'figures_current.json').exists());self.clean()
    def repair(self,directory,name):
        r=json.loads((directory/'receipt.json').read_text());r['files'][name]=file_hash(directory/name);write_json(directory/'receipt.json',r)

    def test_success_and_resume_revalidate_upstream_and_regenerate_all_thirty_payloads(self):
        first=self.run_figures();second=self.run_figures(resume=True)
        self.assertFalse(first['resumed']);self.assertTrue(second['resumed']);self.assertEqual(first['fingerprint'],second['fingerprint'])
        self.assertEqual(self.calls,['M13','correction','render']*2)
        self.assertEqual(len(self.ex.RESULT_FILES),30);self.assertEqual(len(self.ex.IMAGE_FILES),19)
        self.assertEqual({p.name for p in self.destination(first).iterdir()},set(self.ex.RESULT_FILES)|{'receipt.json'})
        self.assertEqual(first['new_model_fits'],0);self.assertEqual(first['corrected_pretb_models'],17);self.clean()

    def test_all_upstream_counts_and_typed_success_are_required(self):
        original=deepcopy(self.prior)
        cases=[('accepted',False),('resumed',1),('new_model_fits',False),('registry_records',3194),
            ('aggregate_sources',141),('source_publications_verified',44),('preceding_modules_verified',11),
            ('fingerprint','../invalid'),('preparation_fingerprint','x'*64)]
        for key,value in cases:
            self.prior=deepcopy(original);self.prior[key]=value
            with self.subTest(key=key),self.assertRaises(ValidationError):self.run_figures()
        for key in original['upstream_models']:
            self.prior=deepcopy(original);self.prior['upstream_models'][key]-=1
            with self.subTest(key=key),self.assertRaises(ValidationError):self.run_figures()
        self.no_pointer()

    def test_invalid_jobs_resume_and_source_tree_fail_before_upstream(self):
        for kwargs in ({'jobs':True},{'jobs':0},{'jobs':4},{'resume':1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValidationError):self.run_figures(**kwargs)
        (self.root/'README.md').write_text('changed source\n')
        with self.assertRaises(ValidationError):self.run_figures()
        self.assertEqual(self.calls,[]);self.no_pointer()

    def test_explicit_resume_and_independent_output_lock_required(self):
        self.run_figures()
        with self.assertRaises(ValidationError):self.run_figures()
        with self.ex.preparation_locks(self.work/'publication_figures',self.output/'publication_figures'):
            with self.assertRaises(ValidationError):self.run_figures(resume=True)
        self.clean()

    def test_every_repaired_hash_payload_change_rejected_against_regeneration(self):
        directory=self.destination(self.run_figures());receipt=(directory/'receipt.json').read_bytes()
        pointer=(self.work/'figures_current.json').read_bytes()
        for name in self.ex.RESULT_FILES:
            path=directory/name;before=path.read_bytes();path.write_bytes(before+b'changed');self.repair(directory,name)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_figures(resume=True)
            path.write_bytes(before);(directory/'receipt.json').write_bytes(receipt)
        self.assertEqual((self.work/'figures_current.json').read_bytes(),pointer);self.clean()

    def test_missing_extra_and_contradictory_receipt_block_cached_success(self):
        directory=self.destination(self.run_figures());path=directory/'receipt.json';before=path.read_bytes()
        for key,value in [('schema_version',True),('status','failed'),('summary',{'accepted':True}),('extra','invented private record')]:
            record=json.loads(before);record[key]=value;write_json(path,record)
            with self.subTest(key=key),self.assertRaises(ValidationError):self.run_figures(resume=True)
            path.write_bytes(before)
        extra=directory/'extra.json';extra.write_text('{}')
        with self.assertRaises(ValidationError):self.run_figures(resume=True)
        extra.unlink();artifact=directory/self.ex.RESULT_FILES[0];data=artifact.read_bytes();artifact.unlink()
        with self.assertRaises(ValidationError):self.run_figures(resume=True)
        artifact.write_bytes(data);self.clean()

    def test_symlink_receipt_payload_parent_and_dangling_destination_rejected(self):
        directory=self.destination(self.run_figures())
        for name in ('receipt.json','plot_data.json'):
            path=directory/name;external=self.base/name;path.rename(external);path.symlink_to(external)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_figures(resume=True)
            path.unlink();external.rename(path)
        external=self.base/'moved';directory.rename(external);directory.symlink_to(external,target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_figures(resume=True)
        directory.unlink();directory.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_figures(resume=True)

    def test_unsafe_roots_and_lexical_parent_paths_cannot_contain_inputs(self):
        for work,output in ((self.root/'generated',self.output),(self.work,self.work/'nested'),(self.base,self.output)):
            local=deepcopy(self.local);local['directories']={'work':str(work),'outputs':str(output)};write_json(self.paths[0],local)
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_figures()
        write_json(self.paths[0],self.local)
        p=self.work/'runtime';p.write_text('synthetic');p.chmod(0o700)
        write_json(self.paths[7],{'schema_version':1,'document_python':str(self.work/'dummy/../runtime')})
        with self.assertRaises(ValidationError):self.run_figures()
        self.assertEqual(self.calls,[]);self.no_pointer()

    def test_changed_config_font_source_or_upstream_blocks_publication(self):
        original=self.ex.render_payloads
        for path in (self.paths[1],self.fonts['arial'],self.upstream_file,self.source/'diagnostics.json',self.root/'README.md'):
            before=path.read_bytes()
            def mutate(*args):original(*args);path.write_bytes(before+b'changed')
            with mock.patch.object(self.ex,'render_payloads',side_effect=mutate):
                with self.subTest(path=path.name),self.assertRaises(ValidationError):self.run_figures()
            path.write_bytes(before);self.no_pointer()
        self.assertEqual(list((self.output/'publication_figures').glob('*/receipt.json')),[])

    def test_changed_runtime_or_preparation_before_completion_rejected(self):
        with mock.patch.object(self.ex,'figure_runtime',side_effect=[self.figure_runtime,{'changed':True}]),self.assertRaises(ValidationError):self.run_figures()
        with mock.patch.object(self.ex,'document_runtime',side_effect=[self.doc_runtime,{'changed':True}]),self.assertRaises(ValidationError):self.run_figures()
        with mock.patch.object(self.preparation,'run_preparation',return_value={'fingerprint':'0'*64}),self.assertRaises(ValidationError):self.run_figures()
        self.no_pointer()

    def test_postrename_failure_rolls_back_fresh_but_preserves_cached_publication(self):
        verify=self.ex.verify_publication;before=self.upstream_file.read_bytes()
        def mutate(*args):verify(*args);self.upstream_file.write_bytes(before+b'changed')
        with mock.patch.object(self.ex,'verify_publication',side_effect=mutate),self.assertRaises(ValidationError):self.run_figures()
        self.assertEqual(list((self.output/'publication_figures').glob('*/receipt.json')),[]);self.no_pointer()
        self.upstream_file.write_bytes(before);accepted=self.run_figures();pointer=(self.work/'figures_current.json').read_bytes()
        with mock.patch.object(self.ex,'verify_publication',side_effect=mutate),self.assertRaises(ValidationError):self.run_figures(resume=True)
        self.assertTrue(self.destination(accepted).is_dir());self.assertEqual((self.work/'figures_current.json').read_bytes(),pointer);self.clean()

    def test_destination_race_and_pointer_failure_cannot_publish_success(self):
        verify=self.ex.verify_publication;created=[]
        def race(directory,identity,*args):
            verify(directory,identity,*args);p=self.output/'publication_figures'/self.ex.digest(identity);p.mkdir();created.append(p)
        with mock.patch.object(self.ex,'verify_publication',side_effect=race),self.assertRaises(ValidationError):self.run_figures()
        self.assertEqual(list(created[0].iterdir()),[]);created[0].rmdir();self.no_pointer()
        with mock.patch.object(self.ex,'write_pointer',side_effect=OSError('fabricated write failure')),self.assertRaises(OSError):self.run_figures()
        self.assertEqual(list((self.output/'publication_figures').glob('*/receipt.json')),[]);self.no_pointer()

    def test_renderer_missing_extra_or_wrong_hash_cannot_publish(self):
        for mode in ('missing','extra','hash'):
            def bad(data,destination,fonts,**kwargs):
                result=self.render(data,destination,fonts,**kwargs)
                if mode=='missing':(destination/self.ex.IMAGE_FILES[0]).unlink()
                elif mode=='extra':(destination/'participant_records.csv').write_text('invented-only\n')
                else:result['files'][self.ex.IMAGE_FILES[0]]='0'*64
                return result
            with mock.patch.object(self.renderer,'render_figures',side_effect=bad):
                with self.subTest(mode=mode),self.assertRaises((ValidationError,OSError)):self.run_figures()
            self.no_pointer()

    def test_invalid_predecessor_receipt_artifacts_or_selected_cox_source_rejected(self):
        receipt=self.publication/'receipt.json';before=receipt.read_bytes()
        changed=json.loads(before);changed['identity']['source_bindings'][0]['fingerprint']='e'*64;write_json(receipt,changed)
        with self.assertRaises(ValidationError):self.run_figures()
        receipt.write_bytes(before)
        artifact=self.publication/'registry.json';data=artifact.read_bytes();artifact.write_bytes(data+b'changed')
        with self.assertRaises(ValidationError):self.run_figures()
        artifact.write_bytes(data)
        (self.source/'receipt.json').unlink()
        with self.assertRaises(ValidationError):self.run_figures()
        self.no_pointer()


class FigureConfigurationCliAndRendererTests(unittest.TestCase):
    def setUp(self):
        from dchs_lung_function import figures_execution,pipeline
        from dchs_lung_function.reporting import figures
        self.ex,self.pipeline,self.renderer=figures_execution,pipeline,figures
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.base=Path(t.name).resolve()
        self.tool=self.base/'pdffonts';self.tool.write_text('# fabricated executable\n');self.tool.chmod(0o700)
        self.path=self.base/'config.json'
        self.config={'schema_version':1,'pdf_fonts_tool':'pdffonts','pretb_uncertainty':'corrected_actual_risk_membership'}
        write_json(self.path,self.config)

    def test_configuration_requires_explicit_corrected_method_and_executable(self):
        self.assertEqual(self.ex.figures_config(self.path)['pdf_fonts_tool'],self.tool)
        for key,value in [('schema_version',True),('pretb_uncertainty','original'),('pdf_fonts_tool','missing'),('extra','private')]:
            write_json(self.path,{**self.config,key:value})
            with self.subTest(key=key),self.assertRaises(ValidationError):self.ex.figures_config(self.path)
        write_json(self.path,self.config);self.tool.chmod(0o600)
        with self.assertRaises(ValidationError):self.ex.figures_config(self.path)

    def test_font_tool_identity_is_hashed_and_process_failures_sanitized(self):
        with mock.patch.object(self.ex.subprocess,'run',return_value=mock.Mock(stdout='',stderr='pdffonts version 25.test\n')):
            value=self.ex.font_tool_identity(self.tool)
        self.assertEqual(value,{'version':'pdffonts version 25.test','executable_sha256':file_hash(self.tool)})
        for error in (OSError('invented private record'),subprocess.TimeoutExpired('private',20)):
            with mock.patch.object(self.ex.subprocess,'run',side_effect=error),self.assertRaisesRegex(ValidationError,'preflight failed'):
                self.ex.font_tool_identity(self.tool)
        with mock.patch.object(self.ex.subprocess,'run',return_value=mock.Mock(stdout='other tool',stderr='')),self.assertRaises(ValidationError):
            self.ex.font_tool_identity(self.tool)

    def cli(self,args,result=None,error=None):
        out,err=StringIO(),StringIO()
        with mock.patch.object(self.ex,'run_figures',return_value=result,side_effect=error) as called,redirect_stdout(out),redirect_stderr(err):
            status=self.pipeline.main(args,root=ROOT)
        return status,out.getvalue(),err.getvalue(),called

    def test_cli_routes_all_sampling_reporting_and_figure_controls(self):
        args=['--through','M14','--config','paths.json','--analysis','analysis.json','--sampling','a.json',
            '--mechanics-sampling','b.json','--supporting-sampling','c.json','--wheeze-sampling','d.json',
            '--sensitivity-sampling','e.json','--reporting','r.json','--figures','f.json','--jobs','3','--resume']
        result={'accepted':True,'status':'accepted','coverage':{'figures':9,'panels':17}}
        code,out,err,called=self.cli(args,result)
        self.assertEqual(code,0);self.assertIn('9 figures and 17 panels',out)
        self.assertEqual(called.call_args.args[1:],('paths.json','analysis.json','a.json','b.json','c.json','d.json','e.json','r.json','f.json'))
        self.assertEqual(called.call_args.kwargs,{'jobs':3,'resume':True})

    def test_cli_refuses_missing_configs_partial_models_and_unavailable_modes(self):
        for args in (['--through','M14','--config','x','--analysis','y','--reporting','r'],
            ['--through','M14','--config','x','--analysis','y','--figures','f'],
            ['--through','M14','--config','x','--analysis','y','--reporting','r','--figures','f','--targets','ratio_pre'],
            ['--through','M13','--figures','f'],['--full'],['--render']):
            with self.subTest(args=args):
                code,_,_,called=self.cli(args);self.assertEqual(code,2);called.assert_not_called()

    def test_cli_sanitizes_failed_model_or_renderer_messages(self):
        args=['--through','M14','--config','x','--analysis','y','--reporting','r','--figures','f']
        for error in (ValueError('invented private record'),RuntimeError('invented private record'),OSError('invented private record')):
            code,out,err,_=self.cli(args,error=error)
            self.assertEqual(code,2);self.assertNotIn('invented private record',out+err)

    def test_renderer_limits_include_points_outside_intervals_and_never_clip(self):
        rows=[{'point':2.,'lower':.5,'upper':1.5},{'point':None,'lower':None,'upper':None}]
        axis={'scale':'log','null':1,'ticks':[.25,4]}
        bounds=self.renderer._limits(rows,axis)
        self.assertLess(bounds[0],.25);self.assertGreater(bounds[1],4)
        with self.assertRaises(ValueError):self.renderer._limits(rows,{**axis,'limits':[.5,2.]})
        with self.assertRaises(ValueError):self.renderer._limits([{'point':0.,'lower':-.1,'upper':1.}],axis)
        self.assertFalse(self.renderer._drawable(rows[1]))
        for row in ({'point':1.,'lower':None,'upper':2.},{'point':True,'lower':.5,'upper':2.}):
            with self.assertRaises(ValueError):self.renderer._drawable(row)

    def test_renderer_subscripts_use_arial_digit_math_without_unicode_fallback(self):
        result=self.renderer._text('FEV₁/FVC and R₆–R₁₈ − 1')
        self.assertEqual(result,'FEV$_{1}$/FVC and R$_{6}$-R$_{18}$ - 1')
        self.assertNotIn('₁',result);self.assertNotIn('−',result)

    def test_shared_trajectory_axes_use_complete_intervals_without_mutating_inputs(self):
        spec=json.loads((ROOT/'config/publication_figures.json').read_text())['figures'][0]
        for i,panel in enumerate(spec['panels']):
            panel['points']=[{'point':float(i),'lower':float(i)-1,'upper':float(i)+2}]
        spec['caption']={'text':'Fabricated caption','bindings':{}}
        before=deepcopy(spec);canvas=mock.MagicMock()
        canvas.wrap.side_effect=lambda text,*args:text
        with mock.patch.object(self.renderer,'_Canvas',return_value=canvas),mock.patch.object(self.renderer,'_trajectory') as draw:
            self.renderer._build_figure(spec,{})
        panels=[call.args[1] for call in draw.call_args_list]
        for left,right in [(0,1),(2,3)]:
            self.assertEqual(panels[left]['y_axis']['limits'],panels[right]['y_axis']['limits'])
            self.assertLess(panels[left]['y_axis']['limits'][0],left-1)
            self.assertGreater(panels[left]['y_axis']['limits'][1],right+2)
        self.assertEqual(spec,before)

    def test_selected_figure_styles_expose_reference_and_pointwise_mean_bands(self):
        figures=json.loads((ROOT/'config/publication_figures.json').read_text())['figures']
        trajectory=figures[0]
        self.assertTrue(all(p['series_styles']['No recorded TB']['line']=='dashed' for p in trajectory['panels']))
        self.assertIn('pointwise, not simultaneous',trajectory['caption']['template'])
        self.assertTrue(all(p['annotation']['show'] is True for p in figures[6]['panels']))

    def test_export_resolution_caption_content_bounds_are_checked(self):
        fonts={'arial':Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
               'arial_bold':Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')}
        if not all(p.is_file() for p in fonts.values()):self.skipTest('Actual external Arial files are required for this font-metric integration check.')
        canvas=self.renderer._Canvas(4,3,fonts)
        text=canvas.wrap('A synthetic caption with enough words to exercise physical wrapping.',3,9.2)
        canvas.caption_artist=canvas.label(.5,2.5,text,size=9.2)
        audited=canvas.audit()
        self.assertEqual(audited['measurement_dpi'],600)
        self.assertGreaterEqual(audited['caption_right_margin_inches'],.5)
        canvas.caption_artist.set_text('Unwrapped text '*20)
        with self.assertRaisesRegex(ValueError,'outside'):canvas.audit()
        canvas.fig.clear()

    def test_rendering_refuses_wrong_dpi_or_order_before_any_figures_are_created(self):
        for dpi in (300,True,600.):
            with self.subTest(dpi=dpi),self.assertRaises(ValueError):
                self.renderer.render_figures({'schema_version':1,'figures':[]},self.base/'out',{},dpi=dpi)
        with self.assertRaises(ValueError):
            self.renderer.render_figures({'schema_version':1,'figures':[]},self.base/'out',{})
        self.assertFalse((self.base/'out').exists())

    def font_inspection(self, mode='valid'):
        from test_reporting import fake_font
        fonts={n:self.base/(n+'.ttf') for n in ('arial','arial_bold')}
        for n,p in fonts.items():p.write_bytes(fake_font('Bold' if n=='arial_bold' else 'Regular'))
        header='name                                 type              encoding         emb sub uni object ID'
        rows=[]
        for family in ('ArialMT','Arial-BoldMT'):
            if mode=='missing_bold' and family=='Arial-BoldMT':continue
            line=[' ']*110
            values={0:'ABCDEF+'+('DejaVuSans' if mode=='fallback' else family),
                header.index('type'):'Type 3' if mode=='type3' else 'CID TrueType',
                header.index('encoding'):'Identity-H',header.index('emb'):'no' if mode=='unembedded' else 'yes',
                header.index('sub'):'yes',header.index('uni'):'no' if mode=='nonunicode' else 'yes',header.index('object ID'):'10 0'}
            for position,value in values.items():line[position:position+len(value)]=value
            rows.append(''.join(line))
        stdout=header+'\n'+'-'*110+'\n'+'\n'.join(rows)+'\n'
        def ttfont(path):
            instance=mock.MagicMock();instance.__enter__.return_value=instance
            instance.__getitem__.return_value.getDebugName.return_value='Arial-BoldMT' if Path(path).stem=='arial_bold' else 'ArialMT'
            return instance
        with mock.patch('fontTools.ttLib.TTFont',side_effect=ttfont),mock.patch.object(self.ex.subprocess,'run',return_value=mock.Mock(stdout=stdout)) as calls:
            result=self.ex.verify_pdf_fonts(self.base,self.tool,fonts)
        return result,calls

    def test_font_inspector_checks_both_actual_faces_in_all_ten_pdfs(self):
        result,calls=self.font_inspection()
        self.assertEqual(result['pdf_count'],10);self.assertTrue(result['only_actual_Arial'])
        self.assertEqual({p['file'] for p in result['pdfs']},{n for n in self.ex.IMAGE_FILES if n.endswith('.pdf')})
        self.assertEqual(calls.call_count,10)
        for pdf in result['pdfs']:
            self.assertEqual({f['postscript_name'] for f in pdf['fonts']},{'ArialMT','Arial-BoldMT'})

    def test_fallback_missing_bold_unembedded_nonunicode_and_type3_fonts_rejected(self):
        for mode in ('missing_bold','fallback','unembedded','nonunicode','type3'):
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.font_inspection(mode)
