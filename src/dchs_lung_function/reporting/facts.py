"""Aggregate numerical-text facts derived from accepted preparation artifacts.

Only fixed aggregate schemas leave this module. Source identifiers are used to
check child uniqueness and are never returned. Existing model estimates and
published cohort counts remain owned by their preceding modules.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..execution import load_artifacts


AGE_COLUMNS = ['population', 'diagnosis_group', 'unit', 'statistic', 'n_children',
               'n_nonmissing', 'n_missing', 'median', 'q1', 'q3',
               'quantile_method', 'status']
DISTRIBUTION_COLUMNS = ['fact', 'numerator', 'denominator', 'percent', 'unit', 'status']
UNAVAILABLE_FLOW_COLUMNS = ['node', 'count', 'unit', 'status', 'evidence']
DIAGNOSIS_GROUPS = ('no_recorded_tb', 'infancy', 'age_one_to_under_five', 'age_five_or_later')


def _validate_descriptive(frame):
    required = {'pid_child', 'tb_age_first_years', 'tb_diagnosis_group'}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError('Descriptive cohort lacks required numerical-fact fields.')
    identifiers = frame['pid_child'].astype('string').str.strip()
    if identifiers.isna().any() or identifiers.eq('').any() or identifiers.duplicated().any():
        raise ValueError('Descriptive numerical facts require unique nonblank children.')
    groups = frame['tb_diagnosis_group'].astype('string')
    if groups.isna().any() or not groups.isin(DIAGNOSIS_GROUPS).all():
        raise ValueError('Descriptive diagnosis groups are invalid.')
    ages = pd.to_numeric(frame['tb_age_first_years'], errors='coerce')
    values = ages.to_numpy(dtype=float, na_value=np.nan)
    if (frame['tb_age_first_years'].notna() & ages.isna()).any() or np.isinf(values).any():
        raise ValueError('Descriptive diagnosis ages contain invalid values.')
    known = groups.ne('no_recorded_tb').to_numpy(dtype=bool)
    if not np.array_equal(np.isfinite(values), known) or (values[known] < 0).any():
        raise ValueError('Descriptive diagnosis age availability disagrees with group.')
    expected = np.select([values < 1, (values >= 1) & (values < 5), values >= 5],
                         ['infancy', 'age_one_to_under_five', 'age_five_or_later'],
                         default='no_recorded_tb')
    if not np.array_equal(expected, groups.to_numpy(dtype=str)):
        raise ValueError('Descriptive diagnosis ages disagree with their established groups.')
    return values, groups


def derive_facts(prep):
    """Return aggregate frames registered as ``facts/<returned key>``.

    ``prep`` is an accepted private preparation directory. Diagnosis ages use
    the stored descriptive-cohort definition. Quantiles use linear
    interpolation on one age per child with first TB strictly before age five.
    The combined antenatal disposition count remains explicitly unavailable;
    component counts alone do not establish a union without overlap evidence.
    """
    prep = Path(prep)
    children = load_artifacts(prep, ('M02', 'cohorts', 'descriptive_children'))
    ages, groups = _validate_descriptive(children)
    early = ages[np.isfinite(ages) & (ages < 5)]
    n_early = len(early)
    q1, median, q3 = (np.quantile(early, [.25, .5, .75], method='linear')
                      if n_early else (np.nan, np.nan, np.nan))
    age_frame = pd.DataFrame([{
        'population': 'descriptive', 'diagnosis_group': 'tb_before_five',
        'unit': 'years', 'statistic': 'median_iqr', 'n_children': n_early,
        'n_nonmissing': n_early, 'n_missing': 0,
        'median': float(median), 'q1': float(q1), 'q3': float(q3),
        'quantile_method': 'linear', 'status': 'estimated' if n_early else 'empty_group',
    }], columns=AGE_COLUMNS)
    proportions = [
        ('tb_before_five_among_cohort', n_early, len(children)),
        ('infancy_among_early_tb', int(groups.eq('infancy').sum()), n_early),
        ('age_one_to_under_five_among_early_tb', int(groups.eq('age_one_to_under_five').sum()), n_early),
    ]
    distribution = pd.DataFrame([{
        'fact': fact, 'numerator': numerator, 'denominator': denominator,
        'percent': 100.0 * numerator / denominator if denominator else np.nan,
        'unit': 'percent', 'status': 'estimated' if denominator else 'undefined_zero_denominator',
    } for fact, numerator, denominator in proportions], columns=DISTRIBUTION_COLUMNS)
    flow = load_artifacts(prep, ('M02', 'flow'))
    if not isinstance(flow, pd.DataFrame) or not {'node', 'count', 'status'}.issubset(flow.columns):
        raise ValueError('Accepted preparation lacks the required cohort flow inventory.')
    components = flow.loc[flow['node'].isin(('antenatal_loss_to_followup', 'pregnancy_loss'))]
    if len(components) != 2 or components['node'].duplicated().any():
        raise ValueError('Accepted preparation has an incomplete antenatal disposition inventory.')
    unavailable = pd.DataFrame([{
        'node': 'antenatal_loss_or_pregnancy_loss', 'count': None, 'unit': 'women',
        'status': 'not_reconstructable_from_supplied_inputs',
        'evidence': 'Combined antenatal disposition and component-overlap evidence are absent from accepted preparation.',
    }], columns=UNAVAILABLE_FLOW_COLUMNS)
    return {'early_tb_age': age_frame, 'early_tb_distribution': distribution,
            'unavailable_flow': unavailable}
