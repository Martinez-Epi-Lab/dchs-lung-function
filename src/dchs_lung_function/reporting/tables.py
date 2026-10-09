"""Selected manuscript table layouts built only from keyed aggregate results.

Rows preserve the accepted manuscript's physical, unique-cell coordinates.
Merged cells carry ``colspan``; those coordinates are not expanded-grid indices.
Numerical values are always selected from the result registry and formatted at
the last step. The table specifications contain no study estimates or counts.
"""
from __future__ import annotations

from .formatting import cell, literal


SPIROMETRY = (
    ('FEV₁/FVC', 'Percentage points', 'ratio_{stage}', 2),
    ('FEV₁/FVC', 'GLI z-score', 'ratio_{stage}_z', 2),
    ('FEV₁', 'L', 'fev1_{stage}_native', 3),
    ('FEV₁', 'GLI z-score', 'fev1_{stage}_z', 2),
    ('FVC', 'L', 'fvc_{stage}_native', 3),
    ('FVC', 'GLI z-score', 'fvc_{stage}_z', 2),
)
AGE_CONTRASTS = ('TB <1', 'TB 1-<5', 'TB >=5', 'TB <5 weighted', 'Any TB weighted')
AGE_HEADERS = ('TB <1 year', 'TB 1 to <5 years', 'TB ≥5 years', 'TB <5 years*', 'Any recorded TB*')
SENSITIVITY_CONTRASTS = ('First TB <5 years', 'First TB <1 year')
SENSITIVITY_MODELS = (
    ('early_core', 'Core model'),
    ('early_maternal_hiv', 'Core + maternal HIV'),
    ('early_maternal_smoking', 'Core + prenatal maternal smoking'),
    ('early_birthweight', 'Core + birthweight'),
    ('early_all_baseline', 'Core + maternal HIV, maternal smoking, and birthweight'),
)


def _literal(text, column, colspan=1):
    return {**literal(text), 'reference_column': column, 'colspan': colspan}


def _value(registry, source, selector, fields, column, *, style='interval', digits=2,
           colspan=1, prefix='', suffix=''):
    result = cell(registry.select(source, **selector), fields, style=style, digits=digits)
    result.update(reference_column=column, colspan=colspan)
    if prefix or suffix:
        result.update(prefix=prefix, suffix=suffix, text=prefix + result['text'] + suffix)
    return result


def _row(identity, number, cells, kind='data'):
    return {'id': identity, 'reference_row': number, 'kind': kind, 'cells': cells}


def _header(number, labels, spans=None):
    spans = spans or [1] * len(labels)
    return _row('header_' + str(number), number,
                [_literal(label, index, span) for index, (label, span) in enumerate(zip(labels, spans), 1)],
                'header')


def _section(identity, number, label, width):
    return _row(identity, number, [_literal(label, 1, width)], 'section')


def _table(identity, title, document, locator, columns, rows, footnotes, header_rows=1):
    return {'id': identity, 'title': title, 'reference_document': document,
            'reference_table': locator, 'header_rows': header_rows,
            'columns': [{'id': name, 'label': label} for name, label in columns],
            'rows': rows, 'footnotes': footnotes}


def _table_one(registry):
    groups = (('overall', 'Overall', 'combined_eligible_lung_function'),
              ('no_tb', 'No TB before age 5', 'no_tb_before_five'),
              ('tb_lt5', 'First TB <5 years', 'tb_before_five'))
    header = [_literal('Characteristic', 1)]
    for column, (_, label, node) in enumerate(groups, 2):
        header.append(_value(registry, 'preparation/flow', {'node': node}, ['count'], column,
                             style='integer', digits=0, prefix=label + '\n(N = ', suffix=')'))
    rows = [_row('header_1', 1, header, 'header'),
            _row('spacing_2', 2, [_literal('', i) for i in range(1, 5)], 'spacer')]
    specifications = (
        (3, None, 'Child characteristics at birth', 'section'),
        (4, 'male', 'Male sex', 'count_percent'),
        (5, 'mbekweni', 'Recruitment site: Mbekweni', 'count_percent'),
        (6, 'black_african', 'Black African ethnicity', 'count_percent'),
        (7, 'gestational_age_weeks', 'Mean gestational age, weeks (SD)', 'mean_sd'),
        (8, 'preterm_bin', 'Preterm birth (<37 weeks)', 'count_percent'),
        (9, 'birthweight_g', 'Mean birthweight, grams (SD)', 'mean_sd'),
        (10, 'low_birthweight_bin', 'Low birthweight (<2500 g)', 'count_percent'),
        (11, 'hiv_exposed_bin', 'HIV exposure at birth', 'count_percent'),
        (12, 'hiv_infected_bin', 'Child HIV infection', 'count_percent'),
        (13, None, 'Maternal and pregnancy characteristics', 'section'),
        (14, 'maternal_age_birth', 'Mean maternal age at childbirth, years (SD)', 'mean_sd'),
        (15, None, 'Maternal education', 'category_heading'),
        (16, 'education:0', 'No education', 'count_percent'),
        (17, 'education:1', 'Primary school only', 'count_percent'),
        (18, 'education:2', 'Some secondary school', 'count_percent'),
        (19, 'education:3', 'Completed secondary school', 'count_percent'),
        (20, 'prenatal_smoking_bin', 'Smoking during pregnancy', 'count_percent'),
        (21, 'prenatal_alcohol_bin', 'Alcohol use during pregnancy', 'count_percent'),
        (22, 'parity', 'Parity, median (IQR)', 'median_iqr'),
        (23, None, 'Household characteristics at enrollment', 'section'),
        (24, None, 'Number of people in household', 'category_heading'),
        (25, 'crowd:1', '2 or fewer', 'count_percent'),
        (26, 'crowd:2', '3–4', 'count_percent'),
        (27, 'crowd:3', '5 or more', 'count_percent'),
        (28, None, 'Monthly household income, South African rand', 'category_heading'),
        (29, 'household_income:0', '<1,000', 'count_percent'),
        (30, 'household_income:1', '1,000–5,000', 'count_percent'),
        (31, 'household_income:2', '>5,000', 'count_percent'),
        (32, 'household_smoker_bin', 'Any smoker in household', 'count_percent'),
    )
    for row_number, variable, label, style in specifications:
        if variable is None:
            rows.append(_row('heading_' + str(row_number), row_number,
                             [_literal(label, 1), *[_literal('', i) for i in range(2, 5)]], style))
            continue
        fields = {'count_percent': ['numerator', 'denominator', 'percent'],
                  'mean_sd': ['mean', 'sd'], 'median_iqr': ['median', 'q1', 'q3']}[style]
        digits = 0 if variable in ('birthweight_g', 'parity') else 1
        cells = [_literal(label, 1)]
        cells.extend(_value(registry, 'preparation/table1_statistics', {'variable': variable, 'group': group},
                            fields, column, style=style, digits=digits)
                     for column, (group, _, _) in enumerate(groups, 2))
        rows.append(_row(variable, row_number, cells))
    rows.append(_row('spacing_33', 33, [_literal('', i) for i in range(1, 5)], 'spacer'))
    return _table('main_table_1',
        'Table 1. Baseline characteristics of children contributing spirometry or oscillometry, by early-childhood tuberculosis status',
        'main', 'T01', [('characteristic', 'Characteristic'), *[(g, label) for g, label, _ in groups]], rows,
        ['Values are numerator/nonmissing denominator (%), mean (SD), or median [IQR].',
         'No TB before age five includes children with first TB at older ages. Infancy TB is nested within TB before age five.',
         'Inherited child/birth indicator recodes assign missing source codes to the reference category; direct baseline and categorical fields retain nonmissing denominators. Missingness is retained in the aggregate audit.'])


def _table_two(registry):
    rows = [_header(1, ['Spirometry outcome', 'Scale / unit', 'First TB <5 years', 'First TB <1 year'])]
    for stage, label, start in [('pre', 'Pre-bronchodilator', 2), ('post', 'Post-bronchodilator', 9)]:
        rows.append(_section(stage, start, label, 4))
        for offset, (outcome, unit, template, digits) in enumerate(SPIROMETRY, 1):
            target = template.format(stage=stage)
            cells = [_literal(outcome, 1), _literal(unit, 2)]
            cells.extend(_value(registry, 'primary/' + target + '/estimates', {'contrast': contrast},
                                ['mean', 'lower', 'upper'], column, digits=digits)
                         for column, contrast in enumerate(('TB <5 weighted', 'TB <1'), 3))
            rows.append(_row(target, start + offset, cells))
    return _table('main_table_2', 'Table 2. Adjusted differences in pre- and post-bronchodilator spirometry following early-childhood tuberculosis',
        'main', 'T02', [('outcome', 'Outcome'), ('unit', 'Scale / unit'), ('before_five', 'First TB <5 years'), ('infancy', 'First TB <1 year')], rows,
        ['Posterior mean adjusted differences with equal-tailed 95% credible intervals from separate shared-parameter joint models.',
         'TB before age five is child-weighted across infancy and ages one to younger than five; infancy is a nested subgroup.',
         'Separate available-case pre- and post-bronchodilator models do not estimate within-visit bronchodilator response. GLI z-scores use the GLI Global equations.'])


def _table_s1(registry):
    rows = [_header(1, ['Adjustment', 'First TB <5 years', 'First TB <1 year'], [1, 2, 2]),
            _header(2, ['', 'Difference, pp (95% CrI)', 'Attenuation from core', 'Difference, pp (95% CrI)', 'Attenuation from core'])]
    for number, (target, label) in enumerate(SENSITIVITY_MODELS, 3):
        cells = [_literal(label, 1)]
        for column, contrast in [(2, SENSITIVITY_CONTRASTS[0]), (4, SENSITIVITY_CONTRASTS[1])]:
            selector = {'target': target, 'contrast': contrast}
            cells.append(_value(registry, 'sensitivity/table_s1', selector, ['mean', 'lower', 'upper'], column))
            cells.append(_literal('Reference', column + 1) if target == 'early_core' else
                         _value(registry, 'sensitivity/table_s1', selector, ['attenuation_percent'], column + 1, style='percent', digits=1))
        rows.append(_row(target, number, cells))
    return _table('appendix_table_s1', 'Table S1. Sensitivity to confounder adjustment in the early-TB model', 'appendix', 'T01',
        [('model', 'Adjustment'), ('before_five', 'Before five'), ('before_five_attenuation', 'Attenuation'), ('infancy', 'Infancy'), ('infancy_attenuation', 'Attenuation')], rows,
        ['Differences are posterior means in percentage points with equal-tailed 95% credible intervals.',
         'Attenuation is 100 × [1 − |adjusted posterior mean| / |core posterior mean|]. It is a ratio of means, without a posterior interval. An exactly zero core mean makes attenuation undefined; negative attenuation is retained.'], 2)


def _table_s2(registry):
    rows = [_header(1, ['FEF₂₅–₇₅ outcome', 'TB before age 5 years', 'TB during infancy (<1 year)'], [1, 2, 2]),
            _header(2, ['', 'Pre-bronchodilator', 'Post-bronchodilator', 'Pre-bronchodilator', 'Post-bronchodilator'])]
    cells = [_literal('Adjusted mean difference, L/s', 1)]
    for column, (contrast, target) in enumerate((
        ('First TB <5 years, child-weighted', 'mmef_pre_paired'),
        ('First TB <5 years, child-weighted', 'mmef_post_paired'),
        ('First TB <1 year', 'mmef_pre_paired'), ('First TB <1 year', 'mmef_post_paired')), 2):
        cells.append(_value(registry, 'supporting/' + target + '/estimates', {'estimand': contrast},
                            ['mean', 'lower', 'upper'], column))
    rows.append(_row('paired_joint', 3, cells))
    cells = [_literal('Post-bronchodilator minus pre-bronchodilator, L/s', 1)]
    for column, design in [(2, 'before5'), (3, 'infancy')]:
        cells.append(_value(registry, 'supporting_repeated/mmef_change', {'design': design},
                            ['estimate', 'ci_low', 'ci_high'], column, digits=3, colspan=2))
    rows.append(_row('paired_change', 4, cells))
    return _table('appendix_table_s2', 'Table S2. Adjusted associations of early-childhood tuberculosis with pre- and post-bronchodilator FEF₂₅–₇₅',
        'appendix', 'T02', [('outcome', 'Outcome'), ('before_five_pre', 'Before five, pre'), ('before_five_post', 'Before five, post'), ('infancy_pre', 'Infancy, pre'), ('infancy_post', 'Infancy, post')], rows,
        ['The paired pre/post joint-model row reports posterior means and equal-tailed 95% credible intervals.',
         'The post-minus-pre row reports the adjusted paired-change coefficient and cluster-robust 95% confidence interval. This available-case change sample differs from the paired joint-model sample.'], 2)


def _table_s3(registry):
    rows = [_header(1, ['Model / sensitivity analysis', 'Children / tests', 'First TB <1 year', 'First TB <5 years'])]
    specifications = [
        (2, None, 'Core model and baseline covariate adjustment'),
        *[(number, target, label) for number, (target, label) in enumerate(SENSITIVITY_MODELS, 3)],
        (8, None, 'Exploratory LRTI adjustment'),
        (9, 'early_ever_lrti', 'Core + ever-recorded LRTI'),
        (10, 'early_all_four', 'Core + all three baseline covariates + LRTI'),
        (11, None, 'Restricted observation window'),
        (12, 'early_core_window', 'Core early-TB model; restricted follow-up'),
        (13, None, 'Expanded all-recorded-TB model'),
        (14, 'ratio_pre', 'Expanded primary; pre-bronchodilator'),
        (15, 'window_ratio_pre', 'Expanded restricted follow-up; pre-bronchodilator'),
        (16, 'ratio_post', 'Expanded primary; post-bronchodilator'),
        (17, 'window_ratio_post', 'Expanded restricted follow-up; post-bronchodilator'),
    ]
    for number, target, label in specifications:
        if target is None:
            rows.append(_section('section_' + str(number), number, label, 4)); continue
        cells = [_literal(label, 1), _value(registry, 'sensitivity/table_s3',
            {'target': target, 'contrast': 'First TB <1 year'}, ['target_children', 'target_observations'],
            2, style='count_pair', digits=0)]
        for column, contrast in [(3, 'First TB <1 year'), (4, 'First TB <5 years')]:
            cells.append(_value(registry, 'sensitivity/table_s3', {'target': target, 'contrast': contrast},
                                ['mean', 'lower', 'upper'], column))
        rows.append(_row(target, number, cells))
    return _table('appendix_table_s3', 'Table S3. Sensitivity to adjustment and observation-window specification', 'appendix', 'T03',
        [('model', 'Model'), ('sample', 'Children / tests'), ('infancy', 'First TB <1 year'), ('before_five', 'First TB <5 years')], rows,
        ['Posterior mean differences in FEV₁/FVC percentage points with equal-tailed 95% credible intervals. Counts refer to the outcome-specific available sample.',
         'Early-TB models exclude measurements following first TB at age five or older. Expanded models retain those observations. Restricted follow-up changes the observation window and sample composition.',
         'Ever-recorded LRTI is exploratory because episode timing is unavailable; it enters nuisance regressions, not the first-TB hazard.'])


def _table_s4(registry, recurrent):
    process = 'Participant-initiated recurrent wheeze episodes' if recurrent else 'Scheduled current wheeze'
    measure = 'IRR' if recurrent else 'OR'
    rows = [_header(1, ['Attained age', 'Sex adjusted\n' + measure + ' (95% CrI)',
                        'Sex, smoking and birthweight adjusted\n' + measure + ' (95% CrI)'])]
    for number, age in enumerate(('1-<3 years', '3-<5 years', '5-10.5 years'), 2):
        cells = [_literal(age, 1)]
        for column, scenario in [(2, 'primary_sex'), (3, 'secondary_sex_smoking_birthweight')]:
            cells.append(_value(registry, 'sensitivity/table_s4', {'scenario': scenario, 'process': process, 'age_band': age},
                                ['posterior_median', 'lower', 'upper'], column))
        rows.append(_row(age, number, cells))
    return _table('appendix_table_s4_' + ('recurrent' if recurrent else 'scheduled'),
        'Table S4. Wheeze sensitivity analyses: ' + process.lower(), 'appendix', 'T05' if recurrent else 'T04',
        [('age', 'Attained age'), ('primary', 'Sex adjusted'), ('adjusted', 'Sex, smoking and birthweight adjusted')], rows,
        ['Posterior medians and equal-tailed 95% credible intervals for infancy TB versus pre/no-TB time within each attained-age band.',
         'These are conditional ' + ('incidence rate ratios for recurrent episodes.' if recurrent else 'odds ratios for scheduled wheeze; they are distinct from standardized risk ratios.')])


def _table_s5(registry):
    rows = [_header(1, ['Measure', *AGE_HEADERS])]
    measures = [('R', 'Resistance', 'cmH₂O·s/L'), ('C', 'Compliance', 'mL/cmH₂O'),
                ('ReE', 'End-expiratory resistance', 'hPa·s/L'), ('ReI', 'End-inspiratory resistance', 'hPa·s/L'),
                ('XeE', 'End-expiratory reactance', 'hPa·s/L'), ('XeI', 'End-inspiratory reactance', 'hPa·s/L')]
    for number, (target, label, unit) in enumerate(measures, 2):
        cells = [_literal(label + '\n(' + unit + ')', 1)]
        cells.extend(_value(registry, 'mechanics/' + target + '/estimates', {'contrast': contrast},
                            ['reported_mean', 'reported_lower', 'reported_upper'], column)
                     for column, contrast in enumerate(AGE_CONTRASTS, 2))
        rows.append(_row(target, number, cells))
    return _table('appendix_table_s5', 'Table S5. Respiratory-mechanics differences after tuberculosis by age at first diagnosis',
        'appendix', 'T06', [('measure', 'Measure'), *[(str(i), label) for i, label in enumerate(AGE_HEADERS)]], rows,
        ['Posterior mean differences with equal-tailed 95% credible intervals from six separate pre-bronchodilator joint models.',
         'The reported native-unit fields already convert compliance to mL/cmH₂O. Negative compliance estimates mean lower actual compliance.',
         '*Child-weighted pooled contrasts. The first three diagnosis-age groups are mutually exclusive. These exploratory intervals are not adjusted for multiple comparisons.'])


def _table_s6(registry, stage):
    rows = [_header(1, ['Measure', *AGE_HEADERS])]
    for number, (label, unit, template, digits) in enumerate(SPIROMETRY, 2):
        target = template.format(stage=stage);cells = [_literal(label + '\n' + unit, 1)]
        cells.extend(_value(registry, 'primary/' + target + '/estimates', {'contrast': contrast},
                            ['mean', 'lower', 'upper'], column, digits=digits)
                     for column, contrast in enumerate(AGE_CONTRASTS, 2))
        rows.append(_row(target, number, cells))
    return _table('appendix_table_s6_' + stage, 'Table S6. Spirometry differences by age at first tuberculosis diagnosis: ' +
        ('pre-bronchodilator' if stage == 'pre' else 'post-bronchodilator'), 'appendix', 'T07' if stage == 'pre' else 'T08',
        [('measure', 'Measure'), *[(str(i), label) for i, label in enumerate(AGE_HEADERS)]], rows,
        ['Posterior mean adjusted differences with equal-tailed 95% credible intervals from separate expanded joint models.',
         '*Pooled contrasts use outcome-specific child weights. Diagnosis-age subgroup intervals are exploratory and are not adjusted for multiplicity.',
         'Pre- and post-bronchodilator models use separate available samples and do not estimate within-visit bronchodilator response.'])


def build_tables(registry):
    """Return ten physical tables with a result key on every scientific cell."""
    return [_table_one(registry), _table_two(registry), _table_s1(registry), _table_s2(registry),
            _table_s3(registry), _table_s4(registry, False), _table_s4(registry, True),
            _table_s5(registry), _table_s6(registry, 'pre'), _table_s6(registry, 'post')]
