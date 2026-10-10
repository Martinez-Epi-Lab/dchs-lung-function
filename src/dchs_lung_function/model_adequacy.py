"""Validate predictive adequacy separately from sampler convergence."""
import math
from .covariance_ppc import evaluate_target_covariance, protocol, _route
from .execution import canonical_bytes
from .pipeline import ValidationError, read_json


def evaluate_covariance(posterior, contract, *, family, target):
    try:
        value=evaluate_target_covariance(posterior,contract,family=family,target=target)
        accepted=value['accepted']
        fields={'protocol','family','target','model_specification_version','seed','status','accepted',
                'joint_pass','reasons','counts','draw_selection','observed','predictive'}
        spec,canonical=_route(family,target)
        count_keys={'joint_children','target_observations','selected_observations','selected_children','repeat_children','within_df'}
        reasons={'insufficient_repeat_children','insufficient_within_df','undefined_observed_covariance',
                 'insufficient_saved_draws','icc_outside_predictive_bounds','within_ms_outside_predictive_bounds'}
        if (set(value) not in (fields,fields|{'population_formula_icc'})
                or value['family']!=canonical or value['model_specification_version']!=spec['specification_version']
                or type(value['seed']) is not int or value['seed']<0
                or not isinstance(value['counts'],dict) or set(value['counts'])!=count_keys
                or any(type(v) is not int or v<0 for v in value['counts'].values())
                or not isinstance(value['reasons'],list) or any(r not in reasons for r in value['reasons'])
                or set(value['observed'])-{'icc','within_ms','between_ms','effective_repeat_count','total_variance'}
                or set(value['predictive'])-{'icc','within_ms','between_ms','total_variance'}
                or set(value['draw_selection'])-{'chains','draws_per_chain','available_draws','requested_replicates','replicates','positions_sha256'}):
            raise ValueError('Covariance evidence has an undeclared field or model identity.')
        if (type(accepted) is not bool or type(value['joint_pass']) is not bool
                or value['joint_pass']!=accepted or value['status'] not in {'passed','failed','not_evaluable'}
                or accepted!=(value['status']=='passed') or value['target']!=target
                or canonical_bytes(value['protocol'])!=canonical_bytes(protocol())):
            raise ValueError('Inconsistent covariance adequacy evidence.')
        def numeric(item):
            return type(item) in (int,float) and math.isfinite(item)
        if not all(numeric(item) for item in value['observed'].values()):
            raise ValueError('Observed evidence must contain finite aggregate numbers.')
        for key,item in value['draw_selection'].items():
            if key=='positions_sha256':
                if not isinstance(item,str) or len(item)!=64 or any(c not in '0123456789abcdef' for c in item):
                    raise ValueError('Invalid draw selection digest.')
            elif type(item) is not int or item<0:raise ValueError('Invalid aggregate draw count.')
        for metric,item in value['predictive'].items():
            if set(item)!={'mean','median','lower','upper','tail_at_or_below_observed','tail_at_or_above_observed','used_for_acceptance','passed'}:
                raise ValueError('Predictive evidence has undeclared fields.')
            values=[item[k] for k in ('mean','median','lower','upper','tail_at_or_below_observed','tail_at_or_above_observed')]
            if (not all(numeric(v) for v in values) or item['lower']>item['upper']
                    or not 0<=item['tail_at_or_below_observed']<=1 or not 0<=item['tail_at_or_above_observed']<=1
                    or item['used_for_acceptance'] is not (metric in ('icc','within_ms'))
                    or (type(item['passed']) is not bool if metric in ('icc','within_ms') else item['passed'] is not None)):
                raise ValueError('Predictive evidence has invalid aggregate values.')
        if 'population_formula_icc' in value and set(value['population_formula_icc'])!={'mean','q025','q975','used_for_acceptance'}:
            raise ValueError('Population covariance evidence has undeclared fields.')
        if 'population_formula_icc' in value:
            item=value['population_formula_icc']
            if item['used_for_acceptance'] is not False or not all(numeric(item[k]) and 0<=item[k]<=1 for k in ('mean','q025','q975')):
                raise ValueError('Invalid population covariance aggregate values.')
        canonical_bytes(value)
        return value
    except (ValueError,TypeError,KeyError,AttributeError):
        raise ValidationError('Covariance adequacy evaluation returned invalid evidence.') from None


def adequacy_status(diagnostics_passed, covariance_passed):
    if type(diagnostics_passed) is not bool or type(covariance_passed) is not bool:
        raise ValidationError('Convergence and covariance adequacy gates must be Boolean.')
    return ('diagnostics_failed' if not diagnostics_passed else
            'diagnostics_passed' if covariance_passed else 'model_adequacy_failed')


def verify_covariance(directory, summary, diagnostics, covariance):
    """Compare saved checks and status with results recomputed from the draws."""
    try:
        if (canonical_bytes(read_json(directory/'covariance_ppc.json','Covariance adequacy evidence'))!=canonical_bytes(covariance)
                or type(summary.get('covariance_ppc_passed')) is not bool
                or summary['covariance_ppc_passed']!=covariance['accepted']
                or summary.get('diagnostics_passed') is not diagnostics['passed']
                or summary.get('status')!=adequacy_status(diagnostics['passed'],covariance['accepted'])):
            raise ValueError('Covariance evidence mismatch.')
    except (ValueError,KeyError,TypeError,AttributeError):
        raise ValidationError('Saved covariance adequacy evidence or completion differs from verified posterior draws.') from None
