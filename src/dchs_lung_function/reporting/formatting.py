"""Central display rules; never change the full-precision result being displayed."""
from __future__ import annotations

import math
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP


class ReportingError(ValueError):
    """A manuscript selector, result or display contract is invalid."""


def number(value, digits=2):
    """Decimal half-up rounding, fixed precision and no negative displayed zero."""
    if type(digits) is not int or not 0 <= digits <= 8:
        raise ReportingError('Display precision must be an integer from zero to eight.')
    if value is None:
        return 'Not available'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReportingError('A numerical display requires a finite numeric value or explicit null.')
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f'{rounded:.{digits}f}'


def literal(text):
    if not isinstance(text, str):
        raise ReportingError('Literal manuscript labels must be strings.')
    return {'text':text, 'result_key':None, 'fields':[], 'style':'literal', 'digits':None}


def cell(record, fields, style='interval', digits=2):
    """Format named fields of one keyed aggregate result and retain their lineage."""
    fields = [fields] if isinstance(fields, str) else list(fields)
    if not fields or any(f not in record['values'] for f in fields):
        raise ReportingError('A display field is absent from its aggregate result.')
    values = [record['values'][f] for f in fields]
    expected = {'interval':3, 'number':1, 'integer':1, 'percent':1,
                'pvalue':1, 'count_percent':3, 'mean_sd':2, 'median_iqr':3, 'count_pair':2,
                'mean_sd_n':3, 'median_iqr_n':4, 'date':1, 'attenuation':4}
    if style not in expected or len(values) != expected[style]:
        raise ReportingError('Display style or number of fields is invalid.')
    # Null is a state, never an invented zero, a suppressed cell or an empty string.
    if style == 'date':
        try:
            parsed = date.fromisoformat(values[0])
        except (ValueError, TypeError) as exc:
            raise ReportingError('A reporting date must be a valid ISO calendar date.') from exc
        if values[0] != parsed.isoformat():
            raise ReportingError('A reporting date must use canonical ISO format.')
        months = ('January','February','March','April','May','June','July','August','September','October','November','December')
        text = f'{parsed.day} {months[parsed.month-1]} {parsed.year}'
    elif style == 'attenuation':
        attenuation, bound, status, passed = values
        if type(passed) is not bool or not passed:
            raise ReportingError('Attenuation cannot be displayed without passing numerical precision.')
        if status == 'reference':
            if attenuation is not None or bound is not None:
                raise ReportingError('A reference row has no attenuation estimate or MCSE bound.')
            text = 'Reference'
        else:
            if status not in {'estimated', 'sign_reversal'}:
                raise ReportingError('Undefined attenuation cannot be published.')
            if not isinstance(bound,(int,float)) or isinstance(bound,bool) or not math.isfinite(bound) or not 0 <= bound <= 1:
                raise ReportingError('Attenuation MCSE bound must be finite and at most one percentage point.')
            upper_display = Decimal(str(bound)).quantize(Decimal('0.1'),rounding=ROUND_CEILING)
            text = f'≈{number(attenuation,0)}%\nMCSE ≤ {upper_display:.1f} pp'
            if status == 'sign_reversal':
                text += '\nSign reversal'
    elif style in {'mean_sd_n','median_iqr_n'}:
        n = values[-1]
        if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n != int(n) or n < 0:
            raise ReportingError('Available N must be a nonnegative integer.')
        base = cell(record,fields[:-1],style=style[:-2],digits=digits)['text']
        text = f'{base}; n={int(n)}'
    elif any(v is None for v in values):
        status = record.get('status', 'unavailable')
        text = 'Not estimable' if 'estimab' in status else 'Not available'
        if style == 'mean_sd' and values[0] is not None:
            text = number(values[0], digits)+' (SD not available)'
    elif style in {'integer','count_pair'}:
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (float,int)) or not math.isfinite(v) or v != int(v) or v < 0:
                raise ReportingError('Counts must be nonnegative integers.')
        text = str(int(values[0])) if style == 'integer' else ' / '.join(f'{int(v):,}' for v in values)
    elif style == 'pvalue':
        v = values[0]
        if not isinstance(v, (int,float)) or isinstance(v,bool) or not 0 <= v <= 1:
            raise ReportingError('P values must be between zero and one.')
        if type(digits) is not int or digits < 1:
            raise ReportingError('P-value precision must be at least one decimal place.')
        threshold = 10 ** -digits
        text = '<'+number(threshold,digits) if v < threshold else number(v,digits)
    elif style == 'count_percent':
        n, denominator, percent = values
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                or v != int(v) or v < 0 for v in (n,denominator))
                or denominator == 0 or n > denominator
                or not math.isclose(percent,100*n/denominator, rel_tol=1e-12, abs_tol=1e-12)):
            raise ReportingError('Counts and their nonmissing denominator disagree with the percentage.')
        text = f'{int(n)}/{int(denominator)} ({number(percent,digits)})'
    else:
        v = [number(x,digits) for x in values]
        if style == 'interval':
            if not values[1] <= values[2]:
                raise ReportingError('Interval bounds are reversed.')
            text = f'{v[0]} ({v[1]} to {v[2]})'
        elif style == 'mean_sd':
            if values[1] < 0:
                raise ReportingError('A standard deviation cannot be negative.')
            text = f'{v[0]} ({v[1]})'
        elif style == 'median_iqr':
            if not values[1] <= values[0] <= values[2]:
                raise ReportingError('Median and quartiles are out of order.')
            text = f'{v[0]} [{v[1]} to {v[2]}]'
        else:
            text = v[0]+('%' if style == 'percent' else '')
    result_key = record['key']
    aliases = record.get('aliases',{})
    if all(f in aliases for f in fields):
        targets = {aliases[f]['result_key'] for f in fields}
        if len(targets)!=1:
            raise ReportingError('One displayed cell cannot combine conflicting canonical results.')
        result_key = targets.pop()
        fields = [aliases[f]['field'] for f in fields]
    return {'text':text, 'result_key':result_key, 'fields':fields,
            'style':style, 'digits':digits, 'status':record.get('status','available')}
