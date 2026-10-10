"""Resolve manuscript cells and prose, preserving explicit coverage and lineage."""
from __future__ import annotations

from collections import Counter
import re
from string import Formatter

from .formatting import ReportingError, cell


COUNTS = {'estimate':186,'count':75,'header':26,'context':17}
TABLE_IDS = ('main_table_1','main_table_2','appendix_table_s1','appendix_table_s2',
             'appendix_table_s3','appendix_table_s4_scheduled','appendix_table_s4_recurrent',
             'appendix_table_s5','appendix_table_s6_pre','appendix_table_s6_post')


def bound_cell(registry, binding):
    if not isinstance(binding,dict) or set(binding)!={'source','where','fields','style','digits'}:
        raise ReportingError('Numerical bindings require exactly source, where, fields, style and digits.')
    record=registry.select(binding['source'],**binding['where'])
    result=cell(record,binding['fields'],style=binding['style'],digits=binding['digits'])
    canonical=registry.get(result['result_key'])
    return {**result,'full_precision_values':[canonical['values'][f] for f in result['fields']],
            'semantics':canonical['semantics']}


def _structure(table):
    return {'id':table['id'],'title':table['title'],'reference_document':table['reference_document'],
            'reference_table':table['reference_table'],'header_rows':table['header_rows'],
            'column_count':len(table['columns']),'row_count':len(table['rows']),
            'rows':[{'id':r['id'],'reference_row':r['reference_row'],'kind':r['kind'],
                     'cells':[{'reference_column':c['reference_column'],'colspan':c.get('colspan',1)}
                              for c in r['cells']]} for r in table['rows']]}


def build_checks(registry, tables, specification):
    """Require every retained numerical location and every scientific cell once."""
    if specification.get('schema_version')!=1 or type(specification.get('schema_version')) is not int:
        raise ReportingError('Manuscript specification requires schema version one.')
    if [t['id'] for t in tables]!=list(TABLE_IDS):
        raise ReportingError('The ten selected physical manuscript tables are required in order.')
    structures=specification['table_structures']
    if len(structures)!=10 or len({x['numbered_table_id'] for x in structures})!=8:
        raise ReportingError('Manuscript table inventory is incomplete.')
    coordinates={}
    for table,structure in zip(tables,structures):
        if _structure(table)!={k:v for k,v in structure.items() if k!='numbered_table_id'}:
            raise ReportingError('Rendered table structure differs from the retained coverage map.')
        for row in table['rows']:
            if sum(c.get('colspan',1) for c in row['cells'])!=len(table['columns']):
                raise ReportingError('A table row has an invalid expanded grid width.')
            for c in row['cells']:
                key=(table['id'],row['reference_row'],c['reference_column'])
                if key in coordinates:
                    raise ReportingError('Physical cell coordinates must be unique.')
                coordinates[key]=c
                if c['result_key']:
                    expected=cell(registry.get(c['result_key']),c['fields'],style=c['style'],digits=c['digits'])
                    if c['text']!=c.get('prefix','')+expected['text']+c.get('suffix',''):
                        raise ReportingError('A table cell differs from its computed result and display rule.')
    entries=specification['numeric_cells']
    if len(entries)!=304 or len({x['id'] for x in entries})!=304 or Counter(x['classification'] for x in entries)!=COUNTS:
        raise ReportingError('All 304 original digit-containing cells must be classified exactly once.')
    checked,seen=[],set()
    for entry in entries:
        coordinate=(entry['physical_table_id'],entry['reference_row'],entry['reference_column'])
        if coordinate not in coordinates or coordinate in seen:
            raise ReportingError('A retained numerical cell is missing or duplicated.')
        seen.add(coordinate);actual=coordinates[coordinate]
        if entry['colspan']!=actual.get('colspan',1):
            raise ReportingError('A retained numerical cell has the wrong span.')
        bound=None
        if entry['classification'] in {'estimate','count'} or (entry['classification']=='context' and 'binding' in entry):
            bound=bound_cell(registry,entry['binding'])
            if any(bound[k]!=actual[k] for k in ('result_key','fields','style','digits')):
                raise ReportingError('Table and coverage map use different scientific results.')
            if entry['classification']=='context' and (actual.get('prefix','')!=entry.get('prefix','')
                    or actual.get('suffix','')!=entry.get('suffix','')):
                raise ReportingError('Configured context label decorations differ from the coverage map.')
        elif actual['result_key'] is not None or actual['text']!=entry['label']:
            raise ReportingError('Context/header digits are mislabeled as scientific results or changed.')
        checked.append({'id':entry['id'],'physical_table_id':entry['physical_table_id'],
                        'row':entry['reference_row'],'column':entry['reference_column'],
                        'classification':entry['classification'],'display':actual['text'],
                        'value':bound,'status':'computed' if bound else 'context_label'})
    bound_coordinates={k for k,c in coordinates.items() if c['result_key']}
    scientific={(e['physical_table_id'],e['reference_row'],e['reference_column']) for e in entries
                if e['classification'] in {'estimate','count'}}
    bound_context={(e['physical_table_id'],e['reference_row'],e['reference_column']) for e in entries
                   if e['classification']=='context' and 'binding' in e}
    if bound_coordinates!=scientific|bound_context or len(scientific)!=261:
        raise ReportingError('Every scientific table cell must be included in the coverage map.')
    sentences=specification['sentences']
    if len(sentences)!=42 or len({s['id'] for s in sentences})!=42:
        raise ReportingError('The 42 retained numerical sentences must be included exactly once.')
    resolved=[]
    for sentence in sentences:
        fields=[]
        for _,name,fmt,conversion in Formatter().parse(sentence['template']):
            if name is not None:
                if not re.fullmatch('[a-z][a-z0-9_]*',name) or fmt or conversion:
                    raise ReportingError('Sentence placeholders must be plain named tokens.')
                fields.append(name)
        if set(fields)!=set(sentence['tokens']):
            raise ReportingError('Sentence placeholders and result bindings disagree.')
        tokens={name:bound_cell(registry,b) for name,b in sentence['tokens'].items()}
        text=sentence['template'].format(**{k:v['text'] for k,v in tokens.items()})
        evidence=[]
        for e in sentence.get('qualitative_evidence',[]):
            bound=bound_cell(registry,e['binding'])
            null=e['null'];compatible=None
            if null is not None:
                values=bound['full_precision_values']
                if len(values)!=3 or any(v is None for v in values):
                    raise ReportingError('A null-compatibility statement requires a complete selected interval.')
                compatible=values[1]<=null<=values[2]
            evidence.append({'value':bound,'null':null,'interval_includes_null':compatible,'review':e['review']})
        unsupported=any(e['interval_includes_null'] is False for e in evidence)
        missing=any(any(v is None for v in t['full_precision_values']) for t in tokens.values())
        resolved.append({'id':sentence['id'],'section':sentence['section'],'classification':sentence['classification'],
                         'text':text,'tokens':tokens,'qualitative_evidence':evidence,
                         'context_numbers':sentence['context_numbers'],'references':sentence['references'],
                         'notes':sentence.get('notes',''),
                         'status':'scientific_wording_review_required' if unsupported else
                                  'contains_explicit_unavailable_facts' if missing else 'resolved'})
    return {'schema_version':1,'coverage':{'numbered_tables':8,'physical_tables':10,'numeric_cells':304,
            'scientific_cells':261,'bound_context_cells':len(bound_context),'cell_classifications':dict(COUNTS),'sentences':42,
            'value_tokens':sum(len(s['tokens']) for s in resolved),
            'qualitative_bindings':sum(len(s['qualitative_evidence']) for s in resolved)},
            'cells':checked,'sentences':resolved,
            'wording_review_required':any(s['status']=='scientific_wording_review_required' for s in resolved)}


def _numbers(text):
    text=text.replace('−','-')
    text=re.sub(r'(?<=[0-9])[·⋅](?=[0-9])','.',text)
    # ASCII numerals only: subscript labels such as FEV₁ are not estimates.
    return [float(x.replace(',','')) for x in re.findall(r'[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?',text)]


def compare_references(checks, reference_cells=None, reference_sentences=None):
    """Informational display comparison; disagreement never overwrites estimates."""
    if reference_cells is None and reference_sentences is None:
        return {'schema_version':1,'status':'reference_not_supplied','cells':[],'sentences':[]}
    if not isinstance(reference_cells,list) or not isinstance(reference_sentences,list):
        raise ReportingError('Reference cells and sentences must be supplied together as JSON arrays.')
    def index(rows,expected):
        if len(rows)!=len(expected) or len({r['id'] for r in rows})!=len(rows) or {r['id'] for r in rows}!={r['id'] for r in expected}:
            raise ReportingError('Reference inventory does not match the declared manuscript locators.')
        return {r['id']:r for r in rows}
    ci=index(reference_cells,checks['cells']);si=index(reference_sentences,checks['sentences'])
    cells=[]
    for c in checks['cells']:
        before=ci[c['id']]['displayed_text'];after=c['display']
        # Header N values are a scientific count embedded in a context label.
        if c['id'].startswith('main_T01_R01C'):
            old=re.search(r'N\s*=\s*([0-9,]+)',before)
            new=re.search(r'N\s*=\s*([0-9,]+)',after)
            same=bool(old and new and _numbers(old.group(1))==_numbers(new.group(1)))
        else:
            same=_numbers(before)==_numbers(after)
        cells.append({'id':c['id'],'classification':c['classification'],'reference_display':before,
                      'computed_display':after,'displayed_numbers_match':same,
                      'status':'matches' if same else 'display_change_for_review'})
    sentences=[]
    for s in checks['sentences']:
        before=si[s['id']]['text'];after=s['text']
        sentences.append({'id':s['id'],'reference_text':before,'computed_text':after,
                          'numeric_sequences_match':_numbers(before)==_numbers(after),
                          'status':'explicit_unavailable' if s['status']=='contains_explicit_unavailable_facts' else
                                   'wording_and_value_comparison_for_review'})
    return {'schema_version':1,'status':'compared','comparison_policy':
            'Informational display comparison only. Sentence wording/context can change numeric order. Scientific correctness is checked against keyed results; retained displays do not replace computed values.',
            'cells':cells,'sentences':sentences,
            'table_numeric_matches':sum(c['displayed_numbers_match'] for c in cells),
            'table_numeric_changes':sum(not c['displayed_numbers_match'] for c in cells),
            'sentence_numeric_sequence_matches':sum(s['numeric_sequences_match'] for s in sentences)}
