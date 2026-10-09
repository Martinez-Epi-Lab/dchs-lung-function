"""Independent fabricated aggregate reporting and editable XML boundary tests."""
from copy import deepcopy
import hashlib
from io import BytesIO
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
from uuid import UUID
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dchs_lung_function.reporting import formatting as fmt,results,facts,tables,sources,render_document as doc
from dchs_lung_function.reporting import numerical_text as nt
from dchs_lung_function.pipeline import ValidationError
SCHEMA=json.loads((ROOT/'config/reporting_schemas.json').read_text())

def frame(source,rows):
    columns=SCHEMA['schemas'][SCHEMA['sources'][source]]
    return pd.DataFrame([{**dict.fromkeys(columns),**r} for r in rows],columns=columns)
def record(**values):return {'key':'fabricated#one','status':'available','values':values}
def estimate(**extra):return {'target':'ratio_pre','contrast':'TB <1','unit':'percentage points','mean':-.015,'median':-.016,'lower':-.125,'upper':.095,**extra}
def provenance():return {'fit_fingerprint':'a'*64,'fingerprint':'b'*64,'receipt_sha256':'c'*64}
def alias(**extra):return {'target':'ratio_pre','contrast':'First TB <1 year','unit':'percentage points','mean':-.015,'lower':-.125,'upper':.095,'fit_fingerprint':'a'*64,'validation_fingerprint':'b'*64,'interval_type':'equal-tailed 95% credible interval',**extra}
def fixture_table():
    return {'id':'main_table_2','title':'Fabricated <title> & estimate','columns':[{'id':str(i)} for i in range(4)],
      'rows':[{'kind':'header','cells':[{'text':'Characteristic','colspan':2},{'text':'Value'},{'text':'Interval'}]},
              {'kind':'section','cells':[{'text':'Fabricated section','colspan':4}]},
              {'kind':'data','cells':[{'text':'Made-up measurement'},{'text':'1'},{'text':'0.00'},{'text':'Not estimable'}]},
              {'kind':'spacer','cells':[]}], 'footnotes':['Synthetic footnote\nSecond line']}
def fake_font(style='Regular',family='Arial',flags=0):
    strings=[(1,family),(2,style)];encoded=[s.encode('utf-16-be') for _,s in strings]
    name=struct.pack('>HHH',0,len(strings),6+12*len(strings));offset=0
    for (identifier,_),text in zip(strings,encoded):name+=struct.pack('>HHHHHH',3,1,0x409,identifier,len(text),offset);offset+=len(text)
    name+=b''.join(encoded);os2=bytearray(12);struct.pack_into('>H',os2,8,flags)
    offset=44;header=b'\0\1\0\0'+struct.pack('>HHHH',2,0,0,0)
    header+=struct.pack('>4sIII',b'OS/2',0,offset,len(os2));offset+=len(os2)
    header+=struct.pack('>4sIII',b'name',0,offset,len(name))
    return header+os2+name


class ReportingFormattingTests(unittest.TestCase):
    def test_half_up_and_negative_displayed_zero(self):
        for value,digits,expected in [(1.005,2,'1.01'),(-1.005,2,'-1.01'),(-.0049,2,'0.00'),(-.005,2,'-0.01'),(2.5,0,'3'),(-2.5,0,'-3'),(0,3,'0.000')]:
            with self.subTest(value=value):self.assertEqual(fmt.number(value,digits),expected)
    def test_null_is_explicit_and_invalid_numbers_are_rejected(self):
        self.assertEqual(fmt.number(None),'Not available')
        for value in (True,'1',float('nan'),float('inf'),complex(1),[1]):
            with self.subTest(value=value),self.assertRaises(fmt.ReportingError):fmt.number(value)
        for digits in (True,-1,9,1.5,'2'):
            with self.subTest(digits=digits),self.assertRaises(fmt.ReportingError):fmt.number(1,digits)
    def test_small_and_zero_counts_are_exact_without_suppression(self):
        for n in range(10):self.assertEqual(fmt.cell(record(n=n),'n','integer')['text'],str(n))
        self.assertEqual(fmt.cell(record(n=1,total=9,p=100/9),['n','total','p'],'count_percent',1)['text'],'1/9 (11.1)')
        self.assertEqual(fmt.cell(record(n=0,total=9,p=0),['n','total','p'],'count_percent',1)['text'],'0/9 (0.0)')
        self.assertEqual(fmt.cell(record(n=1,total=1001),['n','total'],'count_pair')['text'],'1 / 1,001')
    def test_null_and_nonestimable_states_are_not_zero(self):
        r=record(mean=None,lower=None,upper=None);r['status']='non_estimable'
        self.assertEqual(fmt.cell(r,['mean','lower','upper'])['text'],'Not estimable')
        r['status']='not_reconstructable_from_supplied_inputs';self.assertEqual(fmt.cell(r,['mean','lower','upper'])['text'],'Not available')
        self.assertEqual(fmt.cell(record(mean=3,sd=None),['mean','sd'],'mean_sd',1)['text'],'3.0 (SD not available)')
        self.assertEqual(fmt.cell(record(n=0,d=0,p=None),['n','d','p'],'count_percent')['text'],'Not available')
    def test_styles_validate_counts_percentages_and_interval_order(self):
        cases=[(record(n=-1),'n','integer'),(record(n=1.5),'n','integer'),(record(n=True),'n','integer'),
               (record(n=2,d=1,p=200),['n','d','p'],'count_percent'),(record(n=1,d=2,p=49),['n','d','p'],'count_percent'),
               (record(mean=1,low=3,high=2),['mean','low','high'],'interval'),
               (record(mean=3,q1=4,q3=5),['mean','q1','q3'],'median_iqr'),(record(mean=1,sd=-.1),['mean','sd'],'mean_sd')]
        for r,fields,style in cases:
            with self.subTest(style=style),self.assertRaises(fmt.ReportingError):fmt.cell(r,fields,style)
    def test_pvalues_preserve_zero_and_boundary_threshold_rule(self):
        for p,text in [(0,'<0.001'),(.000999,'<0.001'),(.001,'0.001'),(1,'1.000')]:self.assertEqual(fmt.cell(record(p=p),'p','pvalue',3)['text'],text)
        for p in (-.1,1.1,True,float('nan')):
            with self.assertRaises(fmt.ReportingError):fmt.cell(record(p=p),'p','pvalue',3)
    def test_cell_tracks_fields_without_mutating_full_precision(self):
        r=record(mean=-.015,lower=-.125,upper=.095);old=deepcopy(r)
        cell=fmt.cell(r,['mean','lower','upper']);self.assertEqual(cell['text'],'-0.02 (-0.13 to 0.10)');self.assertEqual(r,old)
        self.assertEqual(cell['fields'],['mean','lower','upper']);self.assertEqual(cell['result_key'],r['key'])
        for args in [([], 'number'),(['mean'],'unknown'),(['missing'],'number')]:
            with self.assertRaises(fmt.ReportingError):fmt.cell(r,*args)
    def test_alias_cells_use_one_canonical_result(self):
        r=record(mean=1,low=0,high=2);r['aliases']={k:{'result_key':'canonical','field':k+'2'} for k in r['values']}
        c=fmt.cell(r,['mean','low','high']);self.assertEqual(c['result_key'],'canonical');self.assertEqual(c['fields'],['mean2','low2','high2'])
        r['aliases']['high']['result_key']='different'
        with self.assertRaises(fmt.ReportingError):fmt.cell(r,['mean','low','high'])


class ReportingRegistryTests(unittest.TestCase):
    def test_exact_selectors_and_key_escaping_are_stable(self):
        r=results.ResultRegistry();source='preparation/flow';r.add(source,frame(source,[{'node':'a&b/"c','count':1,'status':'available'},{'node':'second','count':1,'status':'available'}]))
        first=r.select(source,node='a&b/"c');self.assertNotIn('a&b/',first['key']);self.assertEqual(r.get(first['key']),first)
        for where in ({},{'count':1},{'unknown':1},{'node':'missing'},{'count':True}):
            with self.subTest(where=where),self.assertRaises(fmt.ReportingError):r.select(source,**where)
    def test_registry_returns_defensive_copies_and_sorted_export(self):
        source='preparation/flow';data=frame(source,[{'node':'z','count':2},{'node':'a','count':1}]);p={'hash':'a'*64};r=results.ResultRegistry();r.add(source,data,p)
        data.loc[0,'count']=999;p['hash']='changed';selected=r.select(source,node='z');selected['values']['count']=500
        self.assertEqual(r.select(source,node='z')['values']['count'],2);self.assertEqual(r.select(source,node='z')['provenance']['hash'],'a'*64)
        self.assertEqual([x['values']['node'] for x in r.export()['records']],['a','z'])
    def test_full_schema_and_source_whitelists_exclude_records(self):
        source='primary/ratio_pre/estimates';df=frame(source,[estimate()])
        for name in ('primary/unknown/estimates','primary/ratio_pre/participant_records','unknown/estimates'):
            with self.assertRaises(fmt.ReportingError):results.ResultRegistry().add(name,df)
        for mode in ('private','unknown','missing','duplicate','reordered'):
            changed=df.copy()
            if mode=='private':changed['pid_child']='FABRICATED_ID'
            if mode=='unknown':changed['private_notes']='FABRICATED_ID'
            if mode=='missing':changed=changed.drop(columns='mean')
            if mode=='duplicate':changed=pd.concat([changed,changed[['mean']]],axis=1)
            if mode=='reordered':changed=changed[changed.columns[::-1]]
            with self.subTest(mode=mode),self.assertRaises(fmt.ReportingError):results.ResultRegistry().add(source,changed)
    def test_failed_registration_is_atomic_and_can_be_retried(self):
        source='preparation/flow';r=results.ResultRegistry();good=frame(source,[{'node':'a','count':1}]);bad=pd.concat([good,good],ignore_index=True)
        with self.assertRaises(fmt.ReportingError):r.add(source,bad)
        self.assertEqual(r.export()['record_count'],0);r.add(source,good);self.assertEqual(r.export()['record_count'],1)
        with self.assertRaises(fmt.ReportingError):r.add(source,good)
    def test_null_nan_infinite_nested_and_missing_key_handling(self):
        source='preparation/flow';r=results.ResultRegistry();r.add(source,frame(source,[{'node':'a','count':np.nan}]))
        self.assertIsNone(r.select(source,node='a')['values']['count']);json.dumps(r.export(),allow_nan=False)
        for changed in ({'node':None,'count':1},{'node':'a','count':float('inf')},{'node':'a','count':[1]}):
            with self.assertRaises(fmt.ReportingError):results.ResultRegistry().add(source,frame(source,[changed]))
    def test_only_declared_optional_sources_may_be_empty(self):
        for source in ['sensitivity_model/window_ratio_pre/hazard','wheeze/secondary_sex_smoking_birthweight/standardized_rr']:
            r=results.ResultRegistry();r.add(source,frame(source,[]));self.assertEqual(r.export()['record_count'],0)
            with self.assertRaises(fmt.ReportingError):r.select(source,target='anything')
        with self.assertRaises(fmt.ReportingError):results.ResultRegistry().add('preparation/flow',frame('preparation/flow',[]))
    def test_semantics_distinguish_credible_wald_predictive_and_iqr(self):
        cases=[('primary/ratio_pre/estimates',estimate(),'equal-tailed credible interval'),
               ('feno/estimates',{'target':'main_12months','lower':1,'upper':2},'robust normal Wald confidence interval'),
               ('facts/early_tb_age',{'population':'descriptive','diagnosis_group':'tb_before_five','median':2,'q1':1,'q3':3},'descriptive interquartile range'),
               ('wheeze/primary_sex/posterior_predictive_cells',{'process':'p','age_band':'a','exposure_group':'g'},'posterior predictive interval')]
        for source,row,kind in cases:
            r=results.ResultRegistry();r.add(source,frame(source,[row]));recorded=r.export()['records'][0]
            self.assertIn(kind,[x['type'] for x in recorded['semantics']['intervals']])
        r=results.ResultRegistry();r.add('pretb/estimates',frame('pretb/estimates',[{'target':'resistance','lower':1,'upper':2}]))
        self.assertTrue(any('G08' in s for s in r.export()['records'][0]['semantics']['limitations']))
    def _aliases(self,changes=None):
        r=results.ResultRegistry();r.add('primary/ratio_pre/estimates',frame('primary/ratio_pre/estimates',[estimate()]),provenance())
        for source in ('sensitivity/table_s1','sensitivity/table_s3'):r.add(source,frame(source,[alias(**(changes or {}))]))
        return r
    def test_s1_s3_reuse_the_same_primary_key_and_field(self):
        r=self._aliases();canonical=r.select('primary/ratio_pre/estimates',contrast='TB <1')['key']
        for source in ('sensitivity/table_s1','sensitivity/table_s3'):
            c=fmt.cell(r.select(source,target='ratio_pre',contrast='First TB <1 year'),['mean','lower','upper'])
            self.assertEqual(c['result_key'],canonical);self.assertEqual(c['fields'],['mean','lower','upper'])
    def test_repeated_estimate_numeric_unit_identity_or_interval_conflict_rejected(self):
        for changes in ({'mean':0},{'unit':'L'},{'fit_fingerprint':'d'*64},{'validation_fingerprint':'d'*64},{'interval_type':'95% confidence interval'}):
            r=self._aliases(changes)
            with self.subTest(changes=changes),self.assertRaises(fmt.ReportingError):r.export()
    def test_unknown_sensitivity_contrast_cannot_alias_a_pooled_result(self):
        r=results.ResultRegistry();r.add('primary/ratio_pre/estimates',frame('primary/ratio_pre/estimates',[estimate(contrast='TB <5 weighted')]),provenance())
        r.add('sensitivity/table_s3',frame('sensitivity/table_s3',[alias(contrast='unrecognized age group')]))
        with self.assertRaises(fmt.ReportingError):r.export()
    def test_s4_alias_preserves_conditional_median_not_mean_or_standardized_rr(self):
        r=results.ResultRegistry();source='wheeze/primary_sex/effects';base={'scenario':'primary_sex','process':'Scheduled current wheeze','age_band':'1-<3 years','measure':'conditional odds ratio','posterior_median':1.25,'posterior_mean':1.3,'cri_2_5':.7,'cri_97_5':2.8}
        r.add(source,frame(source,[base]),provenance());row={**base,'lower':.7,'upper':2.8,'interval_type':'equal-tailed 95% credible interval','fit_fingerprint':'a'*64,'validation_fingerprint':'b'*64}
        r.add('sensitivity/table_s4',frame('sensitivity/table_s4',[row]));c=fmt.cell(r.select('sensitivity/table_s4',scenario='primary_sex',process=base['process'],age_band=base['age_band']),['posterior_median','lower','upper'])
        self.assertEqual(c['fields'],['posterior_median','cri_2_5','cri_97_5']);self.assertTrue(c['result_key'].startswith(source+'#'))


class ReportingFactsTests(unittest.TestCase):
    def derive(self,ages):
        groups=['no_recorded_tb' if np.isnan(x) else 'infancy' if x<1 else 'age_one_to_under_five' if x<5 else 'age_five_or_later' for x in ages]
        cohort=pd.DataFrame({'pid_child':['FABRICATED_'+str(i) for i in range(len(ages))],'tb_age_first_years':ages,'tb_diagnosis_group':groups})
        flow=pd.DataFrame({'node':['antenatal_loss_to_followup','pregnancy_loss'],'count':[None,None],'status':['unavailable']*2})
        with mock.patch.object(facts,'load_artifacts',side_effect=[cohort,flow]):return facts.derive_facts(Path('/fabricated/preparation'))
    def test_small_cohort_strict_age_boundaries_linear_quartiles(self):
        values=self.derive([np.nan,0,.5,1,4.9,5]);age=values['early_tb_age'].iloc[0]
        self.assertEqual(age.n_children,4);self.assertAlmostEqual(age['median'],.75);self.assertAlmostEqual(age.q1,.375);self.assertAlmostEqual(age.q3,1.975)
        rows=values['early_tb_distribution'].set_index('fact');self.assertEqual(rows.loc['infancy_among_early_tb','numerator'],2)
        self.assertEqual(rows.loc['tb_before_five_among_cohort','denominator'],6)
        for df in values.values():self.assertNotIn('pid_child',df);self.assertNotIn('FABRICATED_',df.to_json())
    def test_empty_and_single_child_groups_are_explicit(self):
        result=self.derive([np.nan,5]);age=result['early_tb_age'].iloc[0];self.assertEqual(age.status,'empty_group');self.assertTrue(np.isnan(age['median']))
        dist=result['early_tb_distribution'].iloc[1];self.assertEqual(dist.denominator,0);self.assertTrue(np.isnan(dist.percent))
        one=self.derive([.25])['early_tb_age'].iloc[0];self.assertEqual([one.n_children,one['median'],one.q1,one.q3],[1,.25,.25,.25])
        self.assertIsNone(result['unavailable_flow'].iloc[0]['count'])
    def test_duplicate_blank_ids_or_inconsistent_age_groups_rejected(self):
        good=pd.DataFrame({'pid_child':['x','y'],'tb_age_first_years':[.5,np.nan],'tb_diagnosis_group':['infancy','no_recorded_tb']})
        for mode in ('duplicate','blank','missing','wrong_group','age_missing','age_string','negative'):
            x=good.copy()
            if mode=='duplicate':x.loc[1,'pid_child']='x'
            elif mode=='blank':x.loc[1,'pid_child']=' '
            elif mode=='missing':x.loc[1,'pid_child']=None
            elif mode=='wrong_group':x.loc[0,'tb_diagnosis_group']='age_five_or_later'
            elif mode=='age_missing':x.loc[0,'tb_age_first_years']=np.nan
            elif mode=='age_string':x['tb_age_first_years']=x['tb_age_first_years'].astype(object);x.loc[0,'tb_age_first_years']='bad'
            else:x.loc[0,'tb_age_first_years']=-1
            with self.subTest(mode=mode),self.assertRaises(ValueError):facts._validate_descriptive(x)
    def test_incomplete_flow_inventory_cannot_be_invented(self):
        c=pd.DataFrame({'pid_child':['x'],'tb_age_first_years':[.5],'tb_diagnosis_group':['infancy']})
        for f in [pd.DataFrame({'node':['pregnancy_loss'],'count':[None],'status':['unavailable']}),pd.DataFrame({'node':['pregnancy_loss']*2,'count':[None]*2,'status':['unavailable']*2})]:
            with mock.patch.object(facts,'load_artifacts',side_effect=[c,f]),self.assertRaises(ValueError):facts.derive_facts(Path('/fabricated'))


class ReportingSourceReceiptTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.base=Path(t.name).resolve();self.directory=self.base/'result';self.directory.mkdir()
        self.identity={'schema_version':1,'fabricated':True};self.expected=['summary.json','estimates.csv']
        self.summary={'accepted':True,'status':'accepted'};self.save(self.directory/'summary.json',self.summary);(self.directory/'estimates.csv').write_text('term,estimate\nfabricated,1\n')
        self.receipt={'schema_version':1,'identity':self.identity,'fingerprint':sources.digest(self.identity),'files':{n:sources.file_hash(self.directory/n) for n in self.expected},'status':'accepted','completed_at':'fabricated','summary':self.summary}
        self.write()
    def save(self,p,x):p.write_text(json.dumps(x))
    def write(self):self.save(self.directory/'receipt.json',self.receipt)
    def verify(self):return sources.verify_receipt(self.directory,self.identity,self.expected)
    def test_valid_receipt_and_exact_inventory(self):self.assertEqual(self.verify(),self.receipt)
    def test_missing_extra_corrupt_and_traversal_artifacts_rejected(self):
        p=self.directory/'estimates.csv';original=p.read_bytes()
        for mode in ('missing','extra','corrupt','traversal'):
            before=deepcopy(self.receipt)
            if mode=='missing':p.unlink()
            elif mode=='extra':(self.directory/'private.txt').write_text('FABRICATED_PRIVATE')
            elif mode=='corrupt':p.write_bytes(original+b'changed')
            else:self.receipt['files']['../outside']='0'*64;self.write()
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.verify()
            p.write_bytes(original);(self.directory/'private.txt').unlink(missing_ok=True);self.receipt=before;self.write()
    def test_receipt_extra_identity_boolean_and_failure_status_rejected(self):
        old=deepcopy(self.receipt)
        for mode in ('extra','identity','bool','status','list'):
            self.receipt=deepcopy(old)
            if mode=='extra':self.receipt['participant_records']=['FABRICATED_PRIVATE']
            elif mode=='identity':self.receipt['identity']['fabricated']=False
            elif mode=='bool':self.receipt['schema_version']=True
            elif mode=='status':self.receipt['status']='failed'
            else:self.receipt=[]
            self.write()
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.verify()
        self.receipt=old;self.write()
    def test_repaired_hash_cannot_hide_contradictory_completion(self):
        for summary in ({'accepted':False,'status':'accepted'},{'accepted':True,'diagnostics_passed':False,'status':'accepted'},{'accepted':1,'status':'accepted'},{'accepted':True,'status':'failed'}):
            self.save(self.directory/'summary.json',summary);self.receipt['summary']=summary;self.receipt['files']['summary.json']=sources.file_hash(self.directory/'summary.json');self.write()
            with self.subTest(summary=summary),self.assertRaises(ValidationError):self.verify()
    def test_malformed_summary_mapping_is_rejected_without_type_leak(self):
        for value in ([],['accepted'],None,'accepted'):
            self.save(self.directory/'summary.json',value);self.receipt['summary']=value
            self.receipt['files']['summary.json']=sources.file_hash(self.directory/'summary.json');self.write()
            with self.subTest(value=value),self.assertRaises(ValidationError):self.verify()
    def test_raw_receipt_cannot_substitute_the_accepted_publication_schema(self):
        raw=self.base/'raw';raw.mkdir();names=['posterior.npz','sample_stats.npz','sampling.json']
        for name in names:(raw/name).write_bytes(b'wholly fabricated boundary bytes')
        receipt={'schema_version':1,'identity':self.identity,'fingerprint':sources.digest(self.identity),
                 'status':'sampled','completed_at':'fabricated','summary':{},'files':{n:sources.file_hash(raw/n) for n in names}}
        self.save(raw/'receipt.json',receipt)
        with self.assertRaises(ValidationError):sources.verify_receipt(raw,self.identity,names,status='sampled')
        del receipt['summary'];receipt['preparation_fingerprint']='d'*64;self.save(raw/'receipt.json',receipt)
        self.assertEqual(sources.verify_receipt(raw,self.identity,names,status='sampled'),receipt)
    def test_joint_and_deterministic_success_phases_cannot_be_crossed(self):
        common={k:deepcopy(self.receipt[k]) for k in ('schema_version','identity','fingerprint','files')}
        for joint,summary in [(False,{'accepted':True,'status':'diagnostics_passed'}),(True,{'diagnostics_passed':True,'status':'accepted'})]:
            value={**common,'summary':summary}
            if joint:value['raw_receipt_sha256']='e'*64
            self.save(self.directory/'summary.json',summary);value['files']['summary.json']=sources.file_hash(self.directory/'summary.json');self.save(self.directory/'receipt.json',value)
            with self.subTest(joint=joint),self.assertRaises(ValidationError):sources.verify_receipt(self.directory,self.identity,self.expected,status=None)
    def test_symlinked_artifact_receipt_directory_and_dangling_path_rejected(self):
        for name in [*self.expected,'receipt.json']:
            p=self.directory/name;outside=self.base/name;p.rename(outside);p.symlink_to(outside)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.verify()
            p.unlink();outside.rename(p)
        original=self.directory;outside=self.base/'outside';original.rename(outside);original.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValidationError):self.verify()
        original.unlink();original.symlink_to(self.base/'absent',target_is_directory=True)
        with self.assertRaises(ValidationError):self.verify()


class ReportingDocumentTests(unittest.TestCase):
    def test_subscripts_use_native_word_runs_with_arial_digits(self):
        xml=ET.fromstring('<root xmlns:w="'+doc.W+'">'+doc._p('FEV₁/FVC and FEF₂₅–₇₅')+'</root>');ns={'w':doc.W}
        subscript=[]
        for run in xml.findall('.//w:r',ns):
            text=''.join(x.text or '' for x in run.findall('w:t',ns));position=run.find('w:rPr/w:vertAlign',ns)
            if position is not None:
                self.assertEqual(position.attrib['{'+doc.W+'}val'],'subscript');subscript.append(text)
            self.assertFalse(any(c in text for c in '₁₂₅₇'))
        self.assertEqual(''.join(subscript),'12575')
    def test_font_family_style_and_embedding_permissions(self):
        for style,bold in [('Regular',False),('Bold',True)]:self.assertEqual(doc._font(fake_font(style),bold),hashlib.sha256(fake_font(style)).hexdigest())
        for data,bold in [(fake_font(family='Other'),False),(fake_font('Regular'),True),(fake_font('Bold'),False),(fake_font(flags=2),False),(fake_font(flags=0x200),False)]:
            with self.assertRaises(ValueError):doc._font(data,bold)
        for style,bold in [('Italic',False),('Bold Italic',True)]:
            with self.subTest(style=style),self.assertRaises(ValueError):doc._font(fake_font(style),bold)
    def test_editable_xml_preserves_spans_exact_text_and_arial(self):
        xml=ET.fromstring('<root xmlns:w="'+doc.W+'">'+doc._table(fixture_table())+'</root>')[0];ns={'w':doc.W}
        self.assertEqual(len(xml.findall('w:tr',ns)),3);self.assertEqual(len(xml.findall('.//w:gridSpan',ns)),2)
        self.assertEqual([x.attrib['{'+doc.W+'}val'] for x in xml.findall('.//w:gridSpan',ns)],['2','4'])
        self.assertEqual(len(xml.findall('.//w:tblHeader',ns)),1);self.assertEqual(len(xml.findall('.//w:cantSplit',ns)),3)
        self.assertEqual(sum(int(x.attrib['{'+doc.W+'}w']) for x in xml.findall('w:tblGrid/w:gridCol',ns)),10440)
        self.assertIn('Not estimable',[x.text for x in xml.findall('.//w:t',ns)])
        for font in xml.findall('.//w:rFonts',ns):self.assertEqual(font.attrib['{'+doc.W+'}ascii'],'Arial')
    def test_malformed_spans_never_produce_ambiguous_grids(self):
        for span in (0,-1,True,1.5,5):
            table=fixture_table();table['rows'][0]['cells'][0]['colspan']=span
            with self.subTest(span=span),self.assertRaises(ValueError):doc._table(table)
        table=fixture_table();table['rows'][2]['cells'].pop()
        with self.assertRaises(ValueError):doc._table(table)
    def test_deterministic_zip_uses_only_editable_tables_and_embedded_fonts(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b=Path(temporary)/'regular.ttf',Path(temporary)/'bold.ttf';a.write_bytes(fake_font());b.write_bytes(fake_font('Bold'))
            first,hashes=doc.document_bytes([fixture_table()],a,b);second,_=doc.document_bytes([fixture_table()],a,b)
        self.assertEqual(first,second);self.assertEqual(hashes,[hashlib.sha256(fake_font()).hexdigest(),hashlib.sha256(fake_font('Bold')).hexdigest()])
        with ZipFile(BytesIO(first)) as z:
            self.assertEqual(len(z.namelist()),9);self.assertFalse(any('/media/' in n for n in z.namelist()))
            for info in z.infolist():self.assertEqual(info.date_time,(2000,1,1,0,0,0))
            xml=ET.fromstring(z.read('word/document.xml'));self.assertEqual(len(xml.findall('.//{'+doc.W+'}tbl')),1)
            self.assertIn('Fabricated <title> & estimate',[x.text for x in xml.findall('.//{'+doc.W+'}t')])
            for i,data in enumerate((fake_font(),fake_font('Bold'))):
                stored=bytearray(z.read('word/fonts/arial'+str(i)+'.odttf'));key=UUID('7B32AF7C-124E-43DC-9731-485348FC020'+str(i+1)).bytes[::-1]
                for j in range(32):stored[j]^=key[j%16]
                self.assertEqual(stored,data)
            for n in z.namelist():
                if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))


class FabricatedTableRegistry:
    """Made-up scalar results isolate layout from scientific source adapters."""
    def __init__(self):self.calls=[];self.records={}
    def select(self,source,**where):
        self.calls.append((source,deepcopy(where)))
        values={'mean':-.015,'lower':-.125,'upper':.095,'reported_mean':-.015,'reported_lower':-.125,'reported_upper':.095,
                'estimate':-.0015,'ci_low':-.0125,'ci_high':.0095,'median':2,'q1':1,'q3':3,'sd':.5,
                'posterior_median':1.25,'count':1,'numerator':1,'denominator':9,'percent':100/9,
                'attenuation_percent':-2.5,'target_children':1,'target_observations':3,**where}
        key=source+'#'+json.dumps(where,sort_keys=True);r={'key':key,'values':values,'status':'available','semantics':{}}
        self.records[key]=r;return deepcopy(r)
    def get(self,key):return deepcopy(self.records[key])


class ReportingTableLayoutTests(unittest.TestCase):
    def test_all_ten_tables_have_unique_physical_coordinates_and_scientific_lineage(self):
        registry=FabricatedTableRegistry();built=tables.build_tables(registry)
        self.assertEqual([t['id'] for t in built],list(nt.TABLE_IDS));self.assertEqual(len({(t['reference_document'],t['reference_table']) for t in built}),10)
        scientific=0
        for t in built:
            coordinates=set()
            for row in t['rows']:
                self.assertEqual(sum(c['colspan'] for c in row['cells']),len(t['columns']))
                for c in row['cells']:
                    coordinate=(row['reference_row'],c['reference_column']);self.assertNotIn(coordinate,coordinates);coordinates.add(coordinate)
                    if c['result_key']:
                        scientific+=1;original=registry.get(c['result_key']);actual=fmt.cell(original,c['fields'],c['style'],c['digits'])
                        self.assertEqual(c['text'],c.get('prefix','')+actual['text']+c.get('suffix',''))
            self.assertTrue(t['footnotes'])
        self.assertEqual(scientific,261)
    def test_nested_contrasts_and_pre_post_samples_have_distinct_exact_selectors(self):
        registry=FabricatedTableRegistry();built=tables.build_tables(registry);main=next(t for t in built if t['id']=='main_table_2')
        rows=[r for r in main['rows'] if r['kind']=='data'];self.assertEqual(len(rows),12)
        for row in rows:
            selected=[registry.get(c['result_key']) for c in row['cells'] if c['result_key']]
            self.assertEqual([r['values']['contrast'] for r in selected],['TB <5 weighted','TB <1'])
        calls=[(s,w) for s,w in registry.calls if s.startswith('primary/')]
        self.assertTrue(any('/ratio_pre/' in s for s,w in calls));self.assertTrue(any('/ratio_post/' in s for s,w in calls))
        self.assertEqual(next(t for t in built if t['id']=='appendix_table_s3')['rows'][2]['cells'][1]['text'],'1 / 3')
    def test_merged_change_cells_keep_their_distinct_model_and_three_decimal_precision(self):
        registry=FabricatedTableRegistry();t=tables._table_s2(registry);row=next(r for r in t['rows'] if r['id']=='paired_change')
        self.assertEqual([c['colspan'] for c in row['cells']],[1,2,2]);self.assertEqual([c['digits'] for c in row['cells'][1:]],[3,3])
        self.assertEqual([registry.get(c['result_key'])['values']['design'] for c in row['cells'][1:]],['before5','infancy'])
        self.assertTrue(any('confidence interval' in f for f in t['footnotes']))
    def test_native_mechanics_fields_are_not_rescaled_by_the_table(self):
        registry=FabricatedTableRegistry();t=tables._table_s5(registry)
        for row in t['rows'][1:]:
            for c in row['cells'][1:]:self.assertEqual(c['fields'],['reported_mean','reported_lower','reported_upper']);self.assertEqual(c['text'],'-0.02 (-0.13 to 0.10)')
        self.assertTrue(any('already convert compliance' in f for f in t['footnotes']))


def fabricated_coverage():
    registry=results.ResultRegistry();source='primary/ratio_pre/estimates';registry.add(source,frame(source,[estimate()]),provenance())
    registry.add('preparation/flow',frame('preparation/flow',[{'node':'made_up_count','count':1,'status':'available'},{'node':'missing_count','count':None,'status':'not_reconstructable_from_supplied_inputs'}]))
    interval={'source':source,'where':{'contrast':'TB <1'},'fields':['mean','lower','upper'],'style':'interval','digits':2}
    count={'source':'preparation/flow','where':{'node':'made_up_count'},'fields':['count'],'style':'integer','digits':0}
    unavailable={**count,'where':{'node':'missing_count'}}
    built=[];entries=[];index=0
    numbered=['1','2','S1','S2','S3','S4','S4','S5','S6','S6']
    for table_index,identity in enumerate(nt.TABLE_IDS):
        row_count=31 if table_index<9 else 25;rows=[]
        for row in range(1,row_count+1):
            kind='estimate' if index<186 else 'count' if index<261 else 'header' if index<287 else 'context'
            binding=interval if kind=='estimate' else count
            if kind in ('estimate','count'):c=fmt.cell(registry.select(binding['source'],**binding['where']),binding['fields'],binding['style'],binding['digits'])
            else:c=fmt.literal('Synthetic context '+str(index))
            c.update(reference_column=1,colspan=1);rows.append({'id':str(row),'reference_row':row,'kind':'data','cells':[c]})
            entry={'id':'cell_'+str(index),'physical_table_id':identity,'reference_row':row,'reference_column':1,'colspan':1,'classification':kind}
            entry['binding' if kind in ('estimate','count') else 'label']=deepcopy(binding) if kind in ('estimate','count') else c['text'];entries.append(entry);index+=1
        built.append({'id':identity,'title':'Fabricated','reference_document':'fabricated','reference_table':str(table_index),'header_rows':0,'columns':[{'id':'only'}],'rows':rows,'footnotes':[]})
    structures=[{**nt._structure(t),'numbered_table_id':n} for t,n in zip(built,numbered)]
    sentences=[{'id':'sentence_'+str(i),'section':'Fabricated results','classification':'scientific','template':'A made-up result was {effect}.','tokens':{'effect':deepcopy(interval)},'context_numbers':[],'references':[]} for i in range(42)]
    sentences[0]['tokens']['effect']=unavailable
    sentences[1]['qualitative_evidence']=[{'binding':deepcopy(interval),'null':0,'review':'Interval includes the null.'}]
    return registry,built,{'schema_version':1,'table_structures':structures,'numeric_cells':entries,'sentences':sentences}


class ReportingNumericalTextTests(unittest.TestCase):
    def test_complete_cells_sentences_full_precision_and_missing_statuses(self):
        r,t,s=fabricated_coverage();out=nt.build_checks(r,t,s)
        self.assertEqual(out['coverage']['numeric_cells'],304);self.assertEqual(out['coverage']['scientific_cells'],261);self.assertEqual(out['coverage']['sentences'],42)
        self.assertEqual(out['coverage']['cell_classifications'],{'estimate':186,'count':75,'header':26,'context':17})
        self.assertEqual(out['sentences'][0]['status'],'contains_explicit_unavailable_facts');self.assertIn('Not available',out['sentences'][0]['text'])
        self.assertEqual(out['sentences'][2]['tokens']['effect']['full_precision_values'],[-.015,-.125,.095]);self.assertFalse(out['wording_review_required'])
    def test_missing_duplicate_wrong_class_and_unmapped_scientific_cells_fail(self):
        for mode in ('missing','duplicate_id','duplicate_coordinate','wrong_class','binding','span','display','table_order'):
            r,t,s=fabricated_coverage()
            if mode=='missing':s['numeric_cells'].pop()
            elif mode=='duplicate_id':s['numeric_cells'][1]['id']=s['numeric_cells'][0]['id']
            elif mode=='duplicate_coordinate':s['numeric_cells'][1]['reference_row']=1
            elif mode=='wrong_class':s['numeric_cells'][0]['classification']='context'
            elif mode=='binding':s['numeric_cells'][0]['binding']['digits']=3
            elif mode=='span':s['numeric_cells'][0]['colspan']=2
            elif mode=='display':t[0]['rows'][0]['cells'][0]['text']='999'
            else:t.reverse()
            with self.subTest(mode=mode),self.assertRaises(fmt.ReportingError):nt.build_checks(r,t,s)
    def test_sentence_inventory_and_placeholder_substitution_are_strict(self):
        for mode in ('missing','duplicate','unbound','attribute','index','format','conversion'):
            r,t,s=fabricated_coverage()
            if mode=='missing':s['sentences'].pop()
            elif mode=='duplicate':s['sentences'][1]['id']=s['sentences'][0]['id']
            else:s['sentences'][2]['template']={'unbound':'{missing}','attribute':'{effect.__class__}','index':'{effect[0]}','format':'{effect:>10}','conversion':'{effect!r}'}[mode]
            with self.subTest(mode=mode),self.assertRaises(fmt.ReportingError):nt.build_checks(r,t,s)
    def test_qualitative_claim_outside_interval_requires_wording_review(self):
        r,t,s=fabricated_coverage();s['sentences'][1]['qualitative_evidence'][0]['null']=1
        out=nt.build_checks(r,t,s);self.assertTrue(out['wording_review_required']);self.assertEqual(out['sentences'][1]['status'],'scientific_wording_review_required')
        self.assertEqual(out['sentences'][1]['tokens']['effect']['full_precision_values'],[-.015,-.125,.095])
    def test_display_comparison_never_changes_computed_values(self):
        r,t,s=fabricated_coverage();out=nt.build_checks(r,t,s);old=deepcopy(out)
        reference_cells=[{'id':c['id'],'displayed_text':c['display']} for c in out['cells']];reference_sentences=[{'id':x['id'],'text':x['text']} for x in out['sentences']]
        reference_cells[0]['displayed_text']='99.9 (2.0 to 100.0)'
        compared=nt.compare_references(out,reference_cells,reference_sentences);self.assertEqual(compared['table_numeric_changes'],1);self.assertEqual(out,old)
        self.assertEqual(nt.compare_references(out)['status'],'reference_not_supplied')
        with self.assertRaises(fmt.ReportingError):nt.compare_references(out,reference_cells,None)
    def test_numeric_comparison_handles_unicode_decimal_and_label_digits(self):
        self.assertEqual(nt._numbers('−1·25 to +0·5; 1,234 children; FEV₁'),[-1.25,.5,1234])
        self.assertEqual(nt._numbers('−1.25 to 0.5'),[-1.25,.5])


class ReportingExecutionTests(unittest.TestCase):
    def setUp(self):
        import shutil
        from dchs_lung_function import reporting_execution as execution,pipeline,sensitivity_execution,execution as preparation
        self.ex,self.pipeline,self.upstream,self.preparation=execution,pipeline,sensitivity_execution,preparation
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.base=Path(temporary.name).resolve();self.root=self.base/'source'
        for name in execution.source_hashes(ROOT):
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        self.work,self.output=self.base/'work',self.base/'output';self.work.mkdir();self.output.mkdir()
        self.runtime=self.base/'document-python';self.runtime.write_text('# wholly fabricated executable, never invoked\n');self.runtime.chmod(0o700)
        self.fonts={name:self.base/(name+'.ttf') for name in ('arial','arial_bold')}
        for name,path in self.fonts.items():path.write_bytes(fake_font('Bold' if name=='arial_bold' else 'Regular'))
        self.paths=[self.base/name for name in ('paths.json','analysis.json','primary.json','mechanics.json','supporting.json','wheeze.json','sensitivity.json','reporting.json')]
        local={'schema_version':1,'inputs':{},'directories':{'work':str(self.work),'outputs':str(self.output)},'fonts':{k:str(p) for k,p in self.fonts.items()}}
        execution.write_json(self.paths[0],local);execution.write_json(self.paths[1],{'schema_version':1,'event_end':'2030-01-01','respiratory_end':'2030-01-01'})
        for p in self.paths[2:7]:execution.write_json(p,{'schema_version':1,'chains':4,'draws':3200,'tune':2200,'cores':2,'target_accept':.95,'max_treedepth':10,'seed_offset':104729})
        execution.write_json(self.paths[-1],{'schema_version':1,'document_python':str(self.runtime)})
        self.prior={'accepted':True,'status':'accepted','through':'M12','selected_models':10,'reused_models':4,'preparation_fingerprint':'d'*64,
          'primary_models_verified':12,'binary_models_verified':28,'trajectory_analyses_verified':4,'mechanics_models_verified':6,'shared_factor_rows_verified':6,
          'supporting_joint_models_verified':7,'supporting_deterministic_fits_verified':10,'pretb_models_verified':17,'wheeze_models_verified':2,'feno_models_verified':11,
          'inherited_limitations':['G08 inherited covariance limitation.','Four antenatal counts unavailable.']}
        self.prep=self.work/'preparation_runs'/self.prior['preparation_fingerprint'];self.prep.mkdir(parents=True)
        self.upstream_file=self.prep/'aggregate.json';execution.write_json(self.upstream_file,{'fabricated':True})
        self.registry,self.tables,self.spec=fabricated_coverage();self.checks=nt.build_checks(self.registry,self.tables,self.spec)
        self.calls=[];self.document_bytes=b'FABRICATED DOCX BYTES; real authoring is never invoked'
        self.runtime_identity={'python':'3.12.test','implementation':'cpython','system':'synthetic','machine':'synthetic','zlib':'synthetic','executable_sha256':sources.file_hash(self.runtime)}
        def prior(*args,**kwargs):self.calls.append('upstream');return deepcopy(self.prior)
        def collect(*args):
            self.calls.append('collect');self.assertEqual(self.calls[-2],'upstream')
            return self.registry,[{'family':'fabricated','fingerprint':'e'*64}],{self.upstream_file:sources.file_hash(self.upstream_file)}
        def document(command,**kwargs):
            self.calls.append('render');self.assertIn('-I',command);self.assertEqual(command[0],str(self.runtime))
            Path(command[command.index('--output')+1]).write_bytes(self.document_bytes)
        self.collect=collect
        patches=[mock.patch.object(sensitivity_execution,'run_sensitivities',side_effect=prior),
                 mock.patch.object(preparation,'run_preparation',return_value={'fingerprint':self.prior['preparation_fingerprint']}),
                 mock.patch.object(execution,'collect_sources',side_effect=collect),mock.patch.object(execution,'build_tables',return_value=self.tables),
                 mock.patch.object(execution,'build_checks',return_value=self.checks),mock.patch.object(execution,'document_runtime',return_value=self.runtime_identity),
                 mock.patch.object(execution.subprocess,'run',side_effect=document)]
        for patch in patches:patch.start();self.addCleanup(patch.stop)
    def run_report(self,**kwargs):return self.ex.run_reporting(self.root,*self.paths,**kwargs)
    def destination(self,result):return self.output/'manuscript_reporting'/result['fingerprint']
    def clean(self):self.assertEqual(list(self.output.rglob('.staging-*')),[]);self.assertEqual(list(self.base.rglob('.preparation.lock')),[])
    def no_pointer(self):self.assertFalse((self.work/'reporting_current.json').exists());self.clean()
    def repair_hash(self,directory,name):
        r=json.loads((directory/'receipt.json').read_text());r['files'][name]=sources.file_hash(directory/name);self.ex.write_json(directory/'receipt.json',r)
    def test_success_and_resume_revalidate_then_regenerate_every_file(self):
        first=self.run_report();second=self.run_report(resume=True)
        self.assertFalse(first['resumed']);self.assertTrue(second['resumed']);self.assertEqual(first['fingerprint'],second['fingerprint'])
        self.assertEqual(self.calls,['upstream','collect','render']*2);self.assertEqual(first['coverage']['numeric_cells'],304);self.assertEqual(first['new_model_fits'],0)
        directory=self.destination(first);self.assertEqual({p.name for p in directory.iterdir()},set(self.ex.RESULT_FILES)|{'receipt.json'})
        self.assertEqual((directory/'manuscript_tables.docx').read_bytes(),self.document_bytes)
        self.assertEqual(json.loads((self.work/'reporting_current.json').read_text()),second);self.clean()
    def test_all_predecessor_counts_success_flags_and_identity_are_required(self):
        old=deepcopy(self.prior)
        changes=[('accepted',False),('status','failed'),('through','M11'),('selected_models',9),('selected_models',10.),('reused_models',4.),('preparation_fingerprint','../source')]
        changes.extend((k,True) for k in old if k.endswith('_verified'))
        for key,value in changes:
            self.prior=deepcopy(old);self.prior[key]=value;self.calls=[]
            with self.subTest(key=key),self.assertRaises(ValidationError):self.run_report()
            self.assertNotIn('collect',self.calls);self.no_pointer()
    def test_invalid_arguments_are_rejected_before_upstream(self):
        for kwargs in ({'jobs':0},{'jobs':4},{'jobs':True},{'jobs':1.5},{'resume':1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValidationError):self.run_report(**kwargs)
        self.assertEqual(self.calls,[])
    def test_failed_upstream_and_unsupported_wording_never_publish(self):
        with mock.patch.object(self.upstream,'run_sensitivities',side_effect=ValidationError('fabricated failure')),self.assertRaises(ValidationError):self.run_report()
        self.no_pointer();self.checks['wording_review_required']=True
        with self.assertRaises(ValidationError):self.run_report()
        self.assertNotIn('render',self.calls);self.no_pointer()
    def test_wrong_root_or_stale_imported_source_rejected(self):
        path=self.root/'src/dchs_lung_function/reporting/formatting.py';path.write_bytes(path.read_bytes()+b'\n# mutation\n')
        with self.assertRaises(ValidationError):self.run_report()
        self.assertEqual(self.calls,[])
        original=self.ex.source_hashes
        with mock.patch.object(self.ex,'source_hashes',side_effect=lambda p:{} if Path(p)==self.ex._IMPORTED_ROOT else original(p)),self.assertRaises(ValidationError):self.run_report()
    def test_explicit_resume_required_and_both_output_and_work_locks_respected(self):
        first=self.run_report();pointer=(self.work/'reporting_current.json').read_bytes()
        with self.assertRaises(ValidationError):self.run_report()
        with self.ex.preparation_locks(self.work/'manuscript_reporting',self.output/'manuscript_reporting'):
            with self.assertRaises(ValidationError):self.run_report(resume=True)
        self.assertEqual((self.work/'reporting_current.json').read_bytes(),pointer);self.assertTrue(self.destination(first).is_dir());self.clean()
    def test_missing_extra_and_hash_tampered_artifacts_rejected_on_resume(self):
        directory=self.destination(self.run_report());pointer=(self.work/'reporting_current.json').read_bytes()
        for mode in ('missing','extra','tamper'):
            path=directory/'tables.json';old=path.read_bytes()
            if mode=='missing':path.unlink()
            elif mode=='extra':(directory/'unexpected.json').write_text('{}')
            else:path.write_bytes(old+b'changed')
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_report(resume=True)
            path.write_bytes(old);(directory/'unexpected.json').unlink(missing_ok=True)
        self.assertEqual((self.work/'reporting_current.json').read_bytes(),pointer);self.clean()
    def test_repaired_hash_cannot_hide_any_changed_regenerated_payload(self):
        directory=self.destination(self.run_report());original_receipt=(directory/'receipt.json').read_bytes()
        for name in self.ex.RESULT_FILES:
            path=directory/name;original=path.read_bytes();path.write_bytes(original+b' ');self.repair_hash(directory,name)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_report(resume=True)
            path.write_bytes(original);(directory/'receipt.json').write_bytes(original_receipt)
        self.clean()
    def test_contradictory_or_extra_receipt_fields_rejected(self):
        directory=self.destination(self.run_report());path=directory/'receipt.json';original=path.read_bytes()
        for mode in ('summary','extra','boolean','identity','traversal'):
            value=json.loads(original)
            if mode=='summary':value['summary']['coverage']['numeric_cells']=303
            elif mode=='extra':value['participant_records']=['FABRICATED_PRIVATE']
            elif mode=='boolean':value['schema_version']=True
            elif mode=='identity':value['identity']['schema_version']=2
            else:value['files']['../private.json']='a'*64
            self.ex.write_json(path,value)
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_report(resume=True)
            path.write_bytes(original)
    def test_symlink_receipt_artifact_parent_and_dangling_destination_rejected(self):
        directory=self.destination(self.run_report())
        for name in ('receipt.json','tables.json'):
            path=directory/name;outside=self.base/name;path.rename(outside);path.symlink_to(outside)
            with self.subTest(name=name),self.assertRaises(ValidationError):self.run_report(resume=True)
            path.unlink();outside.rename(path)
        outside=self.base/'old-result';directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_report(resume=True)
        directory.unlink();directory.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaises(ValidationError):self.run_report(resume=True)
        self.clean()
    def test_private_roots_and_protected_configuration_runtime_fonts_cannot_overlap(self):
        original=json.loads(self.paths[0].read_text())
        for work,output in [(self.root/'work',self.output),(self.work,self.work/'nested'),(self.base,self.output)]:
            local=deepcopy(original);local['directories']={'work':str(work),'outputs':str(output)};self.ex.write_json(self.paths[0],local)
            with self.subTest(work=work),self.assertRaises(ValidationError):self.run_report()
            self.assertEqual(self.calls,[])
        self.ex.write_json(self.paths[0],original)
        font=self.work/'font.ttf';font.write_bytes(fake_font());local=deepcopy(original);local['fonts']['arial']=str(font);self.ex.write_json(self.paths[0],local)
        with self.assertRaises(ValidationError):self.run_report()
        self.assertEqual(self.calls,[])
    def test_config_font_upstream_runtime_and_source_changes_before_publication_rejected(self):
        render=self.ex.render_payloads
        cases=[self.paths[1],self.fonts['arial'],self.upstream_file,self.root/'README.md']
        for path in cases:
            old=path.read_bytes()
            def mutate(*args):render(*args);path.write_bytes(old+b'changed')
            with mock.patch.object(self.ex,'render_payloads',side_effect=mutate):
                with self.subTest(path=path.name),self.assertRaises(ValidationError):self.run_report()
            path.write_bytes(old);self.no_pointer();self.assertEqual(list((self.output/'manuscript_reporting').glob('*/receipt.json')),[])
        with mock.patch.object(self.ex,'document_runtime',side_effect=[self.runtime_identity,{**self.runtime_identity,'python':'changed'}]),self.assertRaises(ValidationError):self.run_report()
        self.no_pointer()
    def test_final_input_change_rolls_back_fresh_publication_but_preserves_cached_result(self):
        original=self.ex.verify_publication;old=self.upstream_file.read_bytes()
        def mutate(*args):original(*args);self.upstream_file.write_bytes(old+b'changed')
        with mock.patch.object(self.ex,'verify_publication',side_effect=mutate),self.assertRaises(ValidationError):self.run_report()
        self.assertEqual(list((self.output/'manuscript_reporting').glob('*/receipt.json')),[]);self.no_pointer()
        self.upstream_file.write_bytes(old);accepted=self.run_report();pointer=(self.work/'reporting_current.json').read_bytes()
        with mock.patch.object(self.ex,'verify_publication',side_effect=mutate),self.assertRaises(ValidationError):self.run_report(resume=True)
        self.assertTrue(self.destination(accepted).is_dir());self.assertEqual((self.work/'reporting_current.json').read_bytes(),pointer);self.clean()
    def test_changed_preparation_at_final_validation_blocks_completion(self):
        with mock.patch.object(self.preparation,'run_preparation',return_value={'fingerprint':'0'*64}),self.assertRaises(ValidationError):self.run_report()
        self.no_pointer()
    def test_raced_destination_is_preserved_without_success_pointer(self):
        original=self.ex.verify_publication;created=[]
        def race(directory,identity,*args):
            original(directory,identity,*args);destination=self.output/'manuscript_reporting'/self.ex.digest(identity);destination.mkdir();created.append(destination)
        with mock.patch.object(self.ex,'verify_publication',side_effect=race),self.assertRaises(ValidationError):self.run_report()
        self.assertEqual(list(created[0].iterdir()),[]);self.no_pointer()
    def test_render_failure_is_sanitized_and_failed_evidence_is_private(self):
        import subprocess
        with mock.patch.object(self.ex.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['private'],stderr=b'FABRICATED_PRIVATE_RECORD')):
            with self.assertRaisesRegex(ValidationError,'Editable document rendering failed'):self.run_report()
        self.no_pointer();self.assertTrue(list((self.work/'manuscript_reporting').glob('failed-*')))
        self.assertEqual(list((self.output/'manuscript_reporting').iterdir()),[])
    def test_renderer_unexpected_file_or_missing_document_cannot_publish(self):
        for mode in ('extra','missing'):
            def bad(command,**kwargs):
                directory=Path(command[command.index('--output')+1]).parent
                if mode=='extra':(directory/'manuscript_tables.docx').write_bytes(b'mock');(directory/'extra').write_text('fabricated')
            with mock.patch.object(self.ex.subprocess,'run',side_effect=bad):
                with self.subTest(mode=mode),self.assertRaises(ValidationError):self.run_report()
            self.no_pointer()
    def test_pointer_failure_rolls_back_new_publication(self):
        with mock.patch.object(self.ex,'write_pointer',side_effect=OSError('fabricated pointer failure')),self.assertRaises(OSError):self.run_report()
        self.assertEqual(list((self.output/'manuscript_reporting').glob('*/receipt.json')),[]);self.no_pointer()
    def test_output_lock_blocks_another_work_root(self):
        other=self.base/'other-work';other.mkdir();config=json.loads(self.paths[0].read_text());config['directories']['work']=str(other);self.ex.write_json(self.paths[0],config)
        for p in (self.work/'manuscript_reporting',self.output/'manuscript_reporting'):p.mkdir()
        with self.ex.preparation_locks(self.work/'manuscript_reporting',self.output/'manuscript_reporting'):
            with self.assertRaises(ValidationError):self.run_report()
        self.assertFalse((other/'reporting_current.json').exists());self.clean()
    def test_preparation_parent_symlink_is_rejected_before_loading(self):
        actual=self.base/'actual-prep';actual.mkdir();linked=self.base/'linked-prep';linked.symlink_to(actual,target_is_directory=True)
        with mock.patch.object(self.preparation,'verify_artifacts',side_effect=AssertionError('must not read preparation')):
            with self.assertRaises(ValidationError):sources.collect_sources(self.root,linked,{'work':self.work,'outputs':self.output},{})
    def test_symlinked_upstream_ancestor_created_late_is_rejected(self):
        original=self.ex.render_payloads;directory=self.upstream_file.parent;outside=self.base/'moved-prep'
        def mutate(*args):original(*args);directory.rename(outside);directory.symlink_to(outside,target_is_directory=True)
        with mock.patch.object(self.ex,'render_payloads',side_effect=mutate),self.assertRaises(ValidationError):self.run_report()
        self.no_pointer()


class ReportingConfigurationAndCliTests(unittest.TestCase):
    def setUp(self):
        from dchs_lung_function import reporting_execution as execution,pipeline
        self.ex,self.pipeline=execution,pipeline;t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.base=Path(t.name).resolve()
        self.exe=self.base/'python';self.exe.write_text('# fabricated executable\n');self.exe.chmod(0o700);self.path=self.base/'reporting.json'
        self.config={'schema_version':1,'document_python':'python'};self.ex.write_json(self.path,self.config)
    def test_reporting_config_resolves_relative_paths_and_requires_paired_references(self):
        self.assertEqual(self.ex.reporting_config(self.path),{'document_python':self.exe})
        for mode in ('one_reference','unknown','boolean_version','missing_executable','blank'):
            value=deepcopy(self.config)
            if mode=='one_reference':value['reference_cells']='cells.json'
            elif mode=='unknown':value['extra']='secret'
            elif mode=='boolean_version':value['schema_version']=True
            elif mode=='missing_executable':value['document_python']='absent'
            else:value['document_python']=' '
            self.ex.write_json(self.path,value)
            with self.subTest(mode=mode),self.assertRaises(ValidationError):self.ex.reporting_config(self.path)
    def test_nonexecutable_runtime_and_symlinked_reference_rejected(self):
        self.exe.chmod(0o600)
        with self.assertRaises(ValidationError):self.ex.reporting_config(self.path)
        self.exe.chmod(0o700);real=self.base/'real.json';real.write_text('[]');link=self.base/'link.json';link.symlink_to(real)
        self.ex.write_json(self.path,{**self.config,'reference_cells':'link.json','reference_sentences':'real.json'})
        with self.assertRaises(ValidationError):self.ex.reporting_config(self.path)
    def test_runtime_preflight_exact_metadata_and_sanitized_failures(self):
        import subprocess
        good={'python':'3.12','implementation':'cpython','system':'synthetic','machine':'synthetic','zlib':'synthetic'}
        for value in (good,{**good,'private':'bad'},[],{**good,'python':None}):
            run=mock.Mock(stdout=json.dumps(value))
            with mock.patch.object(self.ex.subprocess,'run',return_value=run):
                if value==good:self.assertEqual(self.ex.document_runtime(self.exe)['executable_sha256'],sources.file_hash(self.exe))
                else:
                    with self.assertRaises(ValidationError):self.ex.document_runtime(self.exe)
        for error in (OSError('FABRICATED_PRIVATE'),subprocess.TimeoutExpired('private',30)):
            with mock.patch.object(self.ex.subprocess,'run',side_effect=error),self.assertRaisesRegex(ValidationError,'preflight failed'):self.ex.document_runtime(self.exe)
    def cli(self,arguments,result=None,error=None):
        from contextlib import redirect_stdout,redirect_stderr
        out,err=io.StringIO(),io.StringIO()
        with mock.patch.object(self.ex,'run_reporting',return_value=result,side_effect=error) as call,redirect_stdout(out),redirect_stderr(err):code=self.pipeline.main(arguments,root=ROOT)
        return code,out.getvalue(),err.getvalue(),call
    def test_cli_routes_all_five_sampling_flags_and_private_reporting_config(self):
        flags=['--through','M13','--config','paths.json','--analysis','analysis.json','--sampling','a.json','--mechanics-sampling','b.json','--supporting-sampling','c.json','--wheeze-sampling','d.json','--sensitivity-sampling','e.json','--reporting','r.json','--resume','--jobs','3']
        result={'accepted':True,'status':'accepted','coverage':{'physical_tables':10,'numeric_cells':304,'sentences':42}}
        code,out,err,call=self.cli(flags,result);self.assertEqual(code,0);self.assertIn('304',out);self.assertEqual(call.call_args.args[1:],('paths.json','analysis.json','a.json','b.json','c.json','d.json','e.json','r.json'));self.assertEqual(call.call_args.kwargs,{'jobs':3,'resume':True})
    def test_cli_refuses_partial_models_missing_reporting_and_unavailable_modes(self):
        for args in (['--through','M13','--config','x','--analysis','y'],['--through','M13','--config','x','--analysis','y','--reporting','z','--targets','ratio_pre'],['--through','M12','--reporting','z'],['--full'],['--render']):
            code,_,_,call=self.cli(args);self.assertEqual(code,2);call.assert_not_called()
    def test_cli_failure_does_not_print_participant_records(self):
        args=['--through','M13','--config','x','--analysis','y','--reporting','z']
        for error in (OSError('FABRICATED_PRIVATE_RECORD'),RuntimeError('FABRICATED_PRIVATE_RECORD'),ValueError('FABRICATED_PRIVATE_RECORD')):
            code,out,err,_=self.cli(args,error=error);self.assertEqual(code,2);self.assertNotIn('FABRICATED_PRIVATE_RECORD',out+err)
