"""Fingerprint, fit, and validate wheeze age-specific wheeze joint models in private output directories.

Raw sampling identity is separate from validation identity: changed reporting
or diagnostic code rechecks verified draws without pretending to refit them.
"""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .execution import (artifact_subtree, canonical_bytes, file_hash, load_artifacts,
                        preparation_locks, verify_artifacts, write_json,
                        resolve_paths,run_preparation,write_pointer)
from .pipeline import ValidationError,read_json as _read_json
from .models.wheeze import TARGETS, build_model, validate_contract, contract_key, fit_spec
from .model_runtime import configure_runtime,model_runtime_identity
from .wheeze_sampling import resolved_sampling,sample_contract,expected_posterior_shapes

CRITERIA = {'min_chains':4,'min_draws':1600,'max_rhat':1.01,
            'min_bulk_ess':400,'min_tail_ess':400,'min_bfmi':.3,
            'max_divergences':0,'max_treedepth_hits':0}
COMPUTE_SOURCES = ('models/wheeze.py','wheeze_sampling.py','sampling.py','model_runtime.py')
VALIDATION_SOURCES = ('wheeze_execution.py','diagnostics.py','execution.py','pipeline.py',
                      'analysis/wheeze_results.py','analysis/wheeze_standardization.py')
_LOADED_RUNNER_HASH = file_hash(Path(__file__))
TABLES = ('effects','contrasts','global_tests','support_counts','posterior_predictive_cells','standardized_rr')
RESULT_FILES = ('diagnostics.json','summary.json','result_metadata.json','ppc_replicates.npz',
                *(name+'.csv' for name in TABLES))


def read_json(path, label):
    value = _read_json(path, label)
    if not isinstance(value, dict):
        raise ValidationError(label+' must be an object.')
    return value


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def source_hashes(root, names):
    return {name:file_hash(Path(root)/'src/dchs_lung_function'/name) for name in names}



# Bind the imported execution session before any caller can change source files.
_LOADED_SOURCE_HASHES = {name:file_hash(Path(__file__).parent/name)
                         for name in (*COMPUTE_SOURCES, *VALIDATION_SOURCES)}


def assert_loaded_sources(root):
    local = Path(__file__).resolve().parents[2]
    if (source_hashes(root, _LOADED_SOURCE_HASHES) != _LOADED_SOURCE_HASHES
            or source_hashes(local, _LOADED_SOURCE_HASHES) != _LOADED_SOURCE_HASHES):
        raise ValidationError('Wheeze sources changed after import or differ from the supplied source tree; restart the process.')


def contract_descriptor(preparation_directory, target):
    if target not in TARGETS:
        raise ValidationError('Unknown wheeze model target.')
    directory = Path(preparation_directory).resolve()
    receipt = read_json(directory/'receipt.json','Preparation receipt')
    if (receipt.get('through')!='M03' or receipt.get('status')!='prepared'
            or file_hash(directory/'artifacts.json')!=receipt.get('artifact_manifest_sha256')):
        raise ValidationError('Wheeze fitting requires a complete verified M03 preparation.')
    if digest(receipt.get('identity'))!=receipt.get('fingerprint'):
        raise ValidationError('Preparation receipt fingerprint is inconsistent.')
    manifest = verify_artifacts(directory)
    node = artifact_subtree(manifest,tuple(fit_spec(target)['artifact_path']))
    files = set()
    def visit(value):
        if isinstance(value,dict):
            if 'file' in value:files.add(value['file'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(node)
    if not files.issubset(manifest['files']):
        raise ValidationError('A wheeze contract refers to an unverified preparation file.')
    identity = {'tree':node,'files':{name:manifest['files'][name] for name in sorted(files)}}
    return {'sha256':digest(identity),'preparation_fingerprint':receipt['fingerprint']}


def sampling_identity(root, descriptor, target, settings):
    identity = {'schema_version':1,'target':target,'contract_sha256':descriptor['sha256'],
                'sampling':resolved_sampling(target,settings),
                'source':source_hashes(root,COMPUTE_SOURCES),
                'environment_lock_sha256':file_hash(Path(root)/'requirements-models.lock.txt'),
                'runtime':model_runtime_identity()}
    return digest(identity),identity


def validation_identity(root, fit_fingerprint, criteria):
    identity = {'schema_version':1,'fit_fingerprint':fit_fingerprint,
                'source':source_hashes(root,VALIDATION_SOURCES),'criteria':criteria,
                'runtime':model_runtime_identity()}
    return digest(identity),identity


def validate_posterior_inventory(posterior, expected):
    if set(posterior)!=set(expected):
        raise ValidationError('Saved posterior does not contain the complete model variable inventory.')
    if any(np.asarray(posterior[name]).shape!=shape for name,shape in expected.items()):
        raise ValidationError('Saved posterior does not have the complete model coordinate shapes.')


def validate_sampling_metadata(metadata,identity,posterior,stats):
    controls=identity['sampling']
    if not isinstance(metadata,dict) or any(metadata.get(key)!=expected for key,expected in (
        ('target',identity['target']),('controls',controls),('runtime',identity['runtime']),
        ('realized_chains',controls['chains']),('realized_draws_per_chain',controls['draws']),
        ('realized_tuning_steps',controls['tune']))):
        raise ValidationError('Realized sampling metadata disagrees with the requested fit identity.')
    for field,arrays in [('posterior_variables',posterior),('sample_stat_variables',stats)]:
        names=metadata.get(field)
        if not isinstance(names,list) or len(names)!=len(set(names)) or set(names)!=set(arrays):
            raise ValidationError('Realized sampling variable metadata disagrees with saved arrays.')


def read_array_archive(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def verify_files(directory, entries, required):
    directory = Path(directory)
    if (directory.is_symlink() or not directory.is_dir()
            or (directory/'receipt.json').is_symlink()
            or not (directory/'receipt.json').is_file()):
        raise ValidationError('Wheeze publication directories and receipts must be regular nonsymlink artifacts.')
    if (not isinstance(entries,dict) or set(entries) != set(required)
            or {p.name for p in directory.iterdir()} != set(required) | {'receipt.json'}):
        raise ValidationError('Model receipt lacks required artifact identities.')
    for name,expected in entries.items():
        path=Path(directory)/name
        if (Path(name).name!=name or path.is_symlink() or not path.is_file()
                or file_hash(path)!=expected):
            raise ValidationError('A saved model artifact is missing or changed.')


def safe_roots(root,preparation_directory,work,output):
    root,prep,work,output=map(lambda p:Path(p).resolve(),(root,preparation_directory,work,output))
    overlap=lambda a,b:a==b or a in b.parents or b in a.parents
    if overlap(work,output) or any(overlap(folder,protected)
        for folder in (work,output) for protected in (root,prep)):
        raise ValidationError('Wheeze work/output roots must be separate from source and preparation artifacts.')
    for folder in (work,output):folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    return root,prep,work,output


def _validate_results(results, target, validated, posterior, ppc_seed):
    """Require the declared aggregate schema before any result is published."""
    from .analysis.wheeze_results import (TABLE_COLUMNS, PROCESSES, SUPPORT_CELL_KEYS,
        PPC_CELL_KEYS, PPC_REPLICATE_KEYS, TRANSFORMS)
    from .models.wheeze import EFFECT_LABELS
    def require(condition):
        if not condition:
            raise ValidationError('Wheeze result schema, cell inventory or aggregate values are inconsistent.')
    rows = dict(zip(TABLES,(6,6,2,22,22,3 if target=='primary_sex' else 0)))
    for name in TABLES:
        frame = getattr(results,name)
        require(isinstance(frame,pd.DataFrame) and list(frame.columns)==TABLE_COLUMNS[name]
                and len(frame)==rows[name] and frame.scenario.eq(target).all())
    effects, contrasts, globals_ = results.effects,results.contrasts,results.global_tests
    require(list(zip(effects.process,effects.age_band)) ==
            [(p,EFFECT_LABELS[b]) for p in PROCESSES for b in (1,2,3)])
    require(effects.exposure.eq('TB during infancy').all()
            and effects.estimand.eq('Post-infancy-TB versus pre/no-TB time within attained-age band').all()
            and effects.measure.tolist()==['conditional odds ratio']*3+['conditional incidence rate ratio']*3)
    require(list(zip(contrasts.process,contrasts.contrast)) == [(p,f'{EFFECT_LABELS[l]} versus {EFFECT_LABELS[e]}')
            for p in PROCESSES for l,e in ((2,1),(3,1),(3,2))]
            and contrasts.measure.tolist()==['OR ratio']*3+['IRR ratio']*3)
    for frame in (effects,contrasts):
        require(np.isfinite(frame.select_dtypes(include='number').to_numpy()).all())
        for name in ('posterior_mean','posterior_median','cri_2_5','cri_97_5'):
            require(pd.api.types.is_numeric_dtype(frame[name]) and frame[name].gt(0).all())
        require((frame.cri_2_5<=frame.posterior_median).all() and (frame.posterior_median<=frame.cri_97_5).all())
        for name in [c for c in frame if c.startswith('posterior_probability')]:
            require(pd.api.types.is_numeric_dtype(frame[name]) and frame[name].between(0,1).all())
        require((frame.posterior_probability_above_1+frame.posterior_probability_below_1<=1+1e-12).all())
    require(globals_.process.tolist()==list(PROCESSES) and globals_.degrees_of_freedom.eq(2).all()
            and globals_.null_hypothesis.eq('Equal infancy-TB log effects at ages 1-<3, 3-<5, and 5-10.5').all())
    for name in TABLE_COLUMNS['global_tests'][3:]:
        require(pd.api.types.is_numeric_dtype(globals_[name]) and np.isfinite(globals_[name]).all())
        require(globals_[name].between(0,1).all() if name=='reference_p_value' or name.startswith('posterior_probability')
                else globals_[name].ge(0).all())
    # Reconcile the inexpensive scientific summaries directly with primitive
    # log effects, independently of the combined report object.
    from scipy.stats import chi2
    for process_index,name in enumerate(('scheduled_infancy_post_log_or','episode_infancy_post_log_rr')):
        beta=np.asarray(posterior[name]).reshape(-1,4)
        for frame,draws_list in ((effects,[np.exp(beta[:,b]) for b in (1,2,3)]),
                                (contrasts,[np.exp(beta[:,l]-beta[:,e]) for l,e in ((2,1),(3,1),(3,2))])):
            for j,draws in enumerate(draws_list):
                row=frame.iloc[process_index*3+j]
                expected=[draws.mean(),np.median(draws),np.quantile(draws,.025),np.quantile(draws,.975),
                          np.mean(draws>1),np.mean(draws<1)]
                require(np.allclose(row[['posterior_mean','posterior_median','cri_2_5','cri_97_5',
                    'posterior_probability_above_1','posterior_probability_below_1']].to_numpy(float),expected,
                    rtol=1e-12,atol=1e-12))
                if frame is contrasts:
                    require(np.isclose(row.posterior_probability_later_effect_smaller,np.mean(draws<1),rtol=1e-12,atol=1e-12))
        diffs=np.column_stack((beta[:,2]-beta[:,1],beta[:,3]-beta[:,1]))
        mean=diffs.mean(axis=0);stat=float(mean@np.linalg.pinv(np.cov(diffs,rowvar=False))@mean)
        expected=[stat,2.,chi2.sf(stat,2),np.mean((beta[:,1]>beta[:,2])&(beta[:,1]>beta[:,3])),
                  np.mean((beta[:,1]>beta[:,2])&(beta[:,2]>beta[:,3]))]
        require(np.allclose(globals_.iloc[process_index,3:].to_numpy(float),expected,rtol=1e-12,atol=1e-12))
    support,ppc = results.support_counts,results.posterior_predictive_cells
    for frame,keys in ((support,SUPPORT_CELL_KEYS),(ppc,PPC_CELL_KEYS)):
        require(list(zip(frame.process,frame.age_band,frame.exposure_group)) ==
                [(p,EFFECT_LABELS[b],g) for p in PROCESSES for b,g in keys])
    for name in ('children','observations_or_intervals','events_or_positive'):
        values=support[name].to_numpy()
        require(values.dtype.kind in 'iuf' and np.isfinite(values).all()
                and (values>=0).all() and np.equal(values,np.floor(values)).all())
    counts=validated['counts']
    require(support.children.le(validated['joint_children']).all()
            and support.children.le(support.observations_or_intervals).all()
            and support.iloc[:11].person_years.isna().all()
            and np.isfinite(support.iloc[11:].person_years).all()
            and support.iloc[11:].person_years.ge(0).all()
            and support.iloc[:11].events_or_positive.le(support.iloc[:11].observations_or_intervals).all())
    for index,(nkey,ykey) in enumerate((('scheduled_observations','scheduled_positive'),('episode_intervals','recurrent_episodes'))):
        block=support.iloc[index*11:(index+1)*11]
        require(int(block.observations_or_intervals.sum())==counts[nkey]
                and int(block.events_or_positive.sum())==counts[ykey])
    require(np.isclose(support.iloc[11:].person_years.sum(),counts['person_years'],rtol=1e-12,atol=1e-12))
    groups={'post infancy TB':'post-TB after infancy TB','post other TB':'post-TB after TB at age >=1','pre/no TB':'pre/no-TB'}
    indexed=support.set_index(['process','age_band','exposure_group'])
    for i,row in ppc.iterrows():
        selected=indexed.loc[(row.process,row.age_band,groups[row.exposure_group])]
        scheduled=row.process==PROCESSES[0]
        require(row.units==('proportion positive' if scheduled else 'episodes per 1000 child-years')
                and row.observed_events==selected.events_or_positive)
        denom=selected.observations_or_intervals if scheduled else selected.person_years
        require(np.isclose(row.observed_denominator,denom,rtol=1e-12,atol=1e-12))
        stats=row[['observed_statistic','replicated_mean','replicated_2_5','replicated_97_5']].to_numpy(float)
        if denom==0:
            require(np.isnan(stats).all() and row.observed_events==0)
        else:
            require(np.isfinite(stats).all() and (stats>=0).all() and (not scheduled or (stats<=1).all())
                    and row.replicated_2_5<=row.replicated_97_5
                    and np.isclose(row.observed_statistic,row.observed_events*(1 if scheduled else 1000)/denom,
                                   rtol=1e-12,atol=1e-12))
    replicas=results.ppc_replicates
    shape=np.asarray(posterior['scheduled_intercept']).shape
    k=min(200,int(np.prod(shape)))
    require(isinstance(replicas,dict) and set(replicas)==set(PPC_REPLICATE_KEYS))
    for name,values in replicas.items():
        require(isinstance(values,np.ndarray) and values.dtype.kind in 'iuf'
                and values.shape==((k,) if name=='draw_indices' else (k,22)))
    require(np.array_equal(replicas['draw_indices'],np.linspace(0,np.prod(shape)-1,k,dtype=int)))
    values=replicas['counts']
    require(np.isfinite(values).all() and (values>=0).all() and np.equal(values,np.floor(values)).all())
    denom=ppc.observed_denominator.to_numpy(float); nonempty=denom>0
    factor=np.r_[np.ones(11),np.full(11,1000.)]
    require((values[:,:11]<=denom[:11]).all() and (values[:,~nonempty]==0).all())
    for name in ('statistics','conditional_means','conditional_variances'):
        values=replicas[name]
        require(np.isfinite(values[:,nonempty]).all() and (values[:,nonempty]>=0).all()
                and np.isnan(values[:,~nonempty]).all())
    require(np.allclose(replicas['statistics'][:,nonempty],
                        replicas['counts'][:,nonempty]*factor[nonempty]/denom[nonempty],rtol=1e-12,atol=1e-12))
    require(np.allclose(ppc.loc[nonempty,'replicated_mean'],replicas['statistics'][:,nonempty].mean(axis=0),rtol=1e-12,atol=1e-12)
            and np.allclose(ppc.loc[nonempty,'replicated_2_5'],np.quantile(replicas['statistics'][:,nonempty],.025,axis=0),rtol=1e-12,atol=1e-12)
            and np.allclose(ppc.loc[nonempty,'replicated_97_5'],np.quantile(replicas['statistics'][:,nonempty],.975,axis=0),rtol=1e-12,atol=1e-12))
    rr=results.standardized_rr
    if target=='primary_sex':
        require(rr.age_band.tolist()==list(EFFECT_LABELS[1:]) and rr.age.tolist()==['1-<3','3-<5','5-10.5']
                and rr.figure.eq('Age-localized').all() and rr.axis.eq('attained_age_band').all()
                and rr.measure.eq('population-standardized scheduled current-wheeze probability ratio').all())
        for name in TABLE_COLUMNS['standardized_rr'][6:]:
            require(pd.api.types.is_numeric_dtype(rr[name]) and np.isfinite(rr[name]).all())
        for prefix in ('or','rr','no_tb_percent','tb_percent','difference_pp'):
            require((rr[prefix+'_low']<=rr[prefix+'_median']).all() and (rr[prefix+'_median']<=rr[prefix+'_high']).all())
        require(rr.probability_above_1.between(0,1).all())
        for prefix in ('or','rr'):
            require((rr[[prefix+'_'+s for s in ('low','median','high')]]>0).all().all())
        for prefix in ('no_tb_percent','tb_percent'):
            require(rr[[prefix+'_'+s for s in ('low','median','high')]].apply(lambda c:c.between(0,100)).all().all())
        require(rr[['difference_pp_'+s for s in ('low','median','high')]].apply(lambda c:c.between(-100,100)).all().all())
    meta=results.metadata
    require(isinstance(meta,dict) and set(meta)=={'model_id','target','counts','posterior','posterior_predictive','standardization','global_test','privacy'}
            and meta['model_id']=='WHEEZE-JOINT' and meta['target']==target and meta['counts']==counts)
    require(set(meta['posterior'])=={'chains','draws_per_chain','posterior_variables','maximum_transform_errors'}
            and set(meta['posterior_predictive'])=={'seed','replicates','selection','rng','cells','empty_cell_semantics','conditional_mean_checks','note'})
    require(meta['global_test']=='Posterior Wald reference chi-square on two correlated log-effect contrasts; no multiplicity adjustment.'
            and meta['privacy']=='Only aggregate cells and aggregate predictive replicates are returned.')
    for key,text in {
        'selection':'evenly spaced chain-major flattened draws',
        'rng':'numpy.default_rng; scheduled draws then episode draws, source row order',
        'empty_cell_semantics':'zero aggregate counts; normalized statistics are NaN',
        'note':'Predictive intervals describe aggregate replicated data; visual coverage is not a convergence diagnostic.'}.items():
        require(meta['posterior_predictive'][key]==text)
    require(meta['posterior']['chains']==shape[0] and meta['posterior']['draws_per_chain']==shape[1]
            and meta['posterior']['posterior_variables']==len(posterior)
            and set(meta['posterior']['maximum_transform_errors'])==set(TRANSFORMS)
            and meta['posterior_predictive']['seed']==ppc_seed and meta['posterior_predictive']['replicates']==k
            and meta['posterior_predictive']['cells']==22
            and meta['standardization']['applicable'] is (target=='primary_sex'))
    for name,value in meta['posterior']['maximum_transform_errors'].items():
        expected=float(np.max(np.abs(np.asarray(posterior[name])-np.exp(posterior[TRANSFORMS[name]]))))
        require(type(value) in (int,float) and np.isfinite(value) and value>=0 and np.isclose(value,expected,rtol=1e-12,atol=1e-12))
    checks=meta['posterior_predictive']['conditional_mean_checks']
    require(isinstance(checks,list) and len(checks)==22)
    for i,cell in enumerate(checks):
        require(isinstance(cell,dict) and set(cell)=={'process','age_band','exposure_group','empty',
                'simulation_mean_error','conditional_simulation_mcse','within_four_conditional_mcse'}
                and (cell['process'],cell['age_band'],cell['exposure_group'])==tuple(ppc.iloc[i][['process','age_band','exposure_group']])
                and cell['empty'] is bool(not nonempty[i]))
        if not nonempty[i]:
            require(all(cell[n] is None for n in ('simulation_mean_error','conditional_simulation_mcse','within_four_conditional_mcse')))
        else:
            error=abs(float(replicas['statistics'][:,i].mean()-replicas['conditional_means'][:,i].mean()))
            mcse=float(np.sqrt(replicas['conditional_variances'][:,i].sum())/k)
            require(np.isclose(cell['simulation_mean_error'],error,rtol=1e-12,atol=1e-12)
                    and np.isclose(cell['conditional_simulation_mcse'],mcse,rtol=1e-12,atol=1e-12)
                    and cell['within_four_conditional_mcse'] is bool(error<=4*mcse))
    if target=='primary_sex':
        s=meta['standardization']
        require(set(s)=={'applicable','scenario','cohort_children','male_fraction','latent_nodes','age_nodes',
            'baseline_reconstruction_error','latent_standardization','sex_standardization','age_standardization',
            'other_post_effects','interpretation','convergence_nodes','convergence_limit',
            'maximum_draw_level_RR_quadrature_change_80_vs_120','band_quadrature_changes',
            'convergence_passed','direction_and_attenuation_checks_passed'}
            and s['scenario']==target and s['cohort_children']==validated['joint_children']
            and type(s['male_fraction']) in (int,float) and 0<=s['male_fraction']<=1
            and s['latent_nodes']==80 and s['age_nodes']==48 and s['convergence_nodes']==120
            and s['convergence_limit']==1e-4 and s['other_post_effects']==0
            and 0<=s['baseline_reconstruction_error']<1e-10
            and s['convergence_passed'] is True and s['direction_and_attenuation_checks_passed'] is True
            and set(s['band_quadrature_changes'])=={'1','2','3'})
        values=np.asarray(list(s['band_quadrature_changes'].values()),float)
        require(np.isfinite(values).all() and (values>=0).all() and (values<1e-4).all()
                and s['maximum_draw_level_RR_quadrature_change_80_vs_120']==float(values.max()))
        for key,text in {
            'latent_standardization':'N(0,1), same distribution under both states',
            'sex_standardization':'primary-cohort child-weighted sex distribution',
            'age_standardization':'uniform within each attained-age band; average probabilities before ratio',
            'interpretation':'scheduled current-wheeze probability ratio, not cumulative incidence or a causal effect'}.items():
            require(s[key]==text)
    else:
        require(set(meta['standardization'])=={'applicable','reason'}
                and meta['standardization']['reason']=='Selected probability ratios use the primary sex-adjusted model only.')


def validate_results(results, target, validated, posterior, ppc_seed):
    try:
        _validate_results(results,target,validated,posterior,ppc_seed)
    except (KeyError,AttributeError,TypeError,ValueError,IndexError):
        raise ValidationError('Wheeze result schema, cell inventory or aggregate values are inconsistent.') from None


def verify_aggregates(directory, posterior, stats, contract, target, validated, summary, ppc_seed):
    """Recompute every aggregate from complete, verified private posterior draws."""
    from .diagnostics import evaluate_diagnostics
    from .analysis.wheeze_results import summarize_wheeze
    essential = {n:stats[n] for n in ('diverging', 'energy', 'reached_max_treedepth') if n in stats}
    diagnostics = evaluate_diagnostics(posterior, essential, CRITERIA)
    results = summarize_wheeze(posterior, contract, target, ppc_seed=ppc_seed)
    validate_results(results,target,validated,posterior,ppc_seed)
    if (read_json(directory/'diagnostics.json', 'Wheeze diagnostics') != diagnostics
            or read_json(directory/'result_metadata.json', 'Wheeze result metadata') != results.metadata
            or summary.get('counts') != validated or summary.get('target') != target
            or summary.get('result_rows') != {name:len(getattr(results,name)) for name in TABLES} or type(summary.get('diagnostics_passed')) is not bool
            or summary['diagnostics_passed'] != diagnostics['passed']
            or summary.get('status') != ('diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed')):
        raise ValidationError('Wheeze aggregates disagree with verified posterior draws or diagnostic gates.')
    replicates = read_array_archive(directory/'ppc_replicates.npz')
    if (set(replicates) != set(results.ppc_replicates) or any(not np.array_equal(replicates[n], results.ppc_replicates[n], equal_nan=True) for n in replicates)):
        raise ValidationError('Wheeze predictive replicate aggregates disagree with the verified posterior and seed.')
    for name in TABLES:
        try:
            observed = pd.read_csv(directory/(name+'.csv'), float_precision='round_trip')
            pd.testing.assert_frame_equal(observed, getattr(results, name), check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        except (AssertionError, ValueError, TypeError) as exc:
            raise ValidationError('Saved wheeze tables disagree with recomputed posterior summaries.') from None


def sample_one(root,preparation_directory,target,settings,work,output,*,resume=False):
    """Publish a complete raw fit, then a separately checked aggregate result.

    Failed diagnostic results retain their raw private evidence, with an
    explicit failed status. They never create a validated completion marker.
    """
    root,prep,work,output=safe_roots(root,preparation_directory,work,output)
    assert_loaded_sources(root)
    descriptor=contract_descriptor(prep,target)
    fingerprint,identity=sampling_identity(root,descriptor,target,settings)
    work=work/target;output=output/target
    if work.is_symlink() or output.is_symlink() or (work/'compiler_cache').is_symlink():
        raise ValidationError('Wheeze target directories and compiler cache cannot be symbolic links.')
    work.mkdir(exist_ok=True,mode=0o700);output.mkdir(exist_ok=True,mode=0o700)
    final=work/fingerprint
    contract=load_artifacts(prep,tuple(fit_spec(target)['artifact_path']))
    validated=validate_contract(contract,target)
    configure_runtime(work/'compiler_cache')
    controls=identity['sampling']
    ppc_seed=controls['random_seed']+77
    expected=expected_posterior_shapes(contract,target,controls['chains'],controls['draws'])
    def unchanged():
        assert_loaded_sources(root)
        current=contract_descriptor(prep,target)
        if sampling_identity(root,current,target,settings)!=(fingerprint,identity):
            raise ValidationError('Model, inputs or sampling environment changed while the fit was running.')
    with preparation_locks(work,output):
        reused=final.exists()
        if reused:
            if not resume:raise ValidationError('An identical raw fit exists; use --resume to verify and revalidate it.')
            raw=read_json(final/'receipt.json','Raw fit receipt')
            if raw.get('identity')!=identity or raw.get('fingerprint')!=fingerprint or raw.get('status')!='sampled':
                raise ValidationError('Raw fit receipt identity is inconsistent.')
            verify_files(final,raw.get('files'),('posterior.npz','sample_stats.npz','sampling.json'))
            if set(raw['files']) != {'posterior.npz', 'sample_stats.npz', 'sampling.json'}:
                raise ValidationError('Wheeze raw fit has an unexpected file inventory.')
            posterior=read_array_archive(final/'posterior.npz')
            stats=read_array_archive(final/'sample_stats.npz')
            metadata=read_json(final/'sampling.json','Realized sampling metadata')
        else:
            stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=work))
            try:
                posterior,stats,metadata=sample_contract(contract,target,settings,work/'compiler_cache')
                validate_posterior_inventory(posterior,expected)
                validate_sampling_metadata(metadata,identity,posterior,stats)
                np.savez_compressed(stage/'posterior.npz',**posterior)
                np.savez_compressed(stage/'sample_stats.npz',**stats)
                write_json(stage/'sampling.json',metadata)
                raw={'schema_version':1,'status':'sampled','fingerprint':fingerprint,'identity':identity,
                     'preparation_fingerprint':descriptor['preparation_fingerprint'],
                     'completed_at':datetime.now(timezone.utc).isoformat(),
                     'files':{name:file_hash(stage/name) for name in ('posterior.npz','sample_stats.npz','sampling.json')}}
                write_json(stage/'receipt.json',raw)
                unchanged()
                os.replace(stage,final)
            finally:
                if stage.exists():shutil.rmtree(stage)
        validate_posterior_inventory(posterior,expected)
        validate_sampling_metadata(metadata,identity,posterior,stats)
        if file_hash(Path(__file__))!=_LOADED_RUNNER_HASH:
            raise ValidationError('The loaded model runner changed; restart with --resume to validate saved draws.')
        validation_fp,validation_id=validation_identity(root,fingerprint,CRITERIA)
        destination=output/validation_fp
        if destination.exists():
            receipt=read_json(destination/'receipt.json','Validation receipt')
            if receipt.get('identity')!=validation_id or receipt.get('fingerprint')!=validation_fp:
                raise ValidationError('Validation receipt identity is inconsistent.')
            verify_files(destination,receipt.get('files'),RESULT_FILES)
            if set(receipt['files']) != set(RESULT_FILES):
                raise ValidationError('Wheeze validation has an unexpected file inventory.')
            summary=read_json(destination/'summary.json','Wheeze summary')
            if receipt.get('summary')!=summary:
                raise ValidationError('Validation receipt and verified wheeze summary disagree.')
            diagnostic=read_json(destination/'diagnostics.json','Wheeze diagnostics')
            if (type(diagnostic.get('passed')) is not bool
                    or diagnostic['passed']!=summary.get('diagnostics_passed')
                    or diagnostic.get('status')!=('passed' if diagnostic['passed'] else 'failed')
                    or summary.get('status')!=('diagnostics_passed' if diagnostic['passed'] else 'diagnostics_failed')):
                raise ValidationError('Verified diagnostics and wheeze completion summary disagree.')
            if receipt.get('raw_receipt_sha256')!=file_hash(final/'receipt.json'):
                raise ValidationError('Validation receipt is not bound to the verified raw fit.')
            if summary.get('fit_fingerprint') != fingerprint or summary.get('validation_fingerprint') != validation_fp:
                raise ValidationError('Wheeze summary fingerprints disagree with their receipts.')
            verify_aggregates(destination, posterior, stats, contract, target, validated, summary, ppc_seed)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during cached validation.')
            return {**summary,'raw_fit_reused':True,'validation_reused':True}
        from .diagnostics import evaluate_diagnostics
        from .analysis.wheeze_results import summarize_wheeze
        essential={name:stats[name] for name in ('diverging','energy','reached_max_treedepth') if name in stats}
        diagnostics=evaluate_diagnostics(posterior,essential,CRITERIA)
        results=summarize_wheeze(posterior,contract,target,ppc_seed=ppc_seed)
        validate_results(results,target,validated,posterior,ppc_seed)
        summary={'target':target,'fit_fingerprint':fingerprint,'validation_fingerprint':validation_fp,
                 'diagnostics_passed':bool(diagnostics['passed']),
                 'status':'diagnostics_passed' if diagnostics['passed'] else 'diagnostics_failed',
                 'counts':validated,'result_rows':{name:len(getattr(results,name)) for name in TABLES},
                 'scientific_reference_comparison':'requires separate independent validation'}
        stage=Path(tempfile.mkdtemp(prefix='.staging-',dir=output))
        try:
            write_json(stage/'diagnostics.json',diagnostics)
            write_json(stage/'summary.json',summary)
            write_json(stage/'result_metadata.json',results.metadata)
            np.savez_compressed(stage/'ppc_replicates.npz',**results.ppc_replicates)
            for name in TABLES:
                frame=getattr(results,name)
                frame.to_csv(stage/(name+'.csv'),index=False,float_format='%.17g')
            receipt={'schema_version':1,'fingerprint':validation_fp,'identity':validation_id,'summary':summary,
                     'raw_receipt_sha256':file_hash(final/'receipt.json'),
                     'files':{p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}}
            write_json(stage/'receipt.json',receipt)
            verify_aggregates(stage, posterior, stats, contract, target, validated, summary, ppc_seed)
            unchanged()
            if validation_identity(root,fingerprint,CRITERIA)!=(validation_fp,validation_id):
                raise ValidationError('Diagnostic or reporting code changed during validation.')
            os.replace(stage,destination)
        finally:
            if stage.exists():shutil.rmtree(stage)
        return {**summary,'raw_fit_reused':reused,'validation_reused':False}


def run_wheeze(root, config_file, analysis_file, primary_sampling_file,
               mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file,
               *, jobs=1, resume=False):
    """Run cumulative M01-M10, verifying every preceding selected analysis."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from .wheeze_sampling import validate_sampling
    from .pretb_execution import run_pretb
    if type(jobs) is not int or not 1 <= jobs <= 3:
        raise ValidationError('Wheeze model concurrency must be between one and three jobs.')
    assert_loaded_sources(root)
    paths = [Path(p) for p in (config_file, analysis_file, primary_sampling_file,
                               mechanics_sampling_file, supporting_sampling_file, wheeze_sampling_file)]
    hashes = [file_hash(p) for p in paths]
    settings = validate_sampling(read_json(paths[-1], 'Wheeze sampling configuration'))
    configuration = read_json(config_file, 'Local configuration')
    _, directories = resolve_paths(configuration, config_file)
    for directory in directories.values():
        if directory == paths[-1].resolve() or directory in paths[-1].resolve().parents:
            raise ValidationError('Generated directories cannot contain the wheeze sampling configuration.')
    prior = run_pretb(root, *paths[:-1], jobs=jobs, resume=resume)
    expected = {'through':'M09', 'selected_models':17, 'primary_models_verified':12,
                'binary_models_verified':28, 'trajectory_analyses_verified':4,
                'mechanics_models_verified':6, 'shared_factor_rows_verified':6,
                'supporting_joint_models_verified':7, 'supporting_deterministic_fits_verified':10,
                'status':'accepted'}
    if (not isinstance(prior, dict) or prior.get('accepted') is not True
            or any(prior.get(k) != v for k,v in expected.items())):
        raise ValidationError('Cumulative M10 requires all selected preceding analyses to pass.')
    prep = directories['work']/'preparation_runs'/prior['preparation_fingerprint']
    work, output = directories['work']/'wheeze_fits', directories['outputs']/'wheeze_models'
    selected = list(TARGETS)
    if jobs == 1:
        results = [sample_one(root, prep, target, settings, work, output, resume=resume) for target in selected]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs,len(selected)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(sample_one, root, prep, target, settings, work, output, resume=resume)
                       for target in selected]
            results = [done.result() for done in as_completed(pending)]
    if (len(results) != len(selected) or any(not isinstance(result,dict)
            or not isinstance(result.get('target'),str) for result in results)
            or sorted(result.get('target','') for result in results) != sorted(selected)):
        raise ValidationError('Wheeze execution requires exactly the two selected joint fits.')
    for result in results:
        accepted = result.get('diagnostics_passed')
        rows = result.get('result_rows',{})
        if (type(accepted) is not bool
                or result.get('status') != ('diagnostics_passed' if accepted else 'diagnostics_failed')
                or not isinstance(rows,dict)
                or set(rows) != set(TABLES)
                or any(type(v) is not int for v in rows.values())
                or rows.get('effects') != 6 or rows.get('contrasts') != 6
                or rows.get('global_tests') != 2
                or rows.get('support_counts') != 22
                or rows.get('posterior_predictive_cells') != rows['support_counts']
                or rows.get('standardized_rr') != (3 if result['target']=='primary_sex' else 0)):
            raise ValidationError('Wheeze completion contradicts diagnostic status or output inventory.')
    results.sort(key=lambda result:selected.index(result['target']))
    verified = run_preparation(root, config_file, analysis_file, 'M03', resume=True)
    if (verified['fingerprint'] != prior['preparation_fingerprint']
            or [file_hash(p) for p in paths] != hashes):
        raise ValidationError('Preparation or sampling dependencies changed during wheeze execution.')
    assert_loaded_sources(root)
    passed = all(result['diagnostics_passed'] for result in results)
    summary = {'through':'M10', 'preparation_fingerprint':prior['preparation_fingerprint'],
               'selected_models':len(results), 'wheeze_models':results,
               'primary_models_verified':12, 'binary_models_verified':28,
               'trajectory_analyses_verified':4, 'mechanics_models_verified':6,
               'shared_factor_rows_verified':6, 'supporting_joint_models_verified':7,
               'supporting_deterministic_fits_verified':10, 'pretb_models_verified':17,
               'accepted':passed, 'status':'accepted' if passed else 'diagnostics_failed',
               'inherited_limitations':['G08: pre-TB source covariance limitation remains unresolved',
                                       'Four upstream antenatal flow counts remain unavailable'],
               'scientific_reference_comparison':'requires separate independent validation'}
    if passed:
        write_pointer(directories['work']/'wheeze_current.json', summary)
    return summary
