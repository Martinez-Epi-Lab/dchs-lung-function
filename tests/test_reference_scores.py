"""Fabricated physiological inputs verify reference equations and boundaries."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dchs_lung_function.reference_scores import (add_gli_global_zscores,
    add_oscillometry_zscores, add_reference_scores, pft_male, OSC_EQUATIONS)


def inputs():
    # Exactly four years under the analysis day-count convention.
    f=pd.DataFrame({'sex':[0,1], 'height':[110.,108.],
                    'child_dob':pd.to_datetime(['2020-01-01']*2),
                    'date_of_test':pd.to_datetime(['2024-01-01']*2),
                    'date_of_test_ilf2':pd.to_datetime(['2024-01-01']*2),
                    'fev1':[1.0,1.1], 'fvc':[1.2,1.3], 'fev1_fvc':[.833,.846],
                    'fev1_post':[1.1,np.nan], 'fvc_post':[1.3,1.4], 'fev1_fvc_post':[.846,.85],
                    'mmef':[.9,1.0], 'mmef_post':[1.,np.nan]})
    for eq in OSC_EQUATIONS:f[eq.column]=1.
    lookup=pd.DataFrame([{'agebound':4.,'gender':g,'f':measure,'Mspline':0.,'Sspline':0.} for g in [1,2] for measure in ['FEV1','FVC','FEV1FVC']])
    return f,lookup


class ReferenceScoresTests(unittest.TestCase):
    def test_female_fev1_equation_independent_of_males_in_frame(self):
        f,lookup=inputs();f=f.iloc[[1]].reset_index(drop=True)
        actual=add_gli_global_zscores(f,lookup)
        median=np.exp(-10.901689+2.385928*np.log(108.)-.076386*np.log(4.))
        dispersion=np.exp(-2.364047+.129402*np.log(4.))
        expected=((1.1/median)**1.21388-1)/(1.21388*dispersion)
        self.assertAlmostEqual(actual.fev1_gli_z.iloc[0],expected,places=12)
        self.assertAlmostEqual(actual.fev1_gli_pred.iloc[0],median,places=12)

    def test_lookup_unique_coverage_and_source_nonmutation(self):
        f,lookup=inputs();before=f.copy(deep=True)
        with self.assertRaisesRegex(ValueError,'duplicate'):add_gli_global_zscores(f,pd.concat([lookup,lookup]))
        with self.assertRaisesRegex(ValueError,'cover'):add_gli_global_zscores(f,lookup.loc[lookup.gender.eq(1)])
        add_gli_global_zscores(f,lookup);pd.testing.assert_frame_equal(f,before)

    def test_nonpositive_and_out_of_reference_values_do_not_get_scores(self):
        f,lookup=inputs();f.loc[0,'fev1']=0;f.loc[1,'date_of_test']=pd.Timestamp('2022-01-01')
        actual=add_gli_global_zscores(f,lookup)
        self.assertTrue(actual.fev1_gli_z.isna().all())
        self.assertTrue(actual.fev1_gli_pred.isna().all())

    def test_duplicate_index_labels_do_not_duplicate_or_misalign_results(self):
        f,lookup=inputs();f.index=[7,7]
        actual=add_gli_global_zscores(f,lookup)
        self.assertEqual(actual.index.tolist(),[7,7]);self.assertEqual(len(actual),2)
        self.assertTrue(actual.fev1_gli_z.notna().all())

    def test_inverse_reactance_and_compliance_unit_conversion(self):
        f,_=inputs();f['reactance_6hz']=[-3.,-2.];f['compliance']=[.02,.03]
        actual=add_oscillometry_zscores(f)
        np.testing.assert_allclose(actual.x6_sa_z,([-3.,-2.]-(3.46-727/f.height))/1.168)
        np.testing.assert_allclose(actual.compliance_sa_z,(np.log(f.compliance*1000/.980665)-(.099+.0168*f.height))/.295)

    def test_paired_missingness_and_sex_validation(self):
        f,lookup=inputs();actual=add_reference_scores(f,lookup)
        self.assertTrue(pd.isna(actual.fev1_pre_paired.iloc[1]))
        self.assertTrue(pd.isna(actual.mmef_pre_paired.iloc[1]))
        self.assertTrue(pd.isna(actual.mmef_post_paired.iloc[1]))
        self.assertTrue(pd.notna(actual.fvc_pre_paired.iloc[1]))
        with self.assertRaisesRegex(ValueError,'sex'):pft_male(pd.Series([2]))
        with self.assertRaisesRegex(ValueError,'label mapping'):pft_male(pd.Series([2]),{0:'Male',1:'Female'})
        self.assertEqual(pft_male(pd.Series([2,1]),{2:'Male',1:'Female'}).tolist(),[1.,0.])


if __name__=='__main__':unittest.main()
