import pandas as pd
import pytest
from assaylens.quality import audit_dataset
from assaylens.chemistry import parse_molecule


def test_invalid_and_exact_duplicates():
    df = pd.DataFrame({
        'smiles':['CCO','CCO','NOT_SMILES','CCN','CCN'],
        'response':[3.0,3.0,2.0,4.0,float('nan')],
        'assay_id':['A','A','A','B','B'],
        'endpoint':['TEST']*5,'target_id':['T']*5,
    })
    a = audit_dataset(df)
    assert a.summary['reason_counts']['invalid_smiles'] == 1
    assert a.summary['reason_counts']['exact_duplicate'] == 1
    assert a.summary['reason_counts']['invalid_response'] == 1
    assert len(a.curated) == 2


def test_conflicting_replicates_excluded():
    df = pd.DataFrame({'smiles':['CCO','CCO','CCN'], 'response':[3.0,5.0,4.0],
                       'assay_id':['A','A','B'], 'endpoint':['T']*3,'target_id':['P']*3})
    a = audit_dataset(df)
    assert a.summary['within_assay_conflict_pairs_excluded'] == 1
    assert len(a.curated) == 1


def test_cross_endpoint_rejected():
    df=pd.DataFrame({'smiles':['CCO','CCN'],'response':[3,4], 'endpoint':['IC50','EC50']})
    with pytest.raises(ValueError, match='Multiple endpoints'):
        audit_dataset(df)


def test_structure_parent_and_scaffold():
    c,p,g,changed = parse_molecule('CCO.Cl')
    assert changed is True
    assert p=='CCO'
    assert g.startswith('ACYCLIC:')


def test_cannot_hide_invalid_response():
    df=pd.DataFrame({'smiles':['CCO','CCN'],'response':['wrong','4']})
    a=audit_dataset(df)
    assert a.summary['reason_counts']['invalid_response']==1


def test_censored_measurement_removed():
    df = pd.DataFrame({
      'smiles':['CCO','CCN','CCC'], 'response':[3.0,4.0,5.0],
      'standard_relation':['=','>','='], 'data_validity_comment':['','','Possibly invalid']
    })
    audit=audit_dataset(df)
    assert audit.summary['reason_counts']['censored_or_inexact_measurement']==1
    assert audit.summary['reason_counts']['data_validity_warning']==1
    assert len(audit.curated)==1


def test_missing_target_rows_not_pooled():
    df=pd.DataFrame({'smiles':['CCO','CCN'], 'response':[3.0,4.0],
                     'target_id':['P',''], 'endpoint':['X','X']})
    a=audit_dataset(df)
    assert a.summary['reason_counts']['missing_target']==1
    assert len(a.curated)==1
