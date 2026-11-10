from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from assaylens.quality import audit_dataset
from assaylens.modeling import partition_groups, conformal_radius, train_evaluate

DEMO=Path(__file__).resolve().parents[1]/'examples'/'synthetic_assay_demo.csv'


def demo():
    return audit_dataset(pd.read_csv(DEMO)).curated


@pytest.mark.parametrize('split', ['scaffold','assay'])
def test_group_disjoint(split):
    df=demo()
    a,b,c=partition_groups(df,split,42)
    groups=df['scaffold' if split=='scaffold' else 'assay_id']
    assert not(set(groups.iloc[a]) & set(groups.iloc[b]))
    assert not(set(groups.iloc[a]) & set(groups.iloc[c]))
    assert not(set(groups.iloc[b]) & set(groups.iloc[c]))


def test_conformal_order_statistic():
    y=np.array([1,2,3,4,5],dtype=float)
    assert conformal_radius(y,np.zeros(5),alpha=0.2)==5.0
    assert conformal_radius(y,np.zeros(5),alpha=0.5)==3.0


@pytest.mark.parametrize('split', ['scaffold','assay'])
def test_train_evaluate_valid_metrics(split):
    df=demo()
    run=train_evaluate(df,split_type=split,seed=43,trees=16,alpha=0.1)
    assert len(run.summary['results'])==3
    assert run.summary['test_n']==len(run.residuals)
    assert run.residuals['similarity_to_training'].between(0,1).all()
    for row in run.summary['results'].values():
        assert row['rmse']>=0
        assert 0<=row['coverage']<=1
        assert row['mean_interval_width']>=0


def test_unknown_assay_cannot_test_assay_split():
    df=demo();df.loc[0,'assay_id']='UNKNOWN'
    with pytest.raises(ValueError, match='requires an assay_id'):
        partition_groups(df,'assay',42)


def test_conformal_not_falsely_claim_finite_coverage_with_tiny_calibration():
    with pytest.raises(ValueError,match='insufficient'):
        conformal_radius(np.arange(5.),np.zeros(5),alpha=0.05)
