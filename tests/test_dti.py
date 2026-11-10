from pathlib import Path
import json
import tempfile
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assaylens import dti, studio_api

PROJECT = Path(__file__).resolve().parents[1]


def prepared():
    df = pd.read_csv(PROJECT / 'examples' / 'synthetic_multi_target_dti.csv')
    return dti.curate(df)[0]


@pytest.mark.parametrize('protocol', dti.STRATEGIES)
def test_pairwise_entity_disjoint_splits(protocol):
    df = prepared()
    found = 0
    for train, test, excluded in dti.generate_splits(df, protocol, 5, 17):
        molecules, targets = dti._check_split(df, protocol, train, test)
        assert set(train).isdisjoint(test)
        assert set(excluded).isdisjoint(set(train) | set(test))
        assert len(train) + len(test) + len(excluded) == len(df)
        if protocol in ('cold_drug', 'cold_both'):
            assert molecules == 0
        if protocol in ('cold_target', 'cold_both'):
            assert targets == 0
        if protocol == 'warm':
            assert len(excluded) == 0
        if protocol == 'cold_both':
            assert len(excluded) > 0
        found += 1
    assert found == 5


def test_reject_single_target_for_cold_target():
    df = prepared()
    df = df[df.target_id == df.target_id.iloc[0]].copy()
    with pytest.raises(ValueError, match='protein'):
        dti.validate({'task': 'regression', 'strategy': 'cold_target',
                      'models': ['extra_trees'], 'variants': ['joint'],
                      'folds': 5, 'seeds': [17]}, df)


def test_reject_conflicting_sequence_for_one_protein():
    df = pd.read_csv(PROJECT / 'examples' / 'synthetic_multi_target_dti.csv')
    df.at[1, 'target_id'] = df.target_id.iloc[0]
    with pytest.raises(ValueError, match='Inconsistent sequences'):
        dti.curate(df)


def test_reject_censored_inexact_and_conflicted_pairs():
    frame = pd.read_csv(PROJECT / 'examples' / 'synthetic_multi_target_dti.csv')
    frame.at[0, 'standard_relation'] = '>'
    duplicate = frame.iloc[2].copy()
    duplicate['response'] = float(duplicate['response']) + 1.3
    frame = pd.concat([frame, pd.DataFrame([duplicate])], ignore_index=True)
    curated, exclusions, audit = dti.curate(frame)
    assert (exclusions.reason == 'censored_measurement').any()
    assert (exclusions.reason == 'discordant_pair_measurements').any()
    assert audit['discordant_pairs_excluded'] == 1
    assert len(curated) == 690


def test_dti_runs_regression_and_classification(tmp_path):
    df = prepared()
    for task, protocol in [('regression', 'cold_both'), ('classification', 'cold_drug')]:
        config = {'task': task, 'strategy': protocol,
                  'models': ['extra_trees'], 'variants': ['joint', 'drug_only'],
                  'folds': 5, 'seeds': [17], 'n_estimators': 16,
                  'activity_threshold': 5.4, 'fp_bits': 128}
        outcome = dti.run(df, config, tmp_path / task)
        assert outcome['n_fits'] == 10
        assert (tmp_path / task / 'fold_metrics.csv').exists()
        assert (tmp_path / task / 'split_assignments.csv').exists()
        fold = pd.read_csv(tmp_path / task / 'fold_metrics.csv')
        if protocol == 'cold_both':
            assert (fold.drug_overlap_n == 0).all()
            assert (fold.target_overlap_n == 0).all()
            assert (fold.excluded_n > 0).all()
        pred = pd.read_csv(tmp_path / task / 'test_predictions.csv')
        assert len(pred) > 0


def test_dti_web_api_and_async_job(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_api, 'ROOT', tmp_path)
    client = TestClient(studio_api.app)
    response = client.get('/api/v3/dti/catalog')
    assert response.status_code == 200
    assert len(response.json()['protocols']) == 4
    sample = (PROJECT / 'examples' / 'synthetic_multi_target_dti.csv').read_bytes()
    response = client.post('/api/v3/dti/datasets', files={'file': ('dti.csv', sample, 'text/csv')})
    assert response.status_code == 201, response.text
    identifier = response.json()['id']
    assert response.json()['unique_targets'] == 15
    assert client.get('/api/v3/dti/datasets/' + identifier + '/csv').status_code == 200
    response = client.post('/api/v3/dti/experiments', json={
        'dataset_id': identifier, 'task': 'regression', 'strategy': 'warm',
        'folds': 2, 'seeds': [17], 'models': ['extra_trees'],
        'variants': ['joint'], 'n_estimators': 16, 'fp_bits': 128})
    assert response.status_code == 202, response.text
    job_id = response.json()['job_id']
    for _ in range(70):
        state = client.get('/api/jobs/' + job_id).json()
        if state['state'] in ('complete', 'error'):
            break
        time.sleep(.15)
    assert state['state'] == 'complete', state
    runid = state['run_id']
    payload = client.get('/api/v3/dti/experiments/'+runid+'/bundle')
    assert payload.status_code == 200
    assert payload.content[:2] == b'PK'
    details = studio_api.dti_dir('dti_runs', runid)
    assert json.loads((details / 'manifest.json').read_text())['protocol'] == 'warm'
