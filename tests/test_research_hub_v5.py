import io
import json
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assaylens.research_hub import REGISTRY, MODEL_PRESETS, normalize_tdc, load_and_normalize
from assaylens.studio_api import app


@pytest.fixture
def dti_rows():
    seq_a = "M" + "ACDEFGHIKLMNPQRSTVWY" * 3
    seq_b = "M" + "VWYTSRQPNMLKIHGFEDCA" * 3
    return pd.DataFrame({
        "Drug": ["CCO", "CCN", "CC(=O)O", "CCC", "c1ccccc1", "CCCl"],
        "Target": [seq_a, seq_a, seq_b, seq_b, seq_a, seq_b],
        "Target_ID": ["P1", "P1", "P2", "P2", "P1", "P2"],
        "Y": [120, 300, 5, 33, 100, 6],
    })


def test_registry_groups_and_restricted_licensing():
    assert all(x in MODEL_PRESETS for x in ('dti','ddi','ppi'))
    assert {v.kind for v in REGISTRY.values()} == {'dti','ddi','ppi'}
    assert REGISTRY['ddi_drugbank'].availability == 'licensed_upload'
    assert REGISTRY['ddi_twosides'].availability == 'raw_only'
    assert len(MODEL_PRESETS['ppi']['classification']) >= 4


def test_dti_preserves_native_y_and_sequences(dti_rows):
    clean, manifest = normalize_tdc(REGISTRY['dti_bindingdb_ki'], dti_rows, 100)
    assert set(['smiles','target_sequence','target_id','response']).issubset(clean)
    assert len(clean) == 6
    assert clean.response.tolist() == [120,300,5,33,100,6]
    assert manifest['original_rows'] == 6
    assert 'not fabricated' in ';'.join(manifest['notes'])
    assert (clean.assay_id == 'TDC_NOT_PROVIDED').all()


def test_kiba_retains_distinct_label(dti_rows):
    clean, _ = normalize_tdc(REGISTRY['dti_kiba'], dti_rows, 100)
    assert set(clean.endpoint) == {'KIBA_score'}


def test_ppi_positives_not_assumed_binary_ready():
    seq = 'ACDEFGHIKLMNPQRSTVWY' * 3
    raw = pd.DataFrame({'Protein1':[seq,seq], 'Protein2':[seq[::-1], seq+'A'], 'Y':[1,1]})
    clean, audit = normalize_tdc(REGISTRY['ppi_huri'], raw, 100)
    assert len(clean) == 2
    assert audit['trainable_as_is'] is False
    assert (clean.label == 1).all()


def test_ppi_reference_controls_explicit_flag():
    seq = 'ACDEFGHIKLMNPQRSTVWY'*3
    raw = pd.DataFrame({'Protein1':[seq,seq], 'Protein2':[seq[::-1],seq[::-1]], 'Y':[0,1]})
    clean, audit = normalize_tdc(REGISTRY['ppi_huri'],raw,100,reference_negatives=True)
    assert audit['trainable_as_is']
    assert 'generated reference negatives' in ';'.join(audit['notes'])


def test_ddi_multilabel_not_converted_to_binary():
    raw = pd.DataFrame({'Drug1':['CCO','CCC'], 'Drug2':['CCN','CCCl'], 'Y':[17,42]})
    clean,audit=normalize_tdc(REGISTRY['ddi_twosides'],raw,100)
    assert 'event_label' in clean
    assert 'label' not in clean
    assert audit['trainable_as_is'] is False


def test_disallow_license_gated_auto_download():
    with pytest.raises(ValueError,match='license'):
        load_and_normalize('ddi_drugbank',100,loader=lambda *a:None)


def test_api_registry_endpoint():
    with TestClient(app) as c:
        r=c.get('/api/v5/catalog')
        assert r.status_code==200
        d=r.json()
        assert any(x['key']=='ppi_huri' for x in d['resources'])
        assert any(x['key']=='ddi_drugbank' for x in d['resources'])
        bad=c.post('/api/v5/datasets/download',json={'resource_key':'ddi_drugbank','max_rows':100})
        assert bad.status_code==422


def test_offline_download_job(monkeypatch, tmp_path):
    from assaylens import research_hub_api as h
    raw = pd.DataFrame({
        'Protein1':['A'*40,'B'*40, 'C'*40],
        'Protein2':['B'*40,'C'*40,'A'*40],
        'Y':[1,0,1],
    })
    monkeypatch.setattr(h,'load_and_normalize',lambda key,limit,sampling_seed,reference_negatives:
        normalize_tdc(REGISTRY[key],raw,limit,sampling_seed,reference_negatives))
    monkeypatch.setattr(h,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(h,'PAIR_SAVE',lambda *args, **kwargs:{'id':'mock','kind':'ppi'})
    with TestClient(app) as c:
        res=c.post('/api/v5/datasets/download',json={'resource_key':'ppi_huri','max_rows':100,'ppi_reference_negatives':True})
        assert res.status_code==202
        job_id=res.json()['job_id']
        status={}
        for _ in range(80):
            status=c.get(f'/api/v5/datasets/jobs/{job_id}').json()
            if status['state'] in ('complete','error'):
                break
            time.sleep(.05)
        assert status['state']=='complete', status
        assert status['manifest']['import_result']['id']=='mock'
        assert c.get(f'/api/v5/datasets/jobs/{job_id}/csv').status_code==200
        assert c.get(f'/api/v5/datasets/jobs/{job_id}/manifest').status_code==200
