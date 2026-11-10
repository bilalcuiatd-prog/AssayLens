import io
import json
import math
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assaylens import storage, studio_api
from assaylens.catalog import BY_GENE, TARGETS
from assaylens.experiments import conformal, make_folds, run_experiment
from assaylens.quality import audit_dataset
from assaylens.sources import check_chembl_record, parse_bindingdb


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(studio_api, 'ROOT', tmp_path)
    with TestClient(studio_api.app) as c:
        yield c


def test_catalog_and_dashboard(client):
    catalog = client.get('/api/catalog')
    assert catalog.status_code == 200
    assert len(catalog.json()['targets']) == 46
    assert {'ACHE', 'MAOB', 'LRRK2', 'BACE1', 'PDE5A'} <= set(BY_GENE)
    assert len({r['gene'] for r in TARGETS}) == len(TARGETS)
    assert client.get('/').status_code == 200
    assert 'Model laboratory' in client.get('/').text
    assert client.get('/assets/app.js').status_code == 200
    assert client.get('/api/health').json()['status'] == 'ok'


def test_chronological_measurement_and_censoring():
    assay = {'target_chembl_id': 'CHEMBL205', 'confidence_score': 9, 'relationship_type': 'D', 'assay_type': 'B', 'bao_label': 'single protein format'}
    row = {'target_chembl_id': 'CHEMBL205', 'target_organism': 'Homo sapiens', 'standard_type': 'Ki', 'standard_relation': '=', 'standard_units': 'nM', 'standard_value': '10', 'data_validity_comment': None, 'potential_duplicate': False, 'standard_flag': True, 'document_chembl_id': 'CHEMBL_DOC', 'canonical_smiles': 'CCO', 'assay_chembl_id': 'CHEMBL_A', 'pchembl_value': '8.0'}
    observation, rejection = check_chembl_record(row, assay, 'CHEMBL205', 'Ki')
    assert rejection is None and math.isclose(observation['response'], 8.0)
    row['standard_relation'] = '<'
    _, rejection = check_chembl_record(row, assay, 'CHEMBL205', 'Ki')
    assert rejection == 'endpoint_or_censoring'


def test_bindingdb_payload_cannot_manufacture_assay_ids():
    payload = {'getLindsByUniprotsResponse': {'affinities': [{'smile': 'CCO', 'affinity_type': 'Ki', 'affinity': '10.2', 'monomerid': '1234'}]}}
    assert len(parse_bindingdb(payload)) == 1
    assert parse_bindingdb({'getLindsByUniprotsResponse': {}}) == []


def test_group_folds_and_conformal():
    df = audit_dataset(pd.read_csv('examples/synthetic_assay_demo.csv')).curated
    for seed in [17, 29]:
        seen = []
        for _, train, test, groups in make_folds(df, 'scaffold', 5, seed):
            assert not set(groups[train]) & set(groups[test])
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(len(df)))
    assert conformal([2, 1, 5, 4, 3, 0, 0, 1, 2, 4, 7, 2], [0]*12, 0.1) == 7
    assert conformal([1, 2], [0, 0], 0.1) is None


def test_import_and_research_run(client):
    content = open('examples/synthetic_assay_demo.csv', 'rb').read()
    response = client.post('/api/datasets', files={'file': ('study.csv', io.BytesIO(content), 'text/csv')})
    assert response.status_code == 201, response.text
    dataset = response.json()
    ident = dataset['id']
    assert client.get('/api/datasets').json()[0]['id'] == ident
    assert client.get(f'/api/datasets/{ident}/csv').status_code == 200
    assert client.get(f'/api/datasets/{ident}/qc.csv').status_code == 200
    assert client.get(f'/api/datasets/{ident}/source-manifest').status_code == 404
    req = {'dataset_id': ident, 'strategy': 'scaffold', 'folds': 3, 'seeds': [17, 29], 'models': ['extra_trees', 'ridge'], 'variants': ['chemistry', 'fingerprint_only'], 'n_estimators': 24, 'fp_bits': 256}
    response = client.post('/api/experiments', json=req)
    assert response.status_code == 202, response.text
    job_id = response.json()['job_id']
    for i in range(100):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] in ('complete', 'error'):
            break
        time.sleep(.1)
    assert job['state'] == 'complete', job
    runid = job['run_id']
    summary = client.get(f'/api/experiments/{runid}').json()
    assert summary['n_fits'] == 24 and len(summary['summary']) == 4
    assert client.get(f'/api/experiments/{runid}/bundle').status_code == 200
    assert client.get(f'/api/experiments/{runid}/figure').status_code == 200
    assert client.get('/api/experiments').json()[0]['id'] == runid
    assert client.get(f'/api/datasets/{ident}/audit').status_code == 200


def test_validation_rejects_fabricated_assay_groups(client):
    raw = pd.read_csv('examples/synthetic_assay_demo.csv')
    raw['assay_id'] = 'UNKNOWN'
    response = client.post('/api/datasets', files={'file': ('unknown.csv', io.BytesIO(raw.to_csv(index=False).encode()), 'text/csv')})
    assert response.status_code == 201
    dataset_id = response.json()['id']
    resp = client.post('/api/experiments', json={'dataset_id':dataset_id, 'strategy':'assay', 'models':['ridge'], 'variants':['chemistry'], 'folds':5, 'seeds':[17]})
    assert resp.status_code == 422 and 'genuine assay' in resp.text


def test_provider_batch_rejects_unknown_target(client):
    resp = client.post('/api/downloads', json={'provider':'chembl','genes':['NOT_A_GENE'], 'endpoints':['Ki']})
    assert resp.status_code == 422
    resp = client.get('/api/pubchem/NOT_A_GENE/assays')
    assert resp.status_code == 404


def test_ch_embl_export_with_mocked_api(tmp_path, monkeypatch):
    from assaylens import sources
    source = pd.read_csv('examples/synthetic_assay_demo.csv')
    unique = source.smiles.dropna().astype(str).unique().tolist()
    from rdkit import Chem
    unique = [s for s in unique if Chem.MolFromSmiles(s) is not None][:55]
    activities = []
    for i, smi in enumerate(unique):
        activities.append({'activity_id':i+1,'target_chembl_id':'CHEMBL205','target_organism':'Homo sapiens','standard_type':'Ki','standard_relation':'=','standard_units':'nM','standard_value':str(10+i),'data_validity_comment':None,'standard_flag':True,'potential_duplicate':False,'document_chembl_id':'DOC1','canonical_smiles':smi,'assay_chembl_id':f'ASSAY{i%8}','pchembl_value':9-math.log10(10+i)})
    assays = [{'assay_chembl_id':f'ASSAY{i}', 'target_chembl_id':'CHEMBL205','confidence_score':9,'relationship_type':'D','assay_type':'B','bao_label':'biochemical'} for i in range(8)]
    monkeypatch.setattr(sources,'session',lambda: object())
    monkeypatch.setattr(sources,'chembl_target',lambda s,g:{'target_chembl_id':'CHEMBL205','pref_name':'Carbonic anhydrase II'})
    monkeypatch.setattr(sources,'fetch_json',lambda s,url,params=None:{'chembl_db_version':'offline-test'})
    monkeypatch.setattr(sources,'chembl_pages',lambda s,resource,params,collection,max_raw=0:iter(activities if collection=='activities' else assays))
    csv, manifest = sources.download_chembl('CA2','Ki',tmp_path/'CA2',lambda msg: None)
    assert csv.is_file() and manifest['filtered_count']==len(unique)
    output=pd.read_csv(csv)
    assert list(output.columns)==sources.COLUMNS
    assert output.assay_id.nunique()==8
    assert manifest['provider']=='ChEMBL' and manifest['target_id']=='CHEMBL205'
    assert (tmp_path/'CA2'/'raw_activities.jsonl.gz').is_file()
    assert (tmp_path/'CA2'/'excluded.csv').is_file()


def test_bindingdb_export_preserves_missing_assay_id(tmp_path,monkeypatch):
    from assaylens import sources
    from rdkit import Chem
    input_rows=pd.read_csv('examples/synthetic_assay_demo.csv').smiles.dropna().unique().tolist()
    smiles=[s for s in input_rows if Chem.MolFromSmiles(s) is not None][:45]
    payload={'getLindsByUniprotsResponse':{'affinities':[{'monomerid':i+1,'smile':s,'affinity_type':'Ki','affinity':str(10+i),'pmid':'12345','query':'human enzyme'} for i,s in enumerate(smiles)]}}
    monkeypatch.setattr(sources,'session',lambda:object())
    monkeypatch.setattr(sources,'fetch_json',lambda s,url,params=None:payload)
    csv, manifest=sources.download_bindingdb('ACHE','Ki',tmp_path/'ACHE',lambda m:None)
    output=pd.read_csv(csv)
    assert output.assay_id.eq('UNKNOWN').all()
    assert output.target_id.eq('P22303').all()
    assert manifest['cutoff_nm']==100000
    assert manifest['filtered_count']==len(smiles)
    assert 'affinity' in manifest['scientific_limitations'][0].lower()


def test_pubchem_api_contract_mocked(monkeypatch):
    from assaylens import sources
    monkeypatch.setattr(sources,'session',lambda:object())
    monkeypatch.setattr(sources,'fetch_json',lambda s,url,params=None:{'IdentifierList':{'AID':[10,11,12]}})
    result=sources.pubchem_assays('P22303')
    assert [row['aid'] for row in result]==[10,11,12]
    assert all('/assay/aid/' in x['download_url'] and x['download_url'].endswith('/CSV') for x in result)
