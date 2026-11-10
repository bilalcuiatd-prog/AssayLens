from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from assaylens.api import app

CLIENT=TestClient(app)
DEMO=Path(__file__).resolve().parents[1]/'examples'/'synthetic_assay_demo.csv'


def test_health():
    r=CLIENT.get('/api/health')
    assert r.status_code==200
    assert r.json()['status']=='ok'


def test_upload_validation():
    r=CLIENT.post('/api/datasets',files={'file':('data.txt',b'hello','text/plain')})
    assert r.status_code==400
    r=CLIENT.post('/api/datasets',files={'file':('bad.csv',b'smiles,other\nCCO,3\n','text/csv')})
    assert r.status_code==422


def test_api_full_cycle(tmp_path,monkeypatch):
    # TestClient uses local disk only; no network and no published data needed.
    import assaylens.storage as store
    monkeypatch.setattr(store,'ROOT',tmp_path)
    r=CLIENT.post('/api/datasets',files={'file':('synthetic.csv',DEMO.read_bytes(),'text/csv')})
    assert r.status_code==201,r.text
    ds=r.json()['id']
    assert r.json()['demo_data'] is True
    assert CLIENT.get('/api/datasets/'+ds).json()['input_rows']>100
    assert CLIENT.get('/api/datasets/'+ds+'/audit.csv').status_code==200
    run=CLIENT.post('/api/runs',json={'dataset_id':ds,'split_type':'scaffold','trees':16,'seed':21})
    assert run.status_code==201,run.text
    rid=run.json()['id']
    assert CLIENT.get('/api/runs/'+rid).status_code==200
    assert CLIENT.get('/api/runs/'+rid+'/bundle').status_code==200
    assert CLIENT.get('/api/runs/'+rid+'/figures/coverage.png').status_code==200
    assert CLIENT.get('/api/runs/'+rid+'/report').status_code==200
