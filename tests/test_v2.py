import io
import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assaylens import advanced_experiments, custom_models, docking, sources_dynamic
from assaylens.studio_api import app

EXAMPLE=Path(__file__).resolve().parents[1]/'examples'/'synthetic_assay_demo.csv'


def test_extended_catalog_and_homepage():
    with TestClient(app) as client:
        home=client.get('/')
        assert home.status_code==200
        assert 'Model laboratory' in home.text
        assert 'Custom Python models' in home.text
        assert 'Structural docking' in home.text
        assert client.get('/classic').status_code==200
        data=client.get('/api/v2/catalog').json()
        assert 'ann_mlp' in data['models']['classification']
        assert 'xgboost' in data['models']['regression']
        assert len(data['targets'])==46


def test_v2_upload_and_repeated_classification(tmp_path):
    with TestClient(app) as client:
        upload=client.post('/api/datasets',files={'file':('sample.csv',EXAMPLE.read_bytes(),'text/csv')})
        assert upload.status_code==201,upload.text
        dataset_id=upload.json()['id']
        request={'dataset_id':dataset_id,'task':'classification','activity_threshold':5.0,
                 'strategy':'scaffold','folds':2,'seeds':[11,23],'models':['random_forest','logistic'],
                 'variants':['chemistry','fingerprint_only'],'n_estimators':16,'fp_bits':256}
        started=client.post('/api/v2/experiments',json=request)
        assert started.status_code==202,started.text
        job=started.json()['job_id']
        for attempt in range(100):
            state=client.get('/api/jobs/'+job).json()
            if state['state'] in ('complete','error'):
                break
            time.sleep(.12)
        assert state['state']=='complete',state
        result=client.get('/api/v2/experiments/'+state['run_id']).json()
        assert result['task']=='classification'
        assert result['n_fits']==16
        assert 'f1_mean' in result['summary'][0]
        assert client.get('/api/v2/experiments/'+state['run_id']+'/bundle').status_code==200
        assert client.get('/api/v2/experiments/'+state['run_id']+'/figure').status_code==200


def test_custom_model_upload_is_stored_not_executed(tmp_path):
    code=b'from sklearn.linear_model import Ridge\n\ndef build_model(task, params):\n    return Ridge()\n'
    parsed=custom_models.parse_upload('model.py',code)
    assert 'def build_model' in parsed
    notebook={'cells':[{'cell_type':'code','source':['from sklearn.linear_model import Ridge\n','def build_model(task, params):\n','    return Ridge()\n']}]}
    assert 'def build_model' in custom_models.parse_upload('model.ipynb',json.dumps(notebook).encode())
    with pytest.raises(ValueError):
        custom_models.parse_upload('script.ipynb',json.dumps({'cells':[{'cell_type':'code','source':['!pip install something\n','def build_model(task,params): return None']}]}).encode())
    with TestClient(app) as client:
        r=client.post('/api/v2/custom-models',files={'file':('my_model.py',code,'text/x-python')})
        assert r.status_code==201
        assert r.json()['filename']=='my_model.py'
        assert client.get('/api/v2/custom-models').status_code==200


def test_structural_tools_and_optional_execution():
    sdf=docking.generate_conformer('CCO')
    assert 'V2000' in sdf or 'V3000' in sdf
    with TestClient(app) as client:
        r=client.post('/api/v2/docking/conformer',json={'smiles':'CCO'})
        assert r.status_code==200
        assert 'sdf' in r.json()
        assert client.get('/api/v2/docking/capabilities').status_code==200
        invalid=client.post('/api/v2/targets/download',json={'target_id':'NOT_TARGET','endpoint':'Ki'})
        assert invalid.status_code==202


def test_broad_search_excludes_nonhuman_or_multichain():
    payload={'targets':[{'target_chembl_id':'CHEMBL1','organism':'Homo sapiens','target_type':'SINGLE PROTEIN','pref_name':'Human enzyme'},
                        {'target_chembl_id':'CHEMBL2','organism':'Mus musculus','target_type':'SINGLE PROTEIN','pref_name':'Mouse enzyme'},
                        {'target_chembl_id':'CHEMBL3','organism':'Homo sapiens','target_type':'PROTEIN COMPLEX','pref_name':'Complex'}]}
    with patch('assaylens.sources_dynamic.fetch_json',return_value=payload):
        targets=sources_dynamic.search_targets('enzyme')
        assert len(targets)==1
        assert targets[0]['target_chembl_id']=='CHEMBL1'


def test_classification_validation_threshold():
    from assaylens.studio_api import store_csv, read_source_csv
    sample=store_csv(EXAMPLE.read_bytes())
    df=read_source_csv(sample['id'])[2]
    with pytest.raises(ValueError):
        advanced_experiments.validate({'task':'classification','activity_threshold':19,
            'strategy':'scaffold','models':['logistic'],'variants':['chemistry'],'folds':5,'seeds':[3]},df)
