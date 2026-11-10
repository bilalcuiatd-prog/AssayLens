import io
import time
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assaylens import interactions
from assaylens.studio_api import app

ROOT = Path(__file__).resolve().parent.parent
CLIENT = TestClient(app)


@pytest.fixture(params=['ddi','ppi'])
def examples(request):
    kind=request.param
    data=pd.read_csv(ROOT/'examples'/f'synthetic_{kind}_pairs.csv')
    frame,rejected,manifest=interactions.curate(data,kind)
    return kind,frame,rejected,manifest


def test_realistic_pair_datasets_curate(examples):
    kind,frame,rejected,meta=examples
    assert len(frame)>=500
    assert len(rejected)==0
    assert meta['unique_entities']>=40
    assert meta['positive_only'] is False
    assert meta['connected_components']==1
    assert interactions.network_summary(frame)['edges']>=500
    assert len(interactions.features(frame.head(10),kind)['joint'])==10


@pytest.mark.parametrize('mode',['warm','cold_one','cold_both'])
def test_entity_disjoint_splits(examples,mode):
    kind,df,_,_=examples
    for train,test,excluded in interactions.make_splits(df,mode,folds=5,seed=17):
        audit=interactions.check_split(df,mode,train,test,excluded)
        assert audit['train']>10
        assert audit['test']>5
        tr=set(df.entity_a.iloc[train])|set(df.entity_b.iloc[train])
        if mode=='cold_one':
            assert all((a in tr) ^ (b in tr) for a,b in zip(df.entity_a.iloc[test],df.entity_b.iloc[test]))
        if mode=='cold_both':
            assert not (tr & (set(df.entity_a.iloc[test])|set(df.entity_b.iloc[test])))
        if mode=='warm':
            assert not (set(zip(df.entity_a.iloc[train],df.entity_b.iloc[train]))&set(zip(df.entity_a.iloc[test],df.entity_b.iloc[test])))


def test_reverse_duplicate_does_not_leak():
    frame=pd.DataFrame({'drug_a':['CCO','CCN'],'drug_b':['CCN','CCO'],'label':[1,1]})
    df,excluded,meta=interactions.curate(frame,'ddi')
    assert len(df)==1
    assert df.n_records.iloc[0]==2
    assert meta['contradictory_pair_groups']==0
    frame['label']=[1,0]
    with pytest.raises(ValueError,match='conflicting'):
        interactions.curate(frame,'ddi')


def test_positive_only_is_not_classification_dataset():
    data=pd.read_csv(ROOT/'examples'/'synthetic_ppi_pairs.csv')
    data['label']=1
    df,_,meta=interactions.curate(data,'ppi')
    assert meta['positive_only']
    with pytest.raises(ValueError,match='both experimentally supported'):
        interactions.validate({'task':'classification','strategy':'warm','folds':3,'seeds':[3],'models':['random_forest'],'variants':['joint']},df,'ppi')


def test_sequence_requirement_for_ppi_training():
    data=pd.read_csv(ROOT/'examples'/'synthetic_ppi_pairs.csv')
    data['protein_a']=''
    data['protein_b']=''
    curated,_,meta=interactions.curate(data,'ppi')
    assert meta['protein_sequence_complete'] is False
    assert interactions.network_summary(curated)['nodes']>30
    with pytest.raises(ValueError,match='requires protein sequences'):
        interactions.validate({'task':'classification','strategy':'warm','folds':3,'seeds':[2],'models':['random_forest'],'variants':['joint']},curated,'ppi')


def test_interaction_type_cannot_be_pooled(examples):
    kind,df,_,_=examples
    df.loc[:5,'interaction_type']='different_event'
    with pytest.raises(ValueError,match='Multiple interaction types'):
        interactions.validate({'task':'classification','strategy':'warm','folds':3,'seeds':[2],'models':['random_forest'],'variants':['joint']},df,kind)


def test_paired_api_upload_graph_and_experiment(examples):
    kind,_,_,_=examples
    data=(ROOT/'examples'/f'synthetic_{kind}_pairs.csv').read_bytes()
    r=CLIENT.post(f'/api/v4/pairs/{kind}/datasets',files={'file':('pairs.csv',data,'text/csv')})
    assert r.status_code==201,r.text
    did=r.json()['id']
    graph=CLIENT.get(f'/api/v4/pairs/{kind}/datasets/{did}/graph')
    assert graph.status_code==200
    assert graph.json()['network']['nodes']>30
    assert CLIENT.get(f'/api/v4/pairs/{kind}/datasets/{did}/csv').status_code==200
    cfg={'dataset_id':did,'task':'classification','strategy':'cold_one','folds':3,'seeds':[17],
         'models':['extra_trees'],'variants':['similarity_only'],'n_estimators':12}
    response=CLIENT.post(f'/api/v4/pairs/{kind}/experiments',json=cfg)
    assert response.status_code==202,response.text
    jid=response.json()['job_id']
    for _ in range(130):
        job=CLIENT.get(f'/api/v4/jobs/{jid}').json()
        if job['state'] in ('complete','error'):break
        time.sleep(.1)
    assert job['state']=='complete',job
    assert job['n_fits']==3
    assert CLIENT.get(f'/api/v4/pairs/{kind}/runs/{job["run_id"]}/bundle').status_code==200
    assert CLIENT.get(f'/api/v4/pairs/{kind}/runs/{job["run_id"]}/network.png').status_code==200


def test_string_assoc_import_keeps_positive_only(monkeypatch):
    import assaylens.interaction_api as api
    class Response:
        text='stringId_A\tstringId_B\tscore\n9606.ENSP1\t9606.ENSP2\t0.97\n9606.ENSP2\t9606.ENSP3\t0.81\n'
        def raise_for_status(self):pass
    monkeypatch.setattr(api.requests,'post',lambda *a,**kw:Response())
    job=api.make_job('ppi','string')
    api.string_task(job,api.StringRequest(genes=['ACHE','APP']))
    assert api.JOBS[job]['state']=='complete'
    d=api.JOBS[job]['audit']
    assert d['positive_only']
    assert d['protein_sequence_complete'] is False


def test_site_contains_interaction_pages():
    r=CLIENT.get('/')
    assert r.status_code==200
    assert 'page-ddi' in r.text and 'page-ppi' in r.text
    assert 'Drug–drug interactions' in r.text
    assert CLIENT.get('/api/v4/pairs/catalog').json()['kinds']==['ddi','ppi']
