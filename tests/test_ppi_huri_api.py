import time
import pandas as pd
from fastapi.testclient import TestClient
from assaylens.studio_api import app
from assaylens import ppi_huri_api

A='ACDEFGHIKLMNPQRSTVWY'*3
B='YWVTSRQPNMLKIHGFEDCA'*3
C='MLKQWRTPHICDEGAVNSFY'*3


def test_offline_huri_job(monkeypatch):
    data=pd.DataFrame({'Protein1':[A,A,B], 'Protein2':[B,C,C], 'Y':[1,0,1]})
    monkeypatch.setattr(ppi_huri_api,'load_huri',lambda reference_controls:data)
    with TestClient(app) as client:
        response=client.post('/api/v5/ppi/huri',json={'max_rows':100,'seed':42,'include_reference_controls':True})
        assert response.status_code==202
        job=response.json()['job_id']
        state={}
        for _ in range(80):
            state=client.get(f'/api/v5/ppi/jobs/{job}').json()
            if state['state'] in ('complete','error'):
                break
            time.sleep(0.05)
        assert state['state']=='complete',state
        assert state['manifest']['export_rows']==3
        assert client.get(f'/api/v5/ppi/jobs/{job}/csv').status_code==200
        assert client.get(f'/api/v5/ppi/jobs/{job}/manifest').status_code==200
