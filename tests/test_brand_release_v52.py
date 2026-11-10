from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from assaylens.studio_api import app

PROJECT=Path(__file__).resolve().parent.parent

def test_brand_asset_exact_and_reachable():
    img=PROJECT/'web'/'assaylens-logo-original.png'
    assert Image.open(img).size==(1536,1024)
    with TestClient(app) as client:
        reply=client.get('/assets/assaylens-logo-original.png')
        assert reply.status_code==200
        assert reply.content==img.read_bytes()

def test_main_dashboard_has_new_brand_and_no_classic_link():
    with TestClient(app) as client:
        resp=client.get('/')
        assert resp.status_code==200
        assert 'assaylens-logo-original.png' in resp.text
        assert 'Open classic workbench' not in resp.text
        assert '<div class="brand-plate">' in resp.text
        assert resp.text.count('<style>')==resp.text.count('</style>')
        assert client.get('/api/health').json()['status']=='ok'
