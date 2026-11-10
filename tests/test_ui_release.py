from fastapi.testclient import TestClient
from assaylens.studio_api import app


def test_self_contained_dashboard_and_health():
    with TestClient(app) as client:
        response = client.get('/')
        text = response.text
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store, max-age=0'
        assert '<style>' in text
        assert 'id="page-overview"' in text
        assert 'id="page-datasets"' in text
        assert 'id="page-experiments"' in text
        assert 'id="page-results"' in text
        assert 'async function init()' in text
        assert '<link rel="stylesheet" href="/assets/style.css">' not in text
        assert '<script src="/assets/app.js" defer></script>' not in text
        assert client.get('/api/health').json()['version'] == '5.0.0'
        assert len(client.get('/api/catalog').json()['targets']) == 46
