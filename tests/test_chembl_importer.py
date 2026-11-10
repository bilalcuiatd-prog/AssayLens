"""Offline contract tests: fake ChEMBL payloads; NO real biological measurements."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'download_chembl_enzyme.py'
spec = spec_from_file_location('chembl_importer', PATH)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def activity(**override):
    d = dict(activity_id=1, target_chembl_id='CHEMBL205', standard_type='Ki',
             target_organism='Homo sapiens', standard_relation='=', standard_units='nM',
             standard_value='100', data_validity_comment=None, potential_duplicate=0,
             standard_flag=1, assay_chembl_id='CHEMBLASSAY1', canonical_smiles='CCO',
             molecule_chembl_id='CHEMBLMOL1', document_chembl_id='CHEMBLDOC1',
             pchembl_value=7.0)
    d.update(override)
    return d


def assay(**override):
    d = dict(assay_chembl_id='CHEMBLASSAY1', confidence_score=9,
             target_chembl_id='CHEMBL205', relationship_type='D', assay_type='B',
             bao_label='single protein format')
    d.update(override)
    return d


def test_exact_math_and_schema():
    value, reason = mod.filter_activity(activity(), assay(), target='CHEMBL205', endpoint='Ki', organism='Homo sapiens')
    assert reason == 'accepted'
    assert value['response'] == 7.0
    assert list(value)[:10] == mod.SCHEMA


def test_relation_confidence_and_warning_filters():
    cases = [(activity(standard_relation='>'), assay(), 'not_exact_measurement'),
             (activity(), assay(confidence_score=8), 'assay_confidence_not_9'),
             (activity(data_validity_comment='Manually validated'), assay(), 'activity_validity_comment'),
             (activity(potential_duplicate=1), assay(), 'possible_duplicate_or_unknown_flag'),
             (activity(standard_units='uM'), assay(), 'units_not_nM'),
             (activity(standard_value='0'), assay(), 'missing_or_nonpositive_value'),
             (activity(standard_value='100', pchembl_value=9.0), assay(), 'pchembl_disagreement'),
             (activity(canonical_smiles='not-a-molecule'), assay(), 'invalid_or_unsupported_smiles')]
    for record, metadata, expected in cases:
        result, reason = mod.filter_activity(record, metadata, target='CHEMBL205', endpoint='Ki', organism='Homo sapiens')
        assert result is None
        assert reason == expected


def test_curate_and_sample_preserve_schema():
    raw = [activity(activity_id=i+1, assay_chembl_id=f'AS{i%7}',
                    canonical_smiles='CCO' if i%2==0 else 'CCCC',
                    standard_value=str(100 + i), pchembl_value=None)
           for i in range(70)]
    assays = {f'AS{i}': assay(assay_chembl_id=f'AS{i}') for i in range(7)}
    clean, excluded, reasons = mod.curate(raw, assays, target='CHEMBL205', endpoint='Ki')
    assert len(clean) == 70 and excluded.empty and reasons['accepted']==70
    eligible, app_csv = mod.prepare_import(clean, min_assay_records=2, limit=35, seed=7)
    assert len(eligible)==70 and len(app_csv)==35
    assert app_csv.columns.tolist() == mod.SCHEMA
    assert len(app_csv.assay_id.unique()) >= 6


def test_pagination_reads_every_page(monkeypatch):
    calls=[]
    def fake(resource, params):
        calls.append(params['offset'])
        if params['offset']==0:
            return {'activities':[{'activity_id':1},{'activity_id':2}], 'page_meta':{'next':'page2'}}
        return {'activities':[{'activity_id':3}], 'page_meta':{'next':None}}
    monkeypatch.setattr(mod, 'get_json', fake)
    assert [v['activity_id'] for v in mod.iter_pages('activity.json',{},'activities',page_size=2)]==[1,2,3]
    assert calls==[0,2]


def test_invalid_row_cap():
    df=pd.DataFrame([{'activity_id':1,'smiles':'CCO','assay_id':'A','response':7}])
    try:
        mod.prepare_import(df, limit=8001)
    except ValueError as e:
        assert '8000' in str(e)
    else:
        assert False


def test_full_cli_offline_contract(monkeypatch, tmp_path):
    """Simulate ChEMBL response, verify all disk products and the local auditor."""
    from pathlib import Path
    import gzip
    import json
    source = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_assay_demo.csv'
    input_rows = pd.read_csv(source)
    raw = []
    for i, r in input_rows.iterrows():
        raw.append(activity(
            activity_id=i + 1, canonical_smiles=r.smiles,
            molecule_chembl_id=f'FAKE_MOL_{i}',
            assay_chembl_id=f'FAKE_ASSAY_{r.assay_id}',
            document_chembl_id=f'FAKE_DOC_{i // 10}',
            standard_value=10 ** (3.0 - (i % 90) / 90),
            pchembl_value=None,
        ))
    assays = {x['assay_chembl_id']: assay(assay_chembl_id=x['assay_chembl_id']) for x in raw}
    def fake_get_json(resource, params=None, **kwargs):
        if resource.startswith('target/'):
            return {'target_type': 'SINGLE PROTEIN', 'organism': 'Homo sapiens', 'pref_name': 'Fake offline enzyme test'}
        if resource == 'status.json':
            return {'chembl_db_version': 'OFFLINE_TEST_NOT_REAL'}
        raise RuntimeError('Unexpected network request')
    def fake_retrieve_activities(target, endpoint, path, max_raw=0):
        with gzip.open(path, 'wt', encoding='utf8') as f:
            for entry in raw:
                f.write(json.dumps(entry) + '\n')
        return raw
    def fake_retrieve_assays(ids, path):
        with gzip.open(path, 'wt', encoding='utf8') as f:
            for entry in assays.values():
                f.write(json.dumps(entry) + '\n')
        return assays
    monkeypatch.setattr(mod, 'get_json', fake_get_json)
    monkeypatch.setattr(mod, 'retrieve_activities', fake_retrieve_activities)
    monkeypatch.setattr(mod, 'retrieve_assays', fake_retrieve_assays)
    outdir = tmp_path / 'fake_chEMBL'
    mod.main(['--outdir', str(outdir), '--max-upload-rows', '200', '--seed', '13'])
    app = pd.read_csv(outdir / 'assaylens_import.csv')
    manifest = json.loads((outdir / 'manifest.json').read_text())
    assert app.columns.tolist() == mod.SCHEMA
    assert len(app) <= 200 and app['assay_id'].nunique() >= 6
    assert manifest['api_status']['chembl_db_version'] == 'OFFLINE_TEST_NOT_REAL'
    assert manifest['counts']['downloaded_raw'] == len(raw)
    assert (outdir / 'excluded_rows.csv').is_file()
    assert (outdir / 'assaylens_audit_summary.json').is_file()
