from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import rdBase

from .sources import (CHEMBL, COLUMNS, ENDPOINTS, check_chembl_record,
    chembl_pages, fetch_json, session, choose_upload, write_frame)


def search_targets(query, limit=30):
    q=str(query).strip()
    if not 2 <= len(q) <= 100:
        raise ValueError('Search for a protein name, gene, UniProt accession, or ChEMBL identifier (2–100 characters)')
    s=session()
    response=fetch_json(s,CHEMBL+'target/search.json',{'q':q,'limit':min(limit,50)})
    out=[]
    for target in response.get('targets',[]):
        if target.get('organism') != 'Homo sapiens' or target.get('target_type')!='SINGLE PROTEIN':
            continue
        out.append({'target_chembl_id':target['target_chembl_id'],
            'name':target.get('pref_name',''), 'organism':target.get('organism'),
            'accessions':[c.get('accession') for c in target.get('target_components',[]) if c.get('accession')]})
    return out


def download_target(target_id, endpoint, folder, progress=lambda _:None, max_raw=0, max_rows=8000):
    if not re.fullmatch(r'CHEMBL\d{1,9}',str(target_id)):
        raise ValueError('Supply a valid CHEMBL protein target identifier')
    if endpoint not in ENDPOINTS:
        raise ValueError('Choose IC50, Ki, or Kd')
    if not 0<=max_raw<=100000:
        raise ValueError('Raw record limit out of range')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    s=session()
    target=fetch_json(s,CHEMBL+'target/'+target_id+'.json')
    if target.get('organism')!='Homo sapiens' or target.get('target_type')!='SINGLE PROTEIN':
        raise ValueError('This import requires a human, directly assigned single-protein target')
    status=fetch_json(s,CHEMBL+'status.json')
    activities=[]
    with gzip.open(folder/'raw_activities.jsonl.gz','wt',encoding='utf8') as rawfile:
        for activity in chembl_pages(s,'activity',{'target_chembl_id':target_id,'standard_type':endpoint,'order_by':'activity_id'},'activities',max_raw=max_raw):
            activities.append(activity)
            rawfile.write(json.dumps(activity)+'\n')
            if len(activities)%500==0:
                progress(f'Downloaded {len(activities):,} activity records')
    if not activities:
        raise ValueError('No quantitative activity measurements for this target and endpoint')
    ids=sorted({a.get('assay_chembl_id') for a in activities if a.get('assay_chembl_id')})
    assays={}
    for start in range(0,len(ids),25):
        selection=ids[start:start+25]
        for assay in chembl_pages(s,'assay',{'assay_chembl_id__in':','.join(selection),'order_by':'assay_chembl_id'},'assays'):
            if assay.get('assay_chembl_id') in selection:
                assays[assay['assay_chembl_id']]=assay
        for aid in selection:
            if aid not in assays:
                try:
                    assays[aid]=fetch_json(s,CHEMBL+'assay/'+aid+'.json')
                except Exception:
                    continue
        progress(f'Retrieved metadata for {len(assays):,}/{len(ids):,} assays')
    with gzip.open(folder/'assays.jsonl.gz','wt',encoding='utf8') as af:
        for aid in sorted(assays):
            af.write(json.dumps(assays[aid])+'\n')
    accepted=[]; rejected=[]
    for row in activities:
        item, reason=check_chembl_record(row,assays.get(row.get('assay_chembl_id')),target_id,endpoint)
        if reason:
            rejected.append({'activity_id':row.get('activity_id'),'reason':reason})
        else:
            accepted.append(item)
    write_frame(rejected,['activity_id','reason'],folder/'excluded.csv')
    write_frame(accepted,COLUMNS+['original_nm','activity_id'],folder/'filtered_full.csv')
    if not accepted:
        raise ValueError('No qualifying measurements after filtering; review excluded.csv')
    full=pd.DataFrame(accepted)
    upload,sampling=choose_upload(full,max_rows)
    upload[COLUMNS].to_csv(folder/'assaylens_import.csv',index=False)
    manifest={'provider':'ChEMBL','target_id':target_id,'target_name':target.get('pref_name'),
        'endpoint':endpoint,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
        'chembl_version':status.get('chembl_db_version'),'raw_count':len(activities),
        'filtered_count':len(full),'upload_count':len(upload),'sampling_policy':sampling,
        'raw_limit':max_raw,'excluded_counts':dict(Counter(r['reason'] for r in rejected)),
        'rdkit_version':rdBase.rdkitVersion,'raw_sha256':hashlib.sha256((folder/'raw_activities.jsonl.gz').read_bytes()).hexdigest(),
        'warning':'Subsampled CSVs are not a replacement for full-cohort validation'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return folder/'assaylens_import.csv',manifest
