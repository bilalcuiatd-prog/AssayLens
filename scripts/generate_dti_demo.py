from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(12)
AA = np.asarray(list('ACDEFGHIKLMNPQRSTVWY'))
scaffolds = ['c1ccccc1','c1ccncc1','c1ccoc1','c1ccsc1','C1CCCCC1','C1CCOCC1',
             'c1ccc2ccccc2c1','c1ccc2[nH]ccc2c1','C1CCNCC1','c1cccnc1']
prefixes = ['C','CC','CCC','CO','CN','N','O','Cl','Br','COC','CCO','CCN']
smiles = []
for ring in scaffolds:
    for prefix in prefixes:
        m = Chem.MolFromSmiles(prefix + ring)
        if m is not None:
            smiles.append(Chem.MolToSmiles(m))
smiles = sorted(set(smiles))[:50]
sequences = [''.join(rng.choice(AA, size=90 + k*3)) for k in range(15)]
rows=[]
for i,smi in enumerate(smiles):
    for j,sequence in enumerate(sequences):
        if (i*7+j*11)%13==0:
            continue
        # Arbitrary deterministic labels are only for testing split plumbing.
        response = round(5.2 + 0.55*np.sin(i*.7+j*.24) + .36*np.cos(j*.6) + .1*np.sin(i+j), 5)
        rows.append({'smiles':smi,'target_id':f'DEMO_PROTEIN_{j+1:02}',
                     'target_sequence':sequence,'response':response,'endpoint':'DEMO_PACTIVITY',
                     'source_id':'SYNTHETIC_NO_EXPERIMENT','assay_id':f'DEMO_{j+1:02}',
                     'standard_relation':'=','data_validity_comment':''})
out=ROOT/'examples'/'synthetic_multi_target_dti.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(out,len(rows),len(smiles),'drugs',len(sequences),'targets')
