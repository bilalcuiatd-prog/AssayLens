from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem

root=Path(__file__).resolve().parent.parent/'examples'
root.mkdir(exist_ok=True)
rng=np.random.default_rng(6104)
scaffolds=['c1ccccc1','c1ccncc1','c1ccoc1','c1ccsc1','C1CCCCC1','C1CCCC1','C1CCNCC1','C1CCOCC1','c1ccc2ccccc2c1','c1ccc2[nH]ccc2c1','c1ccc2occc2c1','C1CCC2CCCCC2C1']
prefix=['C','CC','CO','N','O','Cl','Br','CCC','CN','COC']
mols=[]
for sc in scaffolds:
 for pre in prefix:
  mol=Chem.MolFromSmiles(pre+sc)
  if mol is None:continue
  s=Chem.MolToSmiles(mol,isomericSmiles=True)
  if s not in mols:mols.append(s)
  if len(mols)>=50:break
 if len(mols)>=50:break
pairs=list(itertools.combinations(range(len(mols)),2))
rng.shuffle(pairs)
rows=[]
for idx,(i,j) in enumerate(pairs[:850]):
 v=int((i*3+j*7+idx%11)%19<9)
 rows.append({'drug_a_id':f'DEMO_D{i:03d}', 'drug_b_id':f'DEMO_D{j:03d}', 'drug_a':mols[i], 'drug_b':mols[j], 'label':v,
              'interaction_type':'synthetic_binary_pair','source_id':'SYNTHETIC_DDI','evidence_type':'Generated for software testing; not experimental'})
pd.DataFrame(rows).to_csv(root/'synthetic_ddi_pairs.csv',index=False)
aa='ACDEFGHIKLMNPQRSTVWY'
seqs=[''.join(rng.choice(list(aa),size=90).tolist()) for _ in range(42)]
pairs=list(itertools.combinations(range(len(seqs)),2))
rng.shuffle(pairs)
rows=[]
for idx,(i,j) in enumerate(pairs[:650]):
 v=int((i*17+j*13+idx%17)%23<11)
 rows.append({'protein_a_id':f'DEMO_P{i:03d}', 'protein_b_id':f'DEMO_P{j:03d}', 'protein_a':seqs[i], 'protein_b':seqs[j],
              'protein_cluster_a':f'DEMO_C{i:03d}', 'protein_cluster_b':f'DEMO_C{j:03d}', 'label':v,
              'interaction_type':'synthetic_binary_pair','source_id':'SYNTHETIC_PPI','evidence_type':'Generated for software testing; not experimental'})
pd.DataFrame(rows).to_csv(root/'synthetic_ppi_pairs.csv',index=False)
print('DDI rows:',850,'PPI rows:',650)
