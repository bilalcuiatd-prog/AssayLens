from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import AllChem

from .storage import ROOT


def capabilities():
    return {'vina':bool(shutil.which('vina')),'obabel':bool(shutil.which('obabel')),
        'structure_visualization':True,'ligand_conformer':True,
        'docking_requires':'Prepared receptor and ligand PDBQT files, validated search box, Vina installed',
        'warning':'Docking energies are approximate scoring-function outputs, not experimentally measured binding affinities'}


def generate_conformer(smiles, max_atoms=150):
    if not isinstance(smiles,str) or not 1<=len(smiles)<=500:
        raise ValueError('Enter a SMILES structure under 500 characters')
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms()>max_atoms:
        raise ValueError('Invalid or oversized molecular structure')
    mol=Chem.AddHs(mol)
    result=AllChem.EmbedMolecule(mol,randomSeed=42,maxAttempts=50)
    if result!=0:
        raise ValueError('3D conformer embedding failed')
    try:
        AllChem.MMFFOptimizeMolecule(mol,maxIters=200)
    except Exception:
        pass
    block=Chem.MolToMolBlock(mol)
    return block


def get_pdb_structure(pdb_id):
    if not re.fullmatch(r'[0-9][A-Za-z0-9]{3}',pdb_id or ''):
        raise ValueError('PDB ID must contain four alphanumeric characters starting with a digit')
    resp=requests.get('https://files.rcsb.org/download/'+pdb_id.upper()+'.pdb',timeout=30,headers={'User-Agent':'AssayLensResearchWorkbench/2.0'})
    resp.raise_for_status()
    if len(resp.content)>12_000_000 or not any(line.startswith(('ATOM  ','HETATM')) for line in resp.text.splitlines()[:2000]):
        raise ValueError('Structure missing expected coordinates or too large')
    return resp.text


def prepare_ligand_pdbqt(smiles):
    if not shutil.which('obabel'):
        raise RuntimeError('Open Babel (obabel) is required for optional ligand PDBQT conversion')
    ident=uuid.uuid4().hex
    folder=ROOT/'docking'/ident
    folder.mkdir(parents=True,exist_ok=True)
    sdf=folder/'conformer.sdf';pdbqt=folder/'ligand.pdbqt'
    sdf.write_text(generate_conformer(smiles),encoding='utf8')
    cmd=['obabel',str(sdf),'-O',str(pdbqt),'-h','--partialcharge','gasteiger']
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=60,check=False)
    if result.returncode!=0 or not pdbqt.exists():
        raise RuntimeError('Ligand conversion failed: '+(result.stderr or '')[-450:])
    return ident


def docking_run(receptor:bytes, ligand:bytes, center, size, exhaustiveness=8, seed=42):
    if not shutil.which('vina'):
        raise RuntimeError('AutoDock Vina executable not found in PATH. Install it first and prepare PDBQT inputs')
    if not 1<=len(receptor)<=12_000_000 or not 1<=len(ligand)<=1_000_000:
        raise ValueError('Receptor or ligand file is empty or exceeds file-size limit')
    try:
        receptor_text=receptor.decode('utf8'); ligand_text=ligand.decode('utf8')
    except UnicodeDecodeError as exc:
        raise ValueError('Expected text-based PDBQT files') from exc
    if 'ATOM' not in receptor_text and 'HETATM' not in receptor_text:
        raise ValueError('Receptor has no PDBQT coordinate records')
    if 'ATOM' not in ligand_text and 'HETATM' not in ligand_text:
        raise ValueError('Ligand has no PDBQT coordinate records')
    if len(center)!=3 or len(size)!=3 or any(not -300<=float(v)<=300 for v in center) or any(not 8<=float(v)<=35 for v in size):
        raise ValueError('Invalid docking box center or size (8–35 Å per dimension)')
    if not 1<=exhaustiveness<=32 or not 0<=seed<=2_000_000_000:
        raise ValueError('Exhaustiveness or seed out of range')
    ident=uuid.uuid4().hex
    folder=ROOT/'docking'/ident
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'receptor.pdbqt').write_bytes(receptor)
    (folder/'ligand.pdbqt').write_bytes(ligand)
    cmd=['vina','--receptor',str(folder/'receptor.pdbqt'),
         '--ligand',str(folder/'ligand.pdbqt'),
         '--out',str(folder/'poses.pdbqt'),
         '--center_x',str(float(center[0])),'--center_y',str(float(center[1])),'--center_z',str(float(center[2])),
         '--size_x',str(float(size[0])),'--size_y',str(float(size[1])),'--size_z',str(float(size[2])),
         '--exhaustiveness',str(exhaustiveness),'--seed',str(seed),'--cpu','2','--num_modes','9']
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=600,check=False,cwd=str(folder))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Docking job exceeded the ten-minute limit') from exc
    (folder/'vina.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    if result.returncode!=0 or not (folder/'poses.pdbqt').exists():
        raise RuntimeError('Vina failed: '+(result.stderr or result.stdout)[-750:])
    meta={'id':ident,'engine':'AutoDock Vina','center_A':center,'size_A':size,'exhaustiveness':exhaustiveness,
          'seed':seed,'warning':'Scores are docking approximations; not experimental affinity measurements',
          'raw_stdout_tail':result.stdout[-1500:]}
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return meta
