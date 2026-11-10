from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from .storage import ROOT

IMAGE='assaylens-custom:2'


def parse_upload(name: str, payload: bytes):
    if len(payload)>300_000:
        raise ValueError('The custom model file exceeds 300 KB')
    if not name.lower().endswith(('.py','.ipynb')):
        raise ValueError('Upload a Python .py or Jupyter .ipynb file')
    if name.lower().endswith('.ipynb'):
        try:
            book=json.loads(payload.decode('utf8'))
            if not isinstance(book['cells'],list):
                raise ValueError('Invalid notebook structure')
            chunks=[]
            for cell in book['cells']:
                if cell.get('cell_type')!='code':
                    continue
                source=cell.get('source','')
                source=''.join(source) if isinstance(source,list) else str(source)
                if any(line.lstrip().startswith(('%','!')) for line in source.splitlines()):
                    raise ValueError('Notebook shell and IPython magic lines are unsupported; use standard Python')
                chunks.append(source)
            code='\n\n'.join(chunks)
        except (UnicodeDecodeError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise ValueError('Could not read Jupyter notebook') from exc
    else:
        try:
            code=payload.decode('utf8')
        except UnicodeDecodeError as exc:
            raise ValueError('Only UTF-8 Python files are supported') from exc
    import ast
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f'Python syntax error on line {exc.lineno}') from exc
    if not re.search(r'\bdef\s+build_model\s*\(',code):
        raise ValueError('Custom models must define build_model(task, params)')
    return code


def save_plugin(name: str,payload:bytes):
    code=parse_upload(name,payload)
    identifier=uuid.uuid4().hex
    folder=ROOT/'custom_models'/identifier
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'model.py').write_text(code,encoding='utf8')
    details={'id':identifier,'filename':Path(name).name,'interface':'build_model(task, params)',
             'execution':'Docker isolation; local deployment only','size':len(payload)}
    (folder/'metadata.json').write_text(json.dumps(details,indent=2),encoding='utf8')
    return details


def list_plugins():
    root=ROOT/'custom_models'
    if not root.exists():
        return []
    result=[]
    for folder in sorted(root.iterdir()):
        meta=folder/'metadata.json'
        if meta.is_file():
            try:
                result.append(json.loads(meta.read_text(encoding='utf8')))
            except (ValueError,OSError):
                continue
    return result


def sandbox_status():
    available=bool(shutil.which('docker'))
    return {'docker_cli':available,'image':IMAGE,'enabled':available,
            'notice':'Custom Python is never executed directly by the web server. Docker must be installed and the sandbox image built.'}


def docker_run(plugin_id: str,dataset,config,output,progress=lambda _:None,timeout_seconds=1200):
    if not re.fullmatch(r'[a-f0-9]{32}',plugin_id):
        raise ValueError('Invalid model identifier')
    plugin=ROOT/'custom_models'/plugin_id/'model.py'
    if not plugin.is_file():
        raise ValueError('Custom model not found')
    if not shutil.which('docker'):
        raise RuntimeError('Docker Desktop is required. Build the isolated image using BUILD_CUSTOM_SANDBOX.bat')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    (output/'curated.csv').write_bytes(Path(dataset).read_bytes())
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    try:
        output.chmod(0o777)
        (output/'curated.csv').chmod(0o644)
        (output/'config.json').chmod(0o644)
    except OSError:
        pass
    work=str(output).replace('\\','/')
    file=str(plugin.resolve()).replace('\\','/')
    argv=['docker','run','--rm','--network','none','--pids-limit','128',
          '--memory','2g','--cpus','2','--cap-drop','ALL',
          '--security-opt','no-new-privileges','--read-only',
          '--tmpfs','/tmp:rw,nosuid,size=128m','--user','65534:65534',
          '--mount',f'type=bind,source={work},target=/job',
          '--mount',f'type=bind,source={file},target=/plugin/model.py,readonly',
          IMAGE,'python','/srv/custom_worker.py','/job']
    progress('Starting isolated Docker custom-model experiment')
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout_seconds,check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Custom-model container exceeded its time limit') from exc
    if result.returncode!=0:
        raise RuntimeError('Isolated custom model failed: '+(result.stderr or result.stdout)[-1200:])
    result_path=output/'custom_result.json'
    if not result_path.is_file():
        raise RuntimeError('Sandbox returned without a result file')
    return json.loads(result_path.read_text(encoding='utf8'))
