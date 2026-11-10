import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import assaylens.advanced_experiments as advanced


def main(folder):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    module_spec=importlib.util.spec_from_file_location('uploaded_model','/plugin/model.py')
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not callable(getattr(module,'build_model',None)):
        raise ValueError('build_model(task, params) is required')
    original=advanced.build_estimator
    def factory(kind,task,seed,cfg):
        if kind!='custom':
            return original(kind,task,seed,cfg)
        return module.build_model(task,dict(cfg,random_state=seed))
    advanced.build_estimator=factory
    advanced.REGRESSION_MODELS+=('custom',)
    advanced.CLASSIFICATION_MODELS+=('custom',)
    result=advanced.run(pd.read_csv(folder/'curated.csv'),config,folder)
    (folder/'custom_result.json').write_text(json.dumps(result,allow_nan=False),encoding='utf8')

if __name__=='__main__':
    main(sys.argv[1])
