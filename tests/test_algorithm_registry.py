import json
from types import SimpleNamespace

import pytest

from dasklearn.algorithms import ALGORITHMS, class_paths, import_class, settings_classes
from dasklearn.session_settings import LearningSettings


@pytest.mark.parametrize('algorithm', list(ALGORITHMS))
def test_every_algorithm_has_importable_simulation_and_serializable_settings(algorithm,tmp_path):
    settings_type,simulation_type=map(import_class,class_paths(algorithm))
    assert callable(simulation_type)
    settings=settings_type(algorithm=algorithm,seed=3,work_dir=str(tmp_path),dataset='synthetic',
        participants=6,learning=LearningSettings(.003,0,0,16,1))
    restored=settings_classes()[settings_type.__name__].from_dict(settings.to_dict())
    assert restored==settings
    assert isinstance(restored.learning,LearningSettings)


def test_loading_saved_run_keeps_algorithm_and_its_parameters(tmp_path,monkeypatch):
    import main
    from dasklearn.simulation.demo.settings import DeMoSettings
    saved=DeMoSettings(algorithm='demo',seed=7,work_dir=str(tmp_path),dataset='synthetic',
        participants=6,learning=LearningSettings(.003,0,0,16,1),compression_topk=7)
    (tmp_path/'settings.json').write_text(saved.to_json())
    monkeypatch.setattr('sys.argv',['main.py','--from-dir',str(tmp_path)])
    monkeypatch.setattr(main,'get_torch_device',lambda:'cpu')
    original=main.instantiate_class;observed=[]
    async def finished():pass
    def instantiate(path,**kwargs):
        if '.settings.' in path:return original(path,**kwargs)
        observed.append(kwargs['settings'])
        return SimpleNamespace(run=finished)
    monkeypatch.setattr(main,'instantiate_class',instantiate)
    monkeypatch.setattr(main,'ensure_future',lambda coroutine:coroutine.close())
    main.run()
    restored=observed[0]
    assert isinstance(restored,DeMoSettings)
    assert restored.algorithm=='demo' and restored.compression_topk==7
    assert restored.learning==saved.learning
    assert restored.from_dir==str(tmp_path)


def test_main_cli_passes_execution_opt_outs(monkeypatch):
    import main
    monkeypatch.setattr('sys.argv',['main.py','--algorithm','demo','--momentum','0',
        '--no-gpu-residency','--no-training-batching','--gpu-state-cache-mb','17'])
    original=main.instantiate_class;observed=[]
    async def finished():pass
    def instantiate(path,**kwargs):
        if '.settings.' in path:return original(path,**kwargs)
        observed.append(kwargs['settings']);return SimpleNamespace(run=finished)
    monkeypatch.setattr(main,'instantiate_class',instantiate)
    monkeypatch.setattr(main,'ensure_future',lambda coroutine:coroutine.close())
    main.run()
    assert not observed[0].gpu_residency and not observed[0].training_batching
    assert observed[0].gpu_state_cache_mb==17
