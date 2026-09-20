import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

from gr00t.eval.inference_trace import InferenceTraceRecorder, load_trace, prefixed, restore_rng


def test_trace_roundtrip_rng_and_cpu_candidate_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    trace = InferenceTraceRecorder(tmp_path / 'inference', {'temperature': 0})
    observation = {'video.camera': np.zeros((2, 1, 8, 8, 3), dtype=np.uint8),
                   'annotation.language': np.array(['pick mug', 'turn stove'], dtype=object),
                   'state.position': np.array([[[0.2]], [[0.7]]])}
    pending = trace.begin(observation, {'env_episodes': [0, 0], 'episode_steps': [0, 8]})
    original_random = (random.random(), np.random.rand(), torch.rand(5))
    candidates = {'action.position': np.arange(6 * 16).reshape(6, 16, 1)}
    scores = torch.tensor([[0., 1.], [3., 0.], [2., 4.]])
    chosen = {'action.position': candidates['action.position'][[2, 5]]}
    trace.finish(pending, SimpleNamespace(last_candidates=candidates, last_scores=scores),
                 chosen, {'action.position': chosen['action.position'][:, :8]}, [8, 8])
    metadata, arrays = load_trace(tmp_path / 'inference/call-000000.npz')
    assert prefixed(arrays, 'input::')['annotation.language'].tolist() == ['pick mug', 'turn stove']
    assert arrays['output::action.position'].shape == (2, 16, 1)
    assert arrays['supplied::action.position'].shape == (2, 8, 1)
    restore_rng(metadata['rng'], arrays)
    assert random.random() == original_random[0]
    assert np.random.rand() == original_random[1]
    assert torch.equal(torch.rand(5), original_random[2])
    script = Path(__file__).resolve().parents[1] / 'scripts/robocasa/replay_inference.py'
    spec = importlib.util.spec_from_file_location('replay_inference', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['replay', '--trace', str(tmp_path / 'inference/call-000000.npz'),
                                    '--report', str(report)])
    module.main()
    assert json.loads(report.read_text())['restored_exactly'] is True

    with np.load(report.with_suffix('.actions.npz'), allow_pickle=False) as output:
        np.testing.assert_array_equal(output['restored::action.position'], chosen['action.position'])
