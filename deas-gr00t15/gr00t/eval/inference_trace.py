"""Portable, pickle-free policy inputs, outputs and RNG snapshots for replay."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def array_copy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.hasobject:
        if not all(isinstance(item, (str, np.str_)) for item in value.flat):
            raise TypeError('Inference trace only accepts numeric arrays or strings')
        value = value.astype(str)
    return value.copy()


def capture_rng():
    numpy_state = np.random.get_state()
    metadata = {
        'python_random': random.getstate(),
        'numpy_random': [numpy_state[0], numpy_state[2], numpy_state[3], numpy_state[4]],
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'torch_version': str(torch.__version__),
    }
    arrays = {'rng_numpy_keys': numpy_state[1].copy(), 'rng_torch_cpu': torch.get_rng_state().numpy()}
    for index, state in enumerate(torch.cuda.get_rng_state_all() if metadata['cuda_devices'] else []):
        arrays[f'rng_torch_cuda_{index}'] = state.cpu().numpy()
    return metadata, arrays


def restore_rng(metadata, arrays):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    random.setstate(tuples(metadata['python_random']))
    mode, position, has_gauss, cached_gauss = metadata['numpy_random']
    np.random.set_state((mode, arrays['rng_numpy_keys'], position, has_gauss, cached_gauss))
    torch.set_rng_state(torch.from_numpy(arrays['rng_torch_cpu'].copy()))
    if metadata['cuda_devices']:
        if torch.cuda.device_count() != metadata['cuda_devices']:
            raise ValueError('Replay must expose the same number of CUDA devices as the recording')
        torch.cuda.set_rng_state_all([torch.from_numpy(arrays[f'rng_torch_cuda_{i}'].copy())
                                      for i in range(metadata['cuda_devices'])])
    torch.backends.cudnn.deterministic = metadata['cudnn_deterministic']
    torch.backends.cudnn.benchmark = metadata['cudnn_benchmark']
    torch.backends.cudnn.allow_tf32 = metadata['cudnn_allow_tf32']
    torch.set_float32_matmul_precision(metadata['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = metadata['matmul_allow_tf32']
    torch.use_deterministic_algorithms(metadata['deterministic_algorithms'],
                                     warn_only=metadata['deterministic_warn_only'])


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays.pop('metadata_json').item()))
    return metadata, arrays


def prefixed(arrays, prefix):
    return {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}


class InferenceTraceRecorder:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = dict(config)
        self.calls = 0

    def begin(self, observation, context):
        inputs = {'input::' + key: array_copy(value) for key, value in observation.items()}
        rng, arrays = capture_rng()
        arrays.update(inputs)
        return dict(schema_version=1, call=self.calls, config=self.config, context=context, rng=rng), arrays

    def finish(self, pending, policy, predicted_actions, supplied_actions, executed_steps):
        metadata, arrays = pending
        metadata['executed_steps'] = [int(value) for value in executed_steps]
        arrays.update({'output::' + key: array_copy(value) for key, value in predicted_actions.items()})
        arrays.update({'supplied::' + key: array_copy(value) for key, value in supplied_actions.items()})
        for key, value in (getattr(policy, 'last_candidates', None) or {}).items():
            arrays['candidate::' + key] = array_copy(value)
        scores = getattr(policy, 'last_scores', None)
        if scores is not None:
            arrays['q_scores'] = array_copy(scores)
        arrays['metadata_json'] = np.asarray(json.dumps(metadata, allow_nan=False))
        filename = f'call-{self.calls:06d}.npz'
        path = self.directory / filename
        temporary = path.with_suffix('.pending')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
        with (self.directory / 'index.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls, 'file':filename, 'context':metadata['context'],
                                     'executed_steps':metadata['executed_steps']}) + '\n')
        self.calls += 1
