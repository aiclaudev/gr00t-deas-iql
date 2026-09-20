#!/usr/bin/env python3
"""Restore saved BoN output on CPU, or recompute it from saved input/RNG on a worker."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output-actions',type=Path,help='NPZ output; defaults next to the report')
    parser.add_argument('--recompute',action='store_true',help='Reload checkpoints and rerun inference on a GPU worker')
    parser.add_argument('--atol',type=float,default=1e-5)
    args=parser.parse_args()
    home=Path('/home/nas_main/dohyunlee')
    actions_path=args.output_actions or args.report.with_suffix('.actions.npz')
    for path in (args.trace,args.report,actions_path):
        if not path.resolve().is_relative_to(home): raise ValueError('Keep replay paths in the user home')
    from gr00t.eval.inference_trace import load_trace,prefixed
    metadata,arrays=load_trace(args.trace)
    config=metadata['config']
    expected=prefixed(arrays,'output::')
    candidates=prefixed(arrays,'candidate::')
    if not candidates or 'q_scores' not in arrays or config['temperature']!=0:
        raise ValueError('CPU restoration requires recorded greedy BoN candidates and scores')
    scores=arrays['q_scores']
    batch=scores.shape[1]
    selected=scores.argmax(axis=0)*batch+np.arange(batch)
    restored={key:value[selected] for key,value in candidates.items()}
    matches={key:bool(np.array_equal(expected[key],restored[key])) for key in expected}
    report={'trace':str(args.trace.resolve()),'restored_exactly':all(matches.values()),
            'restored_keys':matches,'inference_recomputed':False}
    if not report['restored_exactly']: raise ValueError('Saved candidates do not reproduce selected output')
    output_arrays={'restored::'+key:value for key,value in restored.items()}
    if args.recompute:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run GPU replay in an sbatch worker')
        from gr00t.eval.inference_trace import restore_rng
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.checkpoint_bon_policy import CheckpointDEASBoNPolicy
        data=DATA_CONFIG_MAP[config['data_config']](AS=config['action_horizon'])
        if config.get('deas_backend') == 'iql':
            from gr00t.model.iql_bon_policy import CheckpointIQLBoNPolicy
            CheckpointDEASBoNPolicy = CheckpointIQLBoNPolicy
        policy=CheckpointDEASBoNPolicy(config['actor_model_path'],config['critic_model_path'],
            config['embodiment_tag'],data.modality_config(),data.transform(),
            config['denoising_steps'],config['num_samples'],config['temperature'],device='cuda:0')
        restore_rng(metadata['rng'],arrays)
        actual=policy.get_action(prefixed(arrays,'input::'))
        output_arrays.update({'recomputed::'+key:value for key,value in actual.items()})
        errors={key:float(np.max(np.abs(actual[key]-expected[key]))) for key in expected}
        exact={key:bool(np.array_equal(actual[key],expected[key])) for key in expected}
        close={key:bool(np.allclose(actual[key],expected[key],atol=args.atol,rtol=0)) for key in expected}
        report.update(inference_recomputed=True,bitwise_exact=all(exact.values()),
                      within_tolerance=all(close.values()),atol=args.atol,max_abs_error=errors)
    actions_path.parent.mkdir(parents=True,exist_ok=True)
    with actions_path.open('wb') as stream:
        np.savez_compressed(stream,**output_arrays)
    report['output_actions']=str(actions_path.resolve())
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if args.recompute and not report['within_tolerance']: raise SystemExit(1)

if __name__=='__main__': main()
