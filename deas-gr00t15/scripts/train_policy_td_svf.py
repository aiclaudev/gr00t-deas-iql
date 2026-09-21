#!/usr/bin/env python3
"""Single-GPU policy-TD SVF with QC episode prefetch and DiT LoRA."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',required=True)
    p.add_argument('--dataset-path',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--microbatch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--discount',type=float,default=.99)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--actor-lr',type=float,default=1e-5)
    p.add_argument('--q-lr',type=float,default=3e-4)
    p.add_argument('--value-lr',type=float,default=3e-4)
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--kappa',type=float,default=.4)
    p.add_argument('--g',type=float,default=.25)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--flow-steps',type=int,default=10)
    p.add_argument('--save-steps',type=int,default=1000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    return p.parse_args()


def main():
    args=arguments()
    assert os.environ.get('SLURM_JOB_ID'),'Run via sbatch'
    assert args.batch_size%args.microbatch_size==0 and args.steps>0
    assert 0<args.discount<=1 and 0<args.tau<=1
    if args.smoke:assert args.steps<=5
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from gr00t.data.iql_dataset import ChunkEpisodeStream
    from gr00t.model.svf.policy_td import PolicyTDSVF
    from gr00t.model.svf.objective import SVFConfig
    from gr00t.model.transforms import DefaultDataCollator
    from train_svf import export_actor,atomic_json
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=args.resume is not None)
    config=vars(args).copy();config['resume']=str(args.resume) if args.resume else None
    config.update(algorithm='policy-TD SVF, scalar double Q, min aggregation, no IQL V',
        loader='QC complete chunks; bulk episode decode + next-episode prefetch',
        initialization='BC2 actor and frozen reference; fresh Q/projection/soft value',
        reward='existing DEAS sparse last15 expansion minus1',horizon=16,
        soft_value_teacher='target Q with EMA projection',guidance_clip=2.0,
        precision='BF16 backbone/matmuls, FP32 trainable master weights and losses',
        resume_data_order='new episode stream on resume; not bitwise exact')
    atomic_json(output/'config.json',config)
    state=json.loads((Path(args.actor)/'trainer_state.json').read_text())
    assert state['global_step']==30000,'Expected BC2 step30000'
    stream=ChunkEpisodeStream(args.dataset_path,args.actor,seed=args.seed,horizon=16,
        discount=args.discount,samples_per_episode=64,
        normalization_metadata=Path(args.actor)/'experiment_cfg/metadata.json')
    kw=dict(batch_size=args.microbatch_size,num_workers=args.workers,collate_fn=DefaultDataCollator(),pin_memory=True)
    if args.workers:kw.update(prefetch_factor=1,persistent_workers=True)
    iterator=iter(DataLoader(stream,**kw));first=next(iterator)
    print('FIRST_BATCH_READY',flush=True)
    svf=SVFConfig(kappa=args.kappa,g=args.g,K=args.candidates,flow_steps=args.flow_steps)
    model=PolicyTDSVF(args.actor,svf,args.lora_rank,args.discount,args.tau).to('cuda').train()
    groups={'actor':[p for p in model.actor.parameters() if p.requires_grad],
        'q':list(model.q.parameters())+list(model.projection.parameters()),
        'soft_value':list(model.soft_value.parameters())}
    expected={id(p) for ps in groups.values() for p in ps}
    assert expected=={id(p) for p in model.parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([{'params':ps,'lr':{'actor':args.actor_lr,'q':args.q_lr,'soft_value':args.value_lr}[n],'name':n} for n,ps in groups.items()],weight_decay=1e-5)
    start=0
    if args.resume:
        saved=torch.load(args.resume/'training.pt',map_location='cpu',weights_only=False)
        for k in ['actor','dataset_path','discount','tau','lora_rank','kappa','g','candidates','flow_steps','batch_size','microbatch_size']:
            assert saved['config'][k]==config[k],f'Resume mismatch: {k}'
        model.load_state_dict(saved['model'],strict=False);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    run=None
    if not args.smoke:
        import wandb
        run=wandb.init(entity='RwHlabs',project='fmrl-gr00t-robocasa',name='svf-policytd-'+output.parent.name,config=config,dir=str(output))
        atomic_json(output/'wandb.json',dict(id=run.id,url=run.url))
    frozen=[p for p in model.parameters() if not p.requires_grad]
    initial={n:ps[0].detach().clone() for n,ps in groups.items()} if args.smoke else {}
    accumulation=args.batch_size//args.microbatch_size
    begin=time.perf_counter();window=begin;window_steps=0
    for step in range(start+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True);metrics={}
        for m in range(accumulation):
            batch=first if step==start+1 and m==0 else next(iterator)
            batch={k:v.to('cuda',non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            assert bool(batch['qc_valid'].all()) and bool(batch['chunk_valid'].bool().all())
            result=model(batch);loss=result['loss']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SVF loss')
            (loss/accumulation).backward()
            for k,v in dict(loss=loss.detach(),**result['metrics']).items():metrics[k]=metrics.get(k,0.)+float(v.detach())/accumulation
        for name,ps in groups.items():metrics[name+'_grad_norm']=float(torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True))
        optimizer.step();model.update_targets();window_steps+=1
        if args.smoke or step==start+1 or step%10==0:
            elapsed=time.perf_counter()-window
            metrics.update(step=step,seconds_per_step=elapsed/window_steps,walltime_seconds=time.perf_counter()-begin,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            print(json.dumps(metrics),flush=True)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            atomic_json(output/'status.json',dict(state='running',step=step,metrics=metrics))
            if run:run.log(metrics,step=step)
            window=time.perf_counter();window_steps=0
        if step%args.save_steps==0 or step==args.steps:
            tmp=output/f'checkpoint-{step}.partial';tmp.mkdir()
            weights={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith(('actor.backbone.','reference_head.'))}
            torch.save(dict(step=step,model=weights,optimizer=optimizer.state_dict(),config=config,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state()),tmp/'training.pt')
            if not args.smoke:export_actor(model,tmp/'actor',Path(args.actor))
            atomic_json(tmp/'complete.json',dict(step=step));tmp.rename(output/f'checkpoint-{step}')
    if args.smoke:
        assert not any(p.grad is not None for p in frozen)
        assert all(any(p.grad is not None and p.grad.abs().max()>0 for p in ps) for ps in groups.values())
        saved=torch.load(output/f'checkpoint-{args.steps}/training.pt',weights_only=False,map_location='cpu')
        for k,v in saved['model'].items():torch.testing.assert_close(v,model.state_dict()[k].cpu(),rtol=0,atol=0)
        atomic_json(output/'smoke_passed.json',dict(finite_loss=True,all_groups_receive_grad=True,frozen_no_grad=True,checkpoint_roundtrip=True))
    else:(output/'actor').symlink_to(f'checkpoint-{args.steps}/actor',target_is_directory=True)
    atomic_json(output/'status.json',dict(state='completed',step=args.steps))
    if run:run.finish()

if __name__=='__main__':main()
