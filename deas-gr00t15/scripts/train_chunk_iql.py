#!/usr/bin/env python3
"""BC2-initialized scalar IQL critic with episode-cached QC chunks."""
import argparse
from pathlib import Path
import json
import os
import random
import time
import sys
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',required=True)
    p.add_argument('--normalization-metadata',required=True)
    p.add_argument('--dataset-path',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--samples-per-episode',type=int,default=64)
    p.add_argument('--discount',type=float,default=.99)
    p.add_argument('--expectile',type=float,default=.7)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--tau',type=float,default=.005)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--critic-encoder',choices=['deas','none'],default='deas',
                   help='deas: learned 64D tanh projection; none: frozen pooled features directly to Q/V')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-steps',type=int,default=10000)
    p.add_argument('--wandb-entity',default='RwHlabs')
    p.add_argument('--wandb-project',default='fmrl-gr00t-robocasa')
    p.add_argument('--smoke',action='store_true')
    return p.parse_args()


def main():
    args=arguments()
    if args.smoke:
        assert 0<args.steps<=5, 'Smoke is limited to five updates'
    else:
        assert os.environ.get('SLURM_JOB_ID'),'Production requires sbatch'
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from gr00t.data.iql_dataset import ChunkEpisodeStream
    from gr00t.model.iql.model import ChunkIQLCritic
    from gr00t.model.iql.core import expectile_loss,masked_mean,td_target
    from gr00t.model.transforms import DefaultDataCollator
    from safetensors.torch import save_file
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    assert args.horizon==16 and args.steps>0 and args.batch_size>0
    assert 0<args.discount<=1 and 0<args.expectile<1
    state=json.loads((Path(args.actor)/'trainer_state.json').read_text())
    assert state['global_step']==10000,'Expected the requested BC2 step10000'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    projection_policy = ('original category-specific 2048-1024-1024-1024-64 tanh projection'
                         if args.critic_encoder == 'deas' else 'direct pooled features; no learned projection or tanh')
    config=vars(args)|{'algorithm':'scalar IQL + QC complete chunks','reward':'DEAS last15 success expansion, then reward minus1',
        'boundary_policy':'success reward is terminal; boundary-only final transition without next observation excluded',
        'feature_policy':f'BC2 frozen VLM + VLLN + VL self attention, mean2048, {projection_policy}; original Q512x4 GELU and V256x4 BRONet; scalar final outputs',
        'precision':'frozen feature extractor and critic projection/Q/V matmuls BF16 autocast; master weights, optimizer, tanh, TD targets and losses FP32',
        'loader':'TorchCodec CPU full-episode decode, raw current+next episode prefetch, pinned previous critic normalization; length-weighted datasets and episodes',
        'actor_training':False,'wandb_model_upload':False}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    metadata=Path(args.normalization_metadata)
    shutil.copy2(metadata,output/'normalization_metadata.json')
    dataset=ChunkEpisodeStream(args.dataset_path,args.actor,args.seed,args.horizon,args.discount,args.samples_per_episode,args.normalization_metadata)
    params=dict(batch_size=args.batch_size,num_workers=args.workers,collate_fn=DefaultDataCollator(),pin_memory=True)
    if args.workers:params.update(persistent_workers=True,prefetch_factor=1)
    loader=DataLoader(dataset,**params)
    # Start CPU workers before creating CUDA state; no fork-after-CUDA.
    iterator=iter(loader)
    print('PREPARING_FIRST_BATCH',flush=True)
    first_batch=next(iterator)
    print('FIRST_BATCH_READY',flush=True)
    print('LOADING_BC2',args.actor,flush=True)
    model=ChunkIQLCritic(args.actor,horizon=args.horizon,critic_encoder=args.critic_encoder).to('cuda:0').train()
    q_params=list(model.head.critic.parameters())+list(model.head.backbone_encoder.parameters())
    v_params=list(model.head.value.parameters())
    optimizer=torch.optim.Adam(q_params+v_params,lr=args.lr)
    run=None
    if not args.smoke:
        import wandb
        run=wandb.init(entity=args.wandb_entity,project=args.wandb_project,name=output.name,
                       config=config,tags=['scalar-iql','qc-chunk16','bc2-step10000',f'seed{args.seed}','bf16',f'encoder-{args.critic_encoder}'],
                       settings=wandb.Settings(code_dir=None))
    frozen=list(model.backbone.parameters())+list(model.vlln.parameters())+list(model.vl_self_attention.parameters())
    assert not any(p.requires_grad for p in frozen)
    print('TRAIN_READY',json.dumps({'frozen_params':sum(p.numel() for p in frozen),
        'trainable_params':sum(p.numel() for p in q_params+v_params),'feature_dim':model.feature_dim,'critic_feature_dim':model.critic_feature_dim,'critic_encoder':args.critic_encoder}),flush=True)
    observed_dtypes={}
    handles=[]
    if args.smoke:
        # Observe real GEMM outputs once; no per-step nonzero-gradient requirement.
        observed_modules = [('q',model.head.critic.Q1.mlp[-1]),
                            ('v',model.head.value.value.final_layer)]
        if args.critic_encoder == 'deas':
            observed_modules.append(('projection',model.head.backbone_encoder.layer4))
        for name,module in observed_modules:
            handles.append(module.register_forward_hook(
                lambda module, inputs, out, name=name: observed_dtypes.__setitem__(name,str(out.dtype))))
    start=time.perf_counter();window=start;data_wait=0.;window_steps=0;last_batch=None
    for step in range(1,args.steps+1):
        tick=time.perf_counter();batch=first_batch if step==1 else next(iterator);data_wait+=time.perf_counter()-tick
        batch={k:v.to('cuda:0',non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        features=model.encode(batch)
        next_features=model.encode(batch,'next_eagle_')
        features=model.project(features,batch['embodiment_id'])
        with torch.no_grad():
            next_features=model.project(next_features,batch['embodiment_id'])
        states=batch['state'].float()*batch['state_mask'].float()
        next_states=batch['next_state'].float()*batch['next_state_mask'].float()
        actions=batch['action'].float()*batch['action_mask'].float()
        valid=batch['chunk_valid'].float()
        assert bool(batch['qc_valid'][:,-1].all()),'Incomplete chunk escaped sampler'
        with torch.no_grad():
            tq1,tq2=model.head.q(features,states,actions,target=True)
            minq=torch.minimum(tq1,tq2)
        vs=model.head.v(features,states)
        vloss=expectile_loss(minq-vs,args.expectile,valid)
        with torch.no_grad():
            nextv=model.head.v(next_features,next_states)
            target=td_target(batch['chunk_return'],batch['bootstrap_mask'],nextv,args.discount,args.horizon)
        q1,q2=model.head.q(features,states,actions)
        qloss=masked_mean(((q1-target).square()+(q2-target).square())/2,valid)
        if not bool(torch.isfinite(qloss)&torch.isfinite(vloss)):
            raise RuntimeError('Nonfinite IQL loss')
        optimizer.zero_grad(set_to_none=True);(qloss+vloss).backward()
        qnorm=torch.nn.utils.clip_grad_norm_(q_params+v_params,1.0,error_if_nonfinite=True)
        vnorm=qnorm
        optimizer.step()
        model.head.update_target(args.tau)
        window_steps+=1
        if step%10==0 or args.smoke or step==args.steps:
            elapsed=time.perf_counter()-window
            metrics={'step':step,'q_loss':float(qloss.detach()),'v_loss':float(vloss.detach()),
                'q_mean':float(torch.minimum(q1,q2).detach().mean()),'v_mean':float(vs.detach().mean()),
                'td_target_mean':float(target.mean()),'td_abs_error':float((q1.detach()-target).abs().mean()),
                'q_grad_norm':float(qnorm),'v_grad_norm':float(vnorm),
                'projection_grad_max_after_clip':max((float(p.grad.detach().abs().max()) for p in model.head.backbone_encoder.parameters() if p.grad is not None), default=0.0),
                'terminal_fraction':float((1-batch['bootstrap_mask']).mean()),
                'zero_cost_chunk_fraction':float((batch['chunk_return']==0).float().mean()),
                'seconds_per_step':elapsed/window_steps,'samples_per_second':args.batch_size*window_steps/elapsed,
                'data_wait_seconds_per_step':data_wait/window_steps,'elapsed_seconds':time.perf_counter()-start,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'learning_rate':args.lr,'discount':args.discount,'expectile':args.expectile}
            if args.critic_encoder == 'none':
                metrics.pop('projection_grad_max_after_clip')
            for name,values in [('q',torch.minimum(q1,q2)),('v',vs)]:
                values=values.detach().float()
                metrics.update({f'{name}_min':float(values.min()),f'{name}_max':float(values.max()),
                                f'{name}_std':float(values.std(unbiased=False))})
            if args.smoke:
                for k in ['seconds_per_step','samples_per_second','data_wait_seconds_per_step','elapsed_seconds']:
                    metrics.pop(k,None)
            print('IQL_METRICS',json.dumps(metrics),flush=True)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if run:run.log(metrics,step=step)
            window=time.perf_counter();data_wait=0.;window_steps=0
        if step%args.save_steps==0 or step==args.steps:
            checkpoint=output/f'checkpoint-{step}'
            checkpoint.mkdir(exist_ok=True)
            payload={'step':step,'head':model.head.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
            temp=checkpoint/'training.pt.tmp';torch.save(payload,temp);temp.replace(checkpoint/'training.pt')
            # Store frozen transferred weights once, making the critic self-contained.
            if not args.smoke and not (output/'frozen_features.safetensors').exists():
                tensors={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items() if not k.startswith('head.')}
                save_file(tensors,str(output/'frozen_features.safetensors.tmp'))
                (output/'frozen_features.safetensors.tmp').replace(output/'frozen_features.safetensors')
            (checkpoint/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            print('CHECKPOINT_SAVED',checkpoint,flush=True)
        last_batch=(features,states,actions)
    if args.smoke:
        assert not any(p.grad is not None for p in frozen)
        expected_dtypes = {'q':'torch.bfloat16','v':'torch.bfloat16'}
        if args.critic_encoder == 'deas': expected_dtypes['projection']='torch.float32'
        assert observed_dtypes == expected_dtypes, observed_dtypes
        # CategorySpecificLinear adds its FP32 bias after the BF16 bmm.
        assert all(p.dtype == torch.float32 for p in q_params+v_params)
        for handle in handles: handle.remove()
        # Check head checkpoint round trip against actual current action scores.
        saved=torch.load(output/f'checkpoint-{args.steps}/training.pt',weights_only=False,map_location='cpu')
        expected=model.score_actions(*last_batch).detach()
        model.head.load_state_dict(saved['head'])
        actual=model.score_actions(*last_batch).detach()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        (output/'smoke_passed.json').write_text(json.dumps({'status':'passed','steps':args.steps,
            'loss_finite':True,'compute_dtypes':observed_dtypes,'master_weights':'float32','frozen_features_no_grad':True,'checkpoint_scores_identical':True})+'\n')
    if run:run.finish()
    print('IQL_COMPLETE',args.steps,flush=True)

if __name__=='__main__':main()
