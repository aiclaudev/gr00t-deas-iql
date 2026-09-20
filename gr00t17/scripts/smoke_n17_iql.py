"""One real-data IQL update for N1.7 attachment correctness (not a benchmark)."""
import argparse,json
from pathlib import Path
import torch
import pandas as pd
from gr00t.model.iql.critic import N17IQLCritic
from gr00t.model.iql.core import chunk_fields
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag,MessageType

p=argparse.ArgumentParser();p.add_argument('--actor',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);args=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(42)
proc=Gr00tN1d7Processor.from_pretrained(args.actor,local_files_only=True,
    transformers_loading_kwargs={'local_files_only':True})
proc.eval()
mc=proc.modality_configs['robocasa_panda_omron']
loader=LeRobotEpisodeLoader(args.dataset,mc,decoder_kwargs={'num_ffmpeg_threads':1});ep=loader[0]
info=json.loads((Path(args.dataset)/'meta/info.json').read_text());eid=loader.episodes_metadata[0]['episode_index']
raw=pd.read_parquet(Path(args.dataset)/info['data_path'].format(episode_chunk=eid//info.get('chunks_size',1000),episode_index=eid))
r=raw['next.reward'].to_numpy().copy();terminal=r>0;ends=raw['next.done'].to_numpy(bool);ends[-1]=True
if r.sum()>0:r[-15:]=1
r=r-1
fields=[chunk_fields(r,ends,terminal,i,16,.99) for i in [0,1]]
def process(indices):
    samples=[proc([{'type':MessageType.EPISODE_STEP.value,'content':extract_step_data(ep,i,mc,EmbodimentTag.ROBOCASA_PANDA_OMRON)}]) for i in indices]
    return proc.collator(samples)['inputs']
a,b=process([0,1]),process([16,17]);print('PAIRED_REAL_BATCH_READY',flush=True)
model=N17IQLCritic.from_actor_checkpoint(args.actor).cuda().train()
params=[p for p in model.head.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=3e-4)
result=model(a,b,[x['chunk_return'] for x in fields],[x['bootstrap_mask'] for x in fields],[x['chunk_valid'] for x in fields])
assert torch.isfinite(result['loss']);result['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
assert all(p.grad is None for m in [model.backbone,model.vlln,model.vl_self_attention] for p in m.parameters())
assert any(p.grad is not None for p in model.head.backbone_encoder.parameters())
opt.step();model.head.update_target(.005)
model.eval();before=model.score_actions(a).clone();model.save_head(args.output)
model.head.load_state_dict(torch.load(Path(args.output)/'critic_head.pt',weights_only=True,map_location='cpu'))
after=model.score_actions(a);torch.testing.assert_close(before,after,rtol=0,atol=0)
report={'status':'passed','actor':args.actor,'updates':1,'batch':2,'spec':model.spec,'q_loss':float(result['q_loss'].detach()),'v_loss':float(result['v_loss'].detach()),'frozen_no_grad':True,'projection_has_grad':True,'head_roundtrip_identical':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
(Path(args.output)/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
