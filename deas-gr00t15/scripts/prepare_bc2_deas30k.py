from pathlib import Path
import shutil,datetime,json,subprocess
repo=Path('/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T');r=repo/'output/deas-training'/('bc2-deas-critic-bs32-30k-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));r.mkdir();(r/'logs').mkdir();src=r/'source-snapshot/repo';src.mkdir(parents=True)
for name in ['gr00t','scripts'] :shutil.copytree(repo/name,src/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=src/'gr00t/model/gr00t_n1_deas_critic.py';s=p.read_text();assert 'critic_head_cfg.online_q_feature_passes = 1' in s;s=s.replace('critic_head_cfg.online_q_feature_passes = 1','critic_head_cfg.online_q_feature_passes = 2');p.write_text(s)
old=repo/'output/deas-training/20260920T120047Z-bc2-shard-bs32-30k-seed42-fix/bc2_shard.sbatch'
s=old.read_text().replace("export WANDB_PROJECT='gr00t1.5 finetune'","export WANDB_PROJECT='fmrl-gr00t-robocasa'").replace('export WANDB_ENTITY=aiclaudev','export WANDB_ENTITY=RwHlabs');(r/'bc2.sbatch').write_text(s)
base=repo.parent/'models/GR00T-N1.5-3B';data=repo.parent/'data/deas_robocasa';tasks=['CoffeeSetupMug','PnPMicrowaveToCounter','TurnOffStove','PnPCounterToMicrowave'];datasets=[str(data/sub/t) for t in tasks for sub in ['demos','rollouts']]
args=['scripts/gr00t_deas_critic_finetune.py','--dataset-path',*datasets,'--base-model-path',str(base),'--output-dir',str(r/'03-critic'),'--num-gpus','1','--batch-size','32','--max-steps','30000','--save-steps','10000','--seed','42','--data-config','single_panda_gripper_rl','--critic-action-horizon','16','--discount1','0.9','--discount2','0.99','--expectile','0.7','--learning-rate','1e-4','--dataloader-num-workers','8','--run-name',r.name+'-critic']
import shlex
(r/'critic.sbatch').write_text(f'''#!/bin/bash
#SBATCH --account=sub
#SBATCH --qos=own
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --export=NONE
set -euo pipefail
source /home/nas_main/dohyunlee/miniconda3/bin/activate /home/nas_main/dohyunlee/miniconda3/envs/groot-train
export PYTHONPATH={src}
export USE_TF=0 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp/deas-critic-$SLURM_JOB_ID
export HF_MODULES_CACHE="$TMPDIR/hf-modules" TRITON_CACHE_DIR="$TMPDIR/triton"
mkdir -p "$TMPDIR"
export WANDB_ENTITY=RwHlabs WANDB_PROJECT=fmrl-gr00t-robocasa WANDB_MODE=online
export WANDB_LOG_MODEL=false WANDB_WATCH=false WANDB_SAVE_CODE=false
export WANDB_RUN_GROUP={r.name}
cd {src}
exec python -u {shlex.join(args)}
''')
m=dict(status='prepared',bc1=str(repo/'output/deas-training/20260919T173720Z-bc-bs32-30k-1gpu-seed42/01-bc-demo'),bc2_steps=30000,critic_steps=30000,global_batch=32,seed=42,save_steps=10000,critic_initialization=str(base),critic_algorithm='original DEAS option/HL-Gauss',critic_lr=1e-4,online_q_feature_passes=2,discount1=.9,discount2=.99,expectile=.7,bc_loader='shard',critic_loader='original LeRobotMixtureDataset',critic_args=args,notes='Retain executable config serialization fix; original legacy double feature transform restored in isolated snapshot. Existing working repository not changed.')
(r/'manifest.json').write_text(json.dumps(m,indent=2));(repo/'output/deas-training/latest-bc2-deas30k.txt').write_text(str(r));print(r)
