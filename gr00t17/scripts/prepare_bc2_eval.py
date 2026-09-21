from pathlib import Path
import shutil,json,datetime
base=Path('/home/nas_main/dohyunlee/jh_ws');n17=base/'Isaac-GR00T';old=base/'DEAS-Isaac-GR00T'
r=n17/'outputs'/('bc2-eval-10ep-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));r.mkdir();(r/'logs').mkdir()
for src,dst in [(n17/'gr00t',r/'n17-source/gr00t'),(old/'gr00t',r/'sim-source/gr00t')]:shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
s=(old/'scripts/eval_policy_robocasa.py').read_text().replace('if args.model_type == "gr00tn15":','if args.model_type == "gr00tn17":').replace('choices=["deas", "gr00tn15"]','choices=["deas", "gr00tn17"]').replace('if __name__ == "__main__":','from n17_bridge import N17PipePolicy\nGr00tPolicy = N17PipePolicy\n\nif __name__ == "__main__":')
(r/'sim-source/eval.py').write_text(s)
m=dict(checkpoint=str(n17/'outputs/bc2-deas-bs32-30k-seed42-20260920T114616Z/checkpoints/checkpoint-30000'),tasks=['CoffeeSetupMug','PnPMicrowaveToCounter','TurnOffStove','PnPCounterToMicrowave'],episodes=10,seed=42,action_horizon=16,execute_horizon=16,denoising_steps=4,model='GR00T1.7 BC2',qos='own')
(r/'manifest.json').write_text(json.dumps(m,indent=2));(n17/'outputs/latest-bc2-eval.txt').write_text(str(r));print(r)
