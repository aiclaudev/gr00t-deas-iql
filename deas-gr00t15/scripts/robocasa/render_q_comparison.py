#!/usr/bin/env python3
"""Render success/failure videos beside recorded BoN Q curves (CPU only).

Requires numpy, opencv-python and ffmpeg on PATH. Input is one task's
output directory containing result.json, episodes.jsonl, videos/env_0,
and inference/index.jsonl plus call-*.npz. Supports n_envs=1, temperature=0,
and recordings with one video frame per simulation step plus reset frame.
Q is read from saved inference traces; no model inference is performed.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--task-dir',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True,help='New MP4 output path')
 parser.add_argument('--pair-index',type=int,default=0,help='0 = first success and failure; 1 = second pair')
 parser.add_argument('--success-episode',type=int)
 parser.add_argument('--failure-episode',type=int)
 parser.add_argument('--title',default=None)
 args=parser.parse_args()
 if args.pair_index<0:parser.error('--pair-index must be nonnegative')
 if not shutil.which('ffmpeg'):parser.error('ffmpeg is not on PATH')
 cv2.setNumThreads(1)
 d=args.task_dir.resolve();task=d.name;title=args.title or task
 result=json.loads((d/'result.json').read_text());config=result['config']
 if config.get('n_envs',1)!=1 or config.get('temperature',0)!=0:
  parser.error('This renderer requires n_envs=1 and temperature=0 (argmax BoN)')
 eps=[json.loads(x) for x in (d/'episodes.jsonl').read_text().splitlines() if x.strip()]
 selected=[]
 for success,explicit in [(True,args.success_episode),(False,args.failure_episode)]:
  matches=[e for e in eps if e['success']==success and e.get('env_index',0)==0]
  if explicit is not None:
   matches=[e for e in matches if e['episode']==explicit]
   if not matches:parser.error(f'Episode {explicit} missing or success flag does not match')
   selected.append(matches[0])
  else:
   if len(matches)<=args.pair_index:parser.error(f'Not enough episodes with success={success}')
   selected.append(matches[args.pair_index])
 index=[json.loads(x) for x in (d/'inference/index.jsonl').read_text().splitlines() if x.strip()]
 manifest=[]
 panels=[]
 for e in selected:
  rows=[]
  for x in index:
   if x['context']['env_episodes'][0]==e['episode'] and not x['context']['pending_reset'][0]:
    with np.load(d/'inference'/x['file'],allow_pickle=False) as z:q=z['q_scores'][:,0].astype(float)
    if not np.isfinite(q).all():raise ValueError('Non-finite Q scores')
    rows.append((x['context']['episode_steps'][0],float(q.max())))
  cap=cv2.VideoCapture(str(d/f"videos/env_0/rl-video-episode-{e['episode']}.mp4"))
  if not cap.isOpened() or not rows:raise ValueError(f'Missing video or Q trace for episode {e["episode"]}')
  if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != e['length']+1:raise ValueError('Expected one frame per simulator step plus reset frame')
  panels.append(dict(ep=e,rows=np.array(rows),cap=cap,n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),last=None))
 fps=panels[0]['cap'].get(cv2.CAP_PROP_FPS)
 if fps<=0 or any(abs(p['cap'].get(cv2.CAP_PROP_FPS)-fps)>0.01 for p in panels):raise ValueError('Input video FPS must match')
 lo=min(p['rows'][:,1].min() for p in panels);hi=max(p['rows'][:,1].max() for p in panels);pad=max((hi-lo)*.1,.01);lo-=pad;hi+=pad
 dest=args.output.resolve()
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(f'Output already exists: {dest}')
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x560','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
 for frame in range(max(p['n'] for p in panels)):
  canvas=np.zeros((560,1280,3),np.uint8)
  for j,p in enumerate(panels):
   ok,img=p['cap'].read()
   if ok:p['last']=cv2.resize(img,(640,360))
   elif frame<p['n']:raise ValueError('Video decode failed before expected end')
   panel=canvas[:,j*640:(j+1)*640];panel[45:405]=p['last']
   e=p['ep'];step=min(frame,e['length']);rows=p['rows'];idx=max(0,np.searchsorted(rows[:,0],step,side='right')-1);q=rows[idx,1]
   color=(90,225,90) if e['success'] else (90,90,250)
   def text(s,xy,c=(230,230,230),scale=.53):cv2.putText(panel,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,c,1,cv2.LINE_AA)
   text(f"{title} | {'SUCCESS' if e['success'] else 'FAILURE'} | ep {e['episode']}",(12,28),color)
   text(f"Step {step}/{e['length']} | selected Q = {q:.4f}"+(' | END' if frame>=p['n']-1 else ''),(12,425))
   text('Q = highest recorded candidate score (BoN argmax)',(12,446),scale=.46)
   x0,y0,w,h=65,460,555, 70
   cv2.rectangle(panel,(x0,y0),(x0+w,y0+h),(80,80,80),1)
   text(f'{hi:.2f}',(3,y0+10),scale=.4);text(f'{lo:.2f}',(3,y0+h),scale=.4)
   pts=np.array([(x0+int(s/e['length']*w),y0+h-int((v-lo)/(hi-lo)*h)) for s,v in rows],np.int32)
   cv2.polylines(panel,[pts],False,(100,100,100),1)
   cv2.polylines(panel,[pts[:idx+1]],False,color,2)
   cx=x0+int(step/e['length']*w);cv2.line(panel,(cx,y0),(cx,y0+h),(220,220,220),1)
   text('Simulation step (same Q scale on both sides)',(105,550),scale=.43)
  proc.stdin.write(canvas.tobytes())
 proc.stdin.close();assert proc.wait()==0
 for p in panels:p['cap'].release()
 manifest.append(dict(task=task,episodes=selected,file=str(dest),fps=fps,q_definition='max_candidates min(Q1,Q2)',alignment='frame 0 = reset; Q shown at action selection step; shorter video freezes at end'))
 print(dest,flush=True)
 (dest.with_suffix('.json')).write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
 main()
