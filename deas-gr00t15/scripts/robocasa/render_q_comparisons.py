#!/usr/bin/env python3
"""Render recorded BoN videos with synchronized candidate Q summaries on CPU."""
import argparse
import bisect
import json
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TASKS = ('CoffeeSetupMug', 'PnPCounterToMicrowave', 'PnPMicrowaveToCounter', 'TurnOffStove')
BG = '#101827'
WHITE = '#e5edf6'
ORANGE = '#ffb454'
BLUE = '#70bfff'


@lru_cache(maxsize=16)
def font(size):
    # Matplotlib provides a portable TrueType font in this personal environment.
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)


def load_episode(directory, success, episode_id=None):
    result = json.loads((directory / 'result.json').read_text())
    config = result['config']
    if result['status'] != 'completed' or config['num_samples'] != 50 or config['temperature'] != 0:
        raise ValueError('Expected completed greedy BoN50 evaluation')
    video = next(v for v in result['videos'] if bool(v['success']) == success
                 and (episode_id is None or v['episode'] == episode_id))
    episode = video['episode']
    trace_dir = Path(result['inference_trace']['directory'])
    rows = []
    for line in (trace_dir / 'index.jsonl').read_text().splitlines():
        row = json.loads(line)
        context = row['context']
        if context['env_episodes'][0] != episode or context['pending_reset'][0]:
            continue
        if row['executed_steps'][0] <= 0:
            continue
        # Access only the tiny score array; never decompress observation images.
        with np.load(trace_dir / row['file'], allow_pickle=False) as archive:
            q = archive['q_scores'][:, 0].astype(float)
        if q.shape != (50,) or not np.isfinite(q).all():
            raise ValueError('Expected 50 finite candidate scores')
        rows.append(dict(step=context['episode_steps'][0], executed=row['executed_steps'][0],
                         selected=float(q.max()), median=float(np.median(q)), minimum=float(q.min()),
                         selected_candidate=int(q.argmax()), call=row['call'], file=row['file']))
    rows.sort(key=lambda row: row['step'])
    expected = 0
    for row in rows:
        if row['step'] != expected:
            raise ValueError(f'Q trace gap/overlap at {expected}')
        expected += row['executed']
    if expected != video['length']:
        raise ValueError('Q trace does not cover the entire episode')
    return dict(video=video, rows=rows, seed=result['seeds']['evaluation'])


def graph(episode, limits, max_steps):
    panel = Image.new('RGB', (640, 220), BG)
    draw = ImageDraw.Draw(panel)
    left, right, top, bottom = 65, 618, 20, 170
    low, high = limits
    def point(step, q):
        return (left + (right-left)*step/max_steps, bottom-(bottom-top)*(q-low)/(high-low))
    for i in range(5):
        q = low + (high-low)*i/4
        y = point(0, q)[1]
        draw.line((left,y,right,y),fill='#304057',width=1)
        draw.text((4,y-7),f'{q:.3f}',font=font(12),fill=WHITE)
    for step in range(0, max_steps+1, 100):
        x=point(step,low)[0]
        draw.text((x-12,bottom+6),str(step),font=font(12),fill=WHITE)
    rows=episode['rows']
    def stair(key):
        pts=[]
        for row in rows:
            pts.extend([point(row['step'],row[key]),point(row['step']+row['executed'],row[key])])
        return pts
    draw.polygon(stair('selected')+stair('minimum')[::-1],fill='#394354')
    draw.line(stair('median'),fill=BLUE,width=2)
    draw.line(stair('selected'),fill=ORANGE,width=3)
    draw.text((220,198),'Environment step',font=font(14),fill=WHITE)
    return panel, point


def render_pair(task, directory, output, success_episode=None, failure_episode=None, label=None):
    episodes=[load_episode(directory, True, success_episode),
              load_episode(directory, False, failure_episode)]
    max_steps=max(e['video']['length'] for e in episodes)
    values=[r[k] for e in episodes for r in e['rows'] for k in ('minimum','selected')]
    low,high=min(values),max(values)
    padding=max((high-low)*.08,.005)
    limits=(low-padding,high+padding)
    graphs=[graph(e,limits,max_steps) for e in episodes]
    inputs=[av.open(e['video']['path']) for e in episodes]
    streams=[c.streams.video[0] for c in inputs]
    for source_stream in streams:
        source_stream.codec_context.thread_count=1
    rates=[s.average_rate for s in streams]
    if rates[0] != rates[1] or rates[0] != 20:
        raise ValueError('Expected matching 20 FPS source videos')
    iterators=[iter(c.decode(video=0)) for c in inputs]
    last=[None,None];counts=[0,0]
    destination=output/f'{task}-success-vs-failure-q.mp4'
    temporary=destination.with_suffix('.partial.mp4')
    manifest=dict(task=task, label=label, fps=20, q_definition='min(Q1,Q2), HLG decoded scalar; not success probability',
                  selection=('explicit episode IDs, not selected by Q' if success_episode is not None or failure_episode is not None
                             else 'first successful and first failed episode in evaluation seed 0'),
                  shared_q_limits=limits, episodes=episodes, video=str(destination))
    with av.open(str(temporary),'w') as encoded:
        stream=encoded.add_stream('libx264',rate=rates[0]);stream.width=1280;stream.height=760
        stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
        stream.codec_context.thread_count=1
        for index in range(max_steps+1):
            canvas=Image.new('RGB',(1280,760),BG);draw=ImageDraw.Draw(canvas)
            title = f'{task}  |  BoN 50  |  seed {episodes[0]["seed"]}'
            if label: title += '  |  ' + label
            draw.text((20,8),title,font=font(21),fill=WHITE)
            for side,episode in enumerate(episodes):
                try:
                    frame=next(iterators[side]);counts[side]+=1;last[side]=frame.to_image().resize((640,360))
                except StopIteration:
                    if index <= episode['video']['length']:raise ValueError('Source video ended early')
                x=side*640;step=min(index,episode['video']['length'])
                rows=episode['rows'];pos=max(0,bisect.bisect_right([r['step'] for r in rows],step)-1);q=rows[pos]
                outcome='SUCCESS' if side==0 else 'FAILURE'
                color='#78e6a3' if side==0 else '#ff8992'
                draw.text((x+18,42),f"{outcome} (episode outcome)  |  ep {episode['video']['episode']}",font=font(18),fill=color)
                canvas.paste(last[side],(x,72))
                ended = index >= episode['video']['length']
                if ended:
                    draw.rectangle((x,400,x+640,432),fill=BG)
                    draw.text((x+14,407),f"EPISODE ENDED at step {step} | final frame held",font=font(16),fill=color)
                q_label = 'Last action Q' if ended else 'Selected Q'
                draw.text((x+16,441),f"Step {step:3d} / {episode['video']['length']}    {q_label} {q['selected']:.4f}",font=font(20),fill=ORANGE)
                draw.text((x+16,470),f"Median {q['median']:.4f}   Min {q['minimum']:.4f}   Candidate #{q['selected_candidate']+1}",font=font(16),fill=WHITE)
                panel,point=graphs[side];canvas.paste(panel,(x,503))
                px,py=point(step,q['selected'])
                draw.line((x+px,523,x+px,673),fill='white',width=2)
                draw.ellipse((x+px-4,503+py-4,x+px+4,503+py+4),fill=ORANGE)
            draw.line((640,40,640,720),fill='#536079',width=2)
            draw.text((18,731),'Full traces: orange = selected/max Q | blue = median | gray = candidate range | cursor = current step',font=font(17),fill=WHITE)
            if index==0:canvas.save(output/f'{task}-preview.jpg',quality=90)
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):encoded.mux(packet)
        for packet in stream.encode():encoded.mux(packet)
    for side,iterator in enumerate(iterators):
        try:next(iterator);raise ValueError('Source has unexpected extra frames')
        except StopIteration:pass
        if counts[side] != episodes[side]['video']['length']+1:raise ValueError('Video/step mismatch')
    for c in inputs:c.close()
    temporary.replace(destination)
    with av.open(str(destination)) as check:
        decoded=sum(1 for _ in check.decode(video=0))
    if decoded!=max_steps+1:raise ValueError('Encoded frame count mismatch')
    manifest['frames']=decoded
    (output/f'{task}-q-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(task=task,video=str(destination),frames=decoded,
                          success_episode=episodes[0]['video']['episode'],failure_episode=episodes[1]['video']['episode'])),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',default=None)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[render_pair(task,args.results/'eval-seed-0'/task,args.output,label=args.label) for task in TASKS]
    (args.output/'manifest.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'README.md').write_text('# BoN50 success/failure Q videos\n\n'
        'Each task pairs the first successful and first failed episode from evaluation seed 0. '
        'No Q-based selection of examples. Success/failure labels describe the episode outcome, not the current frame.\n\n'
        'Q is min(Q1,Q2), not a probability. The selected score is the maximum over 50 candidates. '
        'Both panels use the same Q axis; full traces include future values for retrospective inspection. '
        'The cursor and numeric values are synchronized to environment step (20 FPS, reset frame at step 0). '
        'Reset-only inference calls are excluded; every executed step is checked for trace coverage. '
        'Shorter episodes hold their final frame, with an EPISODE ENDED banner. '
        'The last action Q is the score before executing the final chunk, not a terminal-state Q prediction. '
        'Different episodes have different initial scenes, '
        'so this is a descriptive comparison, not a controlled test of critic ranking.\n')

if __name__=='__main__':main()
