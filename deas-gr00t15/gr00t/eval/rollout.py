"""Episode accounting for Gymnasium 1.0's NEXT_STEP vector autoreset."""
import time
import os
import json

import numpy as np


def evaluate_vector_policy(policy, env, num_episodes, action_transform=None, progress=None,
                           episode_callback=None, trace_recorder=None, execute_horizon=None):
    """Evaluate exactly N episodes; callbacks receive completion records.

    elapsed_seconds is measured from rollout start, including reset time.
    """
    if num_episodes < 1:
        raise ValueError('num_episodes must be positive')
    if execute_horizon is not None and execute_horizon < 1:
        raise ValueError('execute_horizon must be positive')
    started = time.monotonic()
    diagnostics = os.environ.get('GR00T_EVAL_DIAGNOSTICS') == '1'
    def log_stage(event, **details):
        if diagnostics:
            print('EVAL_STAGE ' + json.dumps(dict(event=event,
                  elapsed_seconds=time.monotonic()-started, **details)), flush=True)
    log_stage('initial_reset_begin', n_envs=env.num_envs)
    obs, _ = env.reset()
    log_stage('initial_reset_end')
    pending_reset = np.zeros(env.num_envs, dtype=bool)
    successes = np.zeros(env.num_envs, dtype=bool)
    lengths = np.zeros(env.num_envs, dtype=int)
    episode_successes, episode_lengths = [], []
    env_episodes = np.zeros(env.num_envs, dtype=int)
    policy_calls = 0
    while len(episode_successes) < num_episodes:
        if pending_reset.all():
            # NEXT_STEP reset ignores actions; reuse the prior action without
            # spending another BoN inference on the finished episode.
            log_stage('autoreset_begin', env_episodes=env_episodes.tolist())
            obs, _, _, _, _ = env.step(actions)
            log_stage('autoreset_end', env_episodes=env_episodes.tolist())
            pending_reset.fill(False)
            continue
        trace = trace_recorder.begin(obs, {
            "env_episodes": env_episodes.tolist(), "episode_steps": lengths.tolist(),
            "pending_reset": pending_reset.tolist(),
        }) if trace_recorder is not None else None
        policy_calls += 1
        log_stage('policy_begin', call=policy_calls, pending_reset=pending_reset.tolist(),
                  episode_steps=lengths.tolist(), completed=len(episode_successes))
        actions = policy.get_action(obs)
        log_stage('policy_end', call=policy_calls)
        predicted_actions = actions
        if action_transform is not None:
            actions = action_transform(actions)
        if execute_horizon is not None:
            for key, value in actions.items():
                if value.ndim < 3 or value.shape[1] < execute_horizon:
                    raise ValueError(f'{key} does not contain {execute_horizon} batched actions')
            # Score the full prediction above, execute only its prefix, then
            # request a fresh prediction from the next observation.
            actions = {key: value[:, :execute_horizon] for key, value in actions.items()}
        log_stage('env_step_begin', call=policy_calls, pending_reset=pending_reset.tolist())
        obs, _, terminated, truncated, info = env.step(actions)
        log_stage('env_step_end', call=policy_calls, terminated=np.asarray(terminated).tolist(),
                  truncated=np.asarray(truncated).tolist(), pending_reset=pending_reset.tolist())
        if trace_recorder is not None:
            executed = [0 if pending_reset[i] else int(info['num_executed_steps'][i])
                        for i in range(env.num_envs)]
            trace_recorder.finish(trace, policy, predicted_actions, actions, executed)
        for index in range(env.num_envs):
            # NEXT_STEP autoreset consumes a vector step without executing an action.
            if pending_reset[index]:
                continue
            successes[index] |= bool(np.any(info['success'][index]))
            lengths[index] += int(info['num_executed_steps'][index])
            if terminated[index] or truncated[index]:
                episode_successes.append(bool(successes[index]))
                episode_lengths.append(int(lengths[index]))
                log_stage('episode_done', episode=len(episode_successes)-1, env_index=index,
                          success=episode_successes[-1], length=episode_lengths[-1])
                if episode_callback is not None:
                    episode_callback({
                        "episode": len(episode_successes) - 1,
                        "env_index": index,
                        "success": episode_successes[-1],
                        "length": episode_lengths[-1],
                        "elapsed_seconds": time.monotonic() - started,
                    })
                env_episodes[index] += 1
                successes[index], lengths[index] = False, 0
                if progress is not None:
                    progress.update(1)
                if len(episode_successes) == num_episodes:
                    break
        pending_reset = np.logical_or(terminated, truncated)
    log_stage('rollout_complete', episodes=len(episode_successes), policy_calls=policy_calls)
    return episode_successes, episode_lengths
