import gymnasium as gym
import numpy as np

from gr00t.eval.rollout import evaluate_vector_policy
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper


class TinyKitchen(gym.Env):
    observation_space = gym.spaces.Dict({'state.x': gym.spaces.Box(-1, 1, (1,))})
    action_space = gym.spaces.Dict({'action.x': gym.spaces.Box(-1, 1, (1,))})

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return {'state.x': np.zeros(1, dtype=np.float32)}, {'success': False}

    def step(self, action):
        self.steps += 1
        return {'state.x': np.zeros(1, dtype=np.float32)}, 0.0, False, False, {'success': self.steps == 1}


def make_env():
    return MultiStepWrapper(TinyKitchen(), np.array([0]), np.array([0]), 4, max_episode_steps=6)


class ZeroPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        return {k: np.zeros_like(v) for k, v in self.env.action_space.sample().items()}


def test_exact_episode_count_and_next_step_autoreset():
    env = gym.vector.SyncVectorEnv([make_env, make_env])
    try:
        successes, lengths = evaluate_vector_policy(ZeroPolicy(env), env, 3)
        assert successes == [True, True, True]
        assert lengths == [6, 6, 6]
    finally:
        env.close()


def test_success_inside_chunk_and_partial_final_chunk():
    env = make_env()
    env.reset()
    action = {'action.x': np.zeros((4, 1))}
    _, _, terminated, truncated, info = env.step(action)
    assert info['success'].tolist() == [True]
    assert info['num_executed_steps'] == 4
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(action)
    assert info['num_executed_steps'] == 2
    assert not terminated and truncated
    env.close()


def test_episode_callback_counts_only_completed_requested_episodes():
    env = gym.vector.SyncVectorEnv([make_env, make_env])
    records = []
    try:
        successes, lengths = evaluate_vector_policy(
            ZeroPolicy(env), env, 3, episode_callback=records.append)
    finally:
        env.close()
    assert [record['episode'] for record in records] == [0, 1, 2]
    assert [record['env_index'] for record in records] == [0, 1, 0]
    assert [record['success'] for record in records] == successes
    assert [record['length'] for record in records] == lengths == [6, 6, 6]
    elapsed = [record['elapsed_seconds'] for record in records]
    assert elapsed == sorted(elapsed)
    assert elapsed[0] >= 0


def test_execute_eight_replans_from_fresh_observation_and_discards_tail():
    class ActionKitchen(TinyKitchen):
        observation_space = gym.spaces.Dict({'state.x': gym.spaces.Box(0, 1000, (1,))})
        action_space = gym.spaces.Dict({'action.x': gym.spaces.Box(-1000, 1000, (1,))})
        def __init__(self):
            self.executed = []
        def step(self, action):
            self.executed.append(float(action['action.x'][0]))
            self.steps += 1
            return {'state.x': np.array([self.steps], dtype=np.float32)}, 0., False, False, {'success': False}
    raw = ActionKitchen()
    env = gym.vector.SyncVectorEnv([lambda: MultiStepWrapper(
        raw, np.array([0]), np.array([0]), 8, max_episode_steps=18)])
    class ChunkPolicy:
        def __init__(self): self.observations = []
        def get_action(self, obs):
            offset = len(self.observations) * 100
            self.observations.append(int(obs['state.x'][0, 0, 0]))
            return {'action.x': np.arange(offset, offset + 16, dtype=np.float32).reshape(1, 16, 1)}
    class Trace:
        def __init__(self): self.rows = []
        def begin(self, obs, context): return context
        def finish(self, context, policy, predicted, supplied, executed):
            self.rows.append((predicted['action.x'].shape, supplied['action.x'].shape, executed))
    policy, trace = ChunkPolicy(), Trace()
    try:
        _, lengths = evaluate_vector_policy(policy, env, 1, execute_horizon=8, trace_recorder=trace)
    finally:
        env.close()
    assert policy.observations == [0, 8, 16]
    assert raw.executed == list(range(8)) + list(range(100, 108)) + [200, 201]
    assert lengths == [18]
    assert trace.rows == [((1, 16, 1), (1, 8, 1), [8]),
                          ((1, 16, 1), (1, 8, 1), [8]),
                          ((1, 16, 1), (1, 8, 1), [2])]
