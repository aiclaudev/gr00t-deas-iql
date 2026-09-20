# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import os
import random
import sys
import time
import warnings
from glob import glob
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
import torch
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm

from gr00t.eval.results import EvaluationRecorder
from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.robocasa_wrapper import load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import (
    BasePolicy,
    Gr00tPolicy,
    Gr00tDEASDualBoNPolicy,
)

warnings.simplefilter("ignore", category=FutureWarning)


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        rewards = []
        dones = []
        successes = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            rewards.extend(dic["rewards"])
            dones.extend(dic["dones"])
            successes.extend(dic["successes"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        if np.sum(successes) > 0:
            # cut transitions to the first successful state
            for i in range(len(states)):
                if successes[i]:
                    break
            states = states[: i + 1]
            actions = actions[: i + 1]
            if len(actions_abs) > 0:
                actions_abs = actions_abs[: i + 1]
            rewards = rewards[: i + 1]
            dones = successes[: i + 1]  # make dones same as successes
        else:
            dones[-1] = True  # make the last state a terminal state

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        ep_data_grp.create_dataset("dones", data=np.array(dones))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} rollouts so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


"""
Example command:

python scripts/eval_policy_robocasa.py --host localhost --port 5555
    --action_horizon 16
    --embodiment_tag gr1
    --data_config gr1_arms_waist
    --env_name CloseDrawer
    --num_episodes 10
provide --model_path to load up the model checkpoint in this script.
"""

def evaluation_protocol(args):
    """Actual environment settings; legacy --layout/--style do not override these pairs."""
    feature_passes = None
    if args.critic_model_path and args.model_type == "deas" and args.deas_backend == "checkpoint":
        checkpoint_config = json.loads((Path(args.critic_model_path) / "config.json").read_text())
        feature_passes = checkpoint_config["critic_cfg"].get("online_q_feature_passes", 2)
        if type(feature_passes) is not int or feature_passes not in (1, 2):
            raise ValueError("online_q_feature_passes must be 1 or 2")
    if args.critic_model_path and args.deas_backend == "iql":
        feature_passes = 1
    return {
        "environment": args.env_name,
        "robots": args.robots,
        "object_instance_split": "B",
        "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
        "generative_textures": "100p" if args.generative_textures else None,
        "randomize_cameras": False,
        "camera_width": 256,
        "camera_height": 256,
        "controller": "robot_default",
        "action_horizon": args.action_horizon,
        "execute_horizon": args.execute_horizon,
        "denoising_steps": args.denoising_steps,
        "action_noise": args.noise,
        "noise_clip": args.noise_smoothing,
        "n_envs": args.n_envs,
        "environment_seed_rule": "evaluation_seed + env_index",
        "episode_limit": "RoboCasa dataset registry horizon",
        "terminate_on_success": True,
        "episode_end_rule": "first success, native termination, or task horizon",
        "success_rule": "any success event during an episode",
        "num_samples": args.num_samples if args.critic_model_path else 1,
        "temperature": args.temperature if args.critic_model_path else None,
        "reward_shaping": args.reward_shaping,
        "critic_feature_passes": feature_passes,
    }


def run_evaluation(args, recorder):
    control_seed(args.seed)
    protocol = evaluation_protocol(args)

    assert args.data_config in ["single_panda_gripper_rl_inference"], (
        "Only single panda gripper RL inference data config is supported for now"
    )
    data_config = DATA_CONFIG_MAP[args.data_config](AS=args.action_horizon)

    if args.critic_model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        if args.model_type == "deas":
            policy_class = Gr00tDEASDualBoNPolicy
            if args.deas_backend == "checkpoint":
                from gr00t.model.checkpoint_bon_policy import CheckpointDEASBoNPolicy
                policy_class = CheckpointDEASBoNPolicy
            if args.deas_backend == "iql":
                from gr00t.model.iql_bon_policy import CheckpointIQLBoNPolicy
                policy_class = CheckpointIQLBoNPolicy
            policy = policy_class(
                actor_model_path=args.actor_model_path,
                critic_model_path=args.critic_model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                num_samples=args.num_samples,
                temperature=args.temperature,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    elif args.actor_model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        if args.model_type == "gr00tn15":
            policy = Gr00tPolicy(
                model_path=args.actor_model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    trace_recorder = None
    if args.save_inference_inputs:
        from gr00t.eval.inference_trace import InferenceTraceRecorder
        trace_recorder = InferenceTraceRecorder(Path(args.output_path) / "inference", vars(args))

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Record the settings actually passed to the simulator, including for HDF5.
    # The held-out evaluation protocol uses fixed layout/style pairs and split B.
    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )
    env_info = json.dumps({
        "env_name": args.env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "generative_textures": protocol["generative_textures"],
        "layout_and_style_ids": protocol["layout_and_style_ids"],
        "obj_instance_split": protocol["object_instance_split"],
    })

    if args.critic_model_path is not None:
        assert args.action_horizon == policy.model.critic_action_horizon, (
            f"Action horizon mismatch: {args.action_horizon} != {policy.model.critic_action_horizon}"
        )

    env = load_robocasa_gym_env(
        args.env_name,
        n_envs=args.n_envs,
        seed=args.seed,
        # robosuite-related configs
        robots=args.robots,
        camera_widths=256,
        camera_heights=256,
        render_onscreen=False,
        # robocasa-related configs
        obj_instance_split=protocol["object_instance_split"],
        randomize_cameras=False,
        layout_and_style_ids=tuple(tuple(pair) for pair in protocol["layout_and_style_ids"]),
        generative_textures="100p" if args.generative_textures else None,
        # data collection configs
        collect_data=args.collect_data,
        collect_directory=Path(args.data_collection_path) if args.collect_data else None,
        # video configs
        video_path=None if not args.save_video else Path(args.output_path) / "videos",
        # multi-step configs
        action_horizon=args.action_horizon,
        execute_horizon=args.execute_horizon,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        # reward configs
        reward_shaping=args.reward_shaping,
        terminate_on_success=protocol["terminate_on_success"],
    )

    try:
        # main evaluation loop
        start_time = time.time()

        from gr00t.eval.rollout import evaluate_vector_policy

        noise_levels = {
            "action.end_effector_position": 0.05,
            "action.end_effector_rotation": 0.3,
            "action.gripper_close": 1.0,
            "action.base_motion": 0.5,
            "action.control_mode": 1.0,
        }

        def add_noise(actions):
            if args.noise > 0:
                actions = {
                    key: value + np.clip(
                        np.random.normal(0, 1, size=value.shape) * noise_levels[key] * args.noise,
                        -args.noise_smoothing, args.noise_smoothing,
                    ) for key, value in actions.items()
                }
            return actions

        with tqdm(total=args.num_episodes, desc="Evaluating episodes") as pbar:
            episode_successes, episode_lengths = evaluate_vector_policy(
                policy, env, args.num_episodes, action_transform=add_noise,
                progress=pbar, episode_callback=recorder.record_episode, trace_recorder=trace_recorder,
                execute_horizon=args.execute_horizon,
            )
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            env.close()
        except BaseException as close_error:
            if not active_error:
                raise
            warnings.warn(f"Environment close also failed: {close_error}")

    print(f"Collecting {args.num_episodes} episodes took {time.time() - start_time:.2f} seconds")
    assert len(episode_successes) >= args.num_episodes, (
        f"Expected at least {args.num_episodes} episodes, got {len(episode_successes)}"
    )

    if args.save_video:
        from gr00t.eval.results import recorded_episode_videos
        recorder.result["videos"] = recorded_episode_videos(recorder.output_path)
        if len(recorder.result["videos"]) != args.num_episodes:
            raise RuntimeError("Every evaluated episode must have a saved video")

    if trace_recorder is not None:
        recorder.result["inference_trace"] = {"directory": str(trace_recorder.directory),
                                               "calls": trace_recorder.calls, "format": "npz-no-pickle"}
    print(f"Saved incremental evaluation results to {recorder.output_path}")

    if args.collect_data:
        print("Change collected data to hdf5 format")
        hdf5_path = gather_demonstrations_as_hdf5(args.data_collection_path, args.data_collection_path, env_info)
        convert_to_robomimic_format(hdf5_path)

    print(f"episode_successes: {episode_successes}")
    print(f"Success Rate (%): {np.mean(episode_successes)}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deas_backend", choices=["legacy", "checkpoint", "iql"], default="legacy",
                        help="checkpoint keeps actor/critic features and normalization separate")
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument("--n_envs", type=int, default=1, help="number of environments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="single_panda_gripper_rl_inference",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--execute_horizon", "--execute-horizon", type=int, default=None,
                        help="Execute this many actions before fresh inference; defaults to action_horizon")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--actor_model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--critic_model_path",
        type=str,
        default=None,
        help="[Optional] Path to the critic model checkpoint directory.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gr00tn15",
        choices=["deas", "gr00tn15"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples for BoN sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for BoN sampling.",
    )

    # robocasa env and evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CloseDrawer",
        help="Name of the robocasa environment to load",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output",
    )
    parser.add_argument(
        "--save_video",
        default=False,
        action="store_true",
        help="Whether to save the video",
    )

    # Robocasa env parameters
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument("--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    # Data collection parameters
    parser.add_argument(
        "--collect_data",
        action="store_true",
        default=False,
        help="Whether to collect data",
    )
    parser.add_argument(
        "--data_collection_path",
        type=str,
        default="",
        help="Path to save the data collection",
    )

    parser.add_argument(
        "--reward_shaping",
        action="store_true",
        default=False,
        help="Whether to use reward shaping",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise level for actions (0 disables added noise)",
    )

    parser.add_argument(
        "--noise_smoothing",
        type=float,
        default=0.3,
        help="ACtion noise smoothing level.",
    )

    parser.add_argument("--save_inference_inputs", action="store_true",
                        help="Save observations, RNG states, candidates and policy outputs for replay")
    parser.add_argument("--report_to", choices=["none", "wandb"], default="none")
    parser.add_argument("--training_seed", type=int, default=None,
                        help="Training seed recorded as checkpoint provenance")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    args = parser.parse_args()
    if args.n_envs < 1 or args.num_episodes < 1 or args.action_horizon < 1:
        parser.error("n_envs, num_episodes and action_horizon must be positive")
    if args.execute_horizon is None:
        args.execute_horizon = args.action_horizon
    if not 1 <= args.execute_horizon <= args.action_horizon:
        parser.error("execute_horizon must be between 1 and action_horizon")
    if args.critic_model_path and (not args.actor_model_path or args.model_type != "deas"):
        parser.error("critic_model_path requires actor_model_path and model_type=deas")
    if (args.save_video or args.save_inference_inputs) and not args.output_path:
        parser.error("video/input recording requires output_path")
    if args.denoising_steps < 1 or args.num_samples < 1:
        parser.error("denoising_steps and num_samples must be positive")
    if args.temperature < 0 or args.noise < 0 or args.noise_smoothing < 0:
        parser.error("temperature and noise settings must be nonnegative")
    with EvaluationRecorder(
        args.output_path or "./", args.num_episodes,
        config=vars(args), protocol=evaluation_protocol(args),
        report_to=args.report_to, training_seed=args.training_seed,
        evaluation_seed=args.seed, run_name=args.run_name, wandb_group=args.wandb_group,
    ) as recorder:
        run_evaluation(args, recorder)
