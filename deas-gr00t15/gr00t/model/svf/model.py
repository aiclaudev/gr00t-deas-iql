"""Joint SVF with independently frozen GR00T BC and DEAS Q teachers.

This preserves the saved DEAS online-Q feature path (two preprocessing passes).
It does not repair or alter the preceding IQL experiment.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .adapters import (
    ActionCoordinateAdapter,
    prepare_head_context,
    velocity_from_head,
    prepare_critic_features,
    score_critic,
)
from .networks import DoubleSoftValue
from .initialization import initialize_soft_value_from_critic
from .objective import SVFConfig, joint_losses
from .lora import ActorTuningConfig, apply_dit_lora


class JointSVFModel(nn.Module):
    def __init__(self, actor_path, critic_path, config: SVFConfig, device,
                 actor_tuning: ActorTuningConfig | None = None, soft_value_init: str = "random",
                 soft_value_loss: str = "mse"):
        super().__init__()
        if soft_value_init not in ("random", "critic-trunk", "critic-full"):
            raise ValueError("soft_value_init must be random, critic-trunk or critic-full")
        if soft_value_loss not in ("mse", "hl-gauss"):
            raise ValueError("soft_value_loss must be mse or hl-gauss")
        if soft_value_init == "critic-full" and soft_value_loss != "hl-gauss":
            raise ValueError("critic-full initialization requires hl-gauss soft-value loss")
        self.soft_value_init = soft_value_init
        self.soft_value_loss = soft_value_loss
        # Lazy imports keep the mathematical and mock unit tests independent of HF.
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        from gr00t.model.gr00t_n1_deas_critic import GR00T_N1_5_DEAS_Critic

        self.actor_path = str(Path(actor_path).resolve())
        self.critic_path = str(Path(critic_path).resolve())
        self.svf_config = config
        self.actor_tuning = actor_tuning or ActorTuningConfig()
        self.lora_target_modules = []
        self.config = config
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Real GR00T SVF training requires an allocated CUDA device")
        self.actor = GR00T_N1_5.from_pretrained(
            self.actor_path, tune_llm=False, tune_visual=False,
            tune_projector=True, tune_diffusion_model=True,
            torch_dtype=torch.float32, local_files_only=True,
        )
        self.actor.to(device)
        self.actor.backbone.requires_grad_(False).to(dtype=torch.bfloat16).eval()
        self.actor.action_head.float()
        self.actor.config.backbone_cfg["tune_visual"] = False
        self.actor.config.backbone_cfg["tune_llm"] = False
        self.reference_head = copy.deepcopy(self.actor.action_head)
        self.reference_head.requires_grad_(False).to(dtype=torch.bfloat16).eval()
        if self.actor_tuning.mode == "dit-lora":
            # Clone the untouched BC reference before adding student-only adapters.
            self.lora_target_modules = apply_dit_lora(self.actor.action_head, self.actor_tuning)

        self.teacher = GR00T_N1_5_DEAS_Critic.from_pretrained(
            self.critic_path, tune_llm=False, tune_visual=False,
            tune_critic=False, tune_value=False,
            torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device).requires_grad_(False).eval()
        if (self.actor.action_horizon, self.actor.action_dim) != (16, 32):
            raise ValueError("This RoboCasa SVF implementation expects 16x32 GR00T action chunks")
        if (self.teacher.critic_action_horizon, self.teacher.action_dim) != (16, 32):
            raise ValueError("DEAS teacher must score the same 16x32 action chunk")
        with open(Path(self.actor_path) / "experiment_cfg/metadata.json") as f:
            actor_metadata = json.load(f)["new_embodiment"]
        with open(Path(self.critic_path) / "experiment_cfg/metadata.json") as f:
            critic_metadata = json.load(f)["new_embodiment"]
        self.coordinates = ActionCoordinateAdapter.from_metadata(actor_metadata, critic_metadata).to(device)
        value_options = {"loss_type": self.soft_value_loss}
        if self.soft_value_loss == "hl-gauss":
            hlg = self.teacher.critic_head.hlg
            value_options.update(num_bins=hlg.num_bins, value_min=hlg.min_value,
                                 value_max=hlg.max_value, sigma=hlg.sigma)
        self.soft_value = DoubleSoftValue(
            feature_dim=64, state_dim=64, action_horizon=16, action_dim=32,
            time_dim=64, hidden_dim=512, depth=4, **value_options,
        ).to(device=device, dtype=torch.float32)
        self.soft_value_initialization = {"mode": "random"}
        if self.soft_value_init in ("critic-trunk", "critic-full"):
            self.soft_value_initialization = initialize_soft_value_from_critic(
                self.soft_value, self.teacher.critic_head, self.coordinates,
                copy_output=self.soft_value_init == "critic-full",
            )
        self.teacher_feature_passes = self.teacher.config.critic_cfg.get("online_q_feature_passes", 2)
        if type(self.teacher_feature_passes) is not int or self.teacher_feature_passes not in (1, 2):
            raise ValueError("Critic online_q_feature_passes must be 1 or 2")
        self.reference_microbatch_size = 16
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "actor"):
            self.actor.backbone.eval()
            if getattr(getattr(self, "actor_tuning", None), "mode", "full") == "dit-lora":
                # Frozen observation/state/action projectors keep their BC eval behavior.
                self.actor.action_head.eval()
                self.actor.action_head.model.train(mode)
        if hasattr(self, "reference_head"):
            self.reference_head.eval()
        if hasattr(self, "teacher"):
            self.teacher.eval()
        return self

    def training_description(self):
        return {
            "method": "joint_svf_frozen_deas_q",
            "actor_checkpoint": self.actor_path,
            "teacher_checkpoint": self.critic_path,
            "teacher_q": "online_min_q1_q2_hlgauss_expectation",
            "teacher_feature_passes": self.teacher_feature_passes,
            "teacher_feature_semantics": ("consistent_single_pass" if self.teacher_feature_passes == 1
                                          else "preserve_legacy_online_q_training_path"),
            "reference_bc_frozen": True,
            "critic_frozen": True,
            "actor_tuning": ("dit_attention_lora_only" if self.actor_tuning.mode == "dit-lora"
                             else "action_head_only_projectors_and_dit"),
            "actor_tuning_config": asdict(self.actor_tuning),
            "lora_target_modules": self.lora_target_modules,
            "soft_value": ("two_hl_gauss_heads_512x4_layernorm_gelu_fourier64"
                           if self.soft_value_loss == "hl-gauss" else
                           "two_scalar_heads_512x4_layernorm_gelu_fourier64"),
            "soft_value_loss": self.soft_value.loss_configuration(),
            "soft_value_initialization": self.soft_value_initialization,
            "svf": asdict(self.svf_config),
        }

    def forward(self, inputs):
        # The actor data transform is pinned to the BC2 checkpoint. Image/token inputs
        # are identical in form for both backbones, while state/action statistics are
        # explicitly converted before scoring the teacher.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actor_backbone_inputs, action_inputs = self.actor.prepare_input(inputs)
            with torch.no_grad():
                actor_raw = self.actor.backbone(actor_backbone_inputs)
                teacher_backbone_inputs, _ = self.teacher.prepare_input(inputs)
                teacher_raw = self.teacher.backbone(teacher_backbone_inputs)
                features = prepare_critic_features(
                    self.teacher, teacher_raw, action_inputs.embodiment_id,
                    passes=self.teacher_feature_passes,
                ).float()
                states = self.coordinates.convert_state(action_inputs.state.float())
                reference_context = prepare_head_context(
                    self.reference_head, actor_raw, action_inputs,
                )

        batch_size = action_inputs.action.shape[0]
        actor_context = None

        def actor_velocity(x, t):
            nonlocal actor_context
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if actor_context is None:
                    actor_context = prepare_head_context(
                        self.actor.action_head, actor_raw, action_inputs,
                    )
                return velocity_from_head(self.actor.action_head, actor_context, x, t).float()

        def reference_velocity(x, t):
            if x.shape[0] % batch_size:
                raise ValueError("Reference endpoint batches must repeat complete state batches")
            repeats = x.shape[0] // batch_size
            context = reference_context.repeat_batches(repeats)
            outputs = []
            # Bound temporary DiT activations even when all K candidates are vectorized.
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, x.shape[0], self.reference_microbatch_size):
                    stop = min(start + self.reference_microbatch_size, x.shape[0])
                    outputs.append(velocity_from_head(
                        self.reference_head, context.slice_batch(start, stop), x[start:stop], t[start:stop],
                    ).float())
            return torch.cat(outputs, dim=0)

        def teacher_score(endpoints):
            count, batch, horizon, width = endpoints.shape
            if batch != batch_size:
                raise ValueError("Teacher candidate batch shape does not match observations")
            with torch.no_grad():
                actions = self.coordinates.convert_action(endpoints.reshape(count * batch, horizon, width))
                repeated_features = features.repeat((count,) + (1,) * (features.ndim - 1))
                repeated_states = states.repeat((count,) + (1,) * (states.ndim - 1))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    q = score_critic(self.teacher, repeated_features, repeated_states, actions)
                return q.float().reshape(count, batch)

        # Small value-network regression, logsumexp and input gradients stay FP32.
        with torch.autocast(device_type="cuda", enabled=False):
            loss, metrics = joint_losses(
                self.soft_value, actor_velocity, reference_velocity, teacher_score,
                features.detach(), states.detach(), action_inputs.action.float(),
                action_inputs.action_mask.float(), self.svf_config,
            )
        return {"loss": loss, "metrics": metrics}
