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

import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.experiment.trainer import DualBrainTrainer, DualBrainRLTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator
from gr00t.utils.training_metrics import TrainingMetricsCallback
from gr00t.utils.experiment import (
    CheckpointFormatCallback,
    PolyakUpdateCallback,
    safe_save_model_for_hf_trainer,
)


class TrainRunner:
    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] = (None, None),
        resume_from_checkpoint: bool = False,
    ):
        self.training_args = training_args
        self.output_dir = Path(training_args.output_dir)
        self.exp_cfg_dir = self.output_dir / "experiment_cfg"
        self.exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.rank = int(os.environ.get("RANK", 0))
        report_to = training_args.report_to
        if isinstance(report_to, str):
            report_to = [] if report_to == "none" else [report_to]
        else:
            report_to = list(report_to or [])
        training_args.report_to = report_to
        if "wandb" in report_to:
            os.environ.setdefault("WANDB_PROJECT", "gr00t1.5 finetune")
            os.environ.setdefault("WANDB_MODE", "online")
            os.environ["WANDB_LOG_MODEL"] = "false"
            os.environ["WANDB_WATCH"] = "false"
            os.environ["WANDB_DIR"] = str(self.output_dir.expanduser().resolve())
            if os.environ.get("RUNTIME_ID"):
                os.environ.setdefault("WANDB_RUN_ID", os.environ["RUNTIME_ID"])
        # Set up training arguments
        training_args.run_name = (
            training_args.output_dir.split("/")[-1]
            if training_args.run_name is None
            else training_args.run_name
        )
        print(f"Run name: {training_args.run_name}")

        data_collator = DefaultDataCollator()

        # Make sure model_dtype and training_args dtype are compatible
        compute_dtype = torch.float16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)
        # Create trainer
        trainer = self.create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            compute_dtype=compute_dtype,
        )
        self.trainer = trainer
        trainer.callback_handler.callbacks.insert(0, TrainingMetricsCallback())

        # write the metadata to the experiment config dir
        train_dataset = getattr(train_dataset, "source_dataset", train_dataset)
        if self.rank == 0:
            metadata_json = {}
            if os.path.exists(self.exp_cfg_dir / "metadata.json"):
                with open(self.exp_cfg_dir / "metadata.json", "r") as f:
                    metadata_json = json.load(f)
            if isinstance(train_dataset, LeRobotSingleDataset):
                metadata_json.update(
                    {train_dataset.tag: train_dataset.metadata.model_dump(mode="json")}
                )
            elif isinstance(train_dataset, LeRobotMixtureDataset):
                metadata_json.update(
                    {
                        tag: metadata.model_dump(mode="json")
                        for tag, metadata in train_dataset.merged_metadata.items()
                    }
                )
            else:
                raise ValueError(f"Invalid dataset type: {type(train_dataset)}")
            with open(self.exp_cfg_dir / "metadata.json", "w") as f:
                json.dump(metadata_json, f, indent=4)

        # Record reporting without replacing the callbacks selected by TrainingArguments.
        if "wandb" in report_to and self.rank == 0:
            import wandb

            run = wandb.run
            wandb_config_file = self.output_dir / "wandb_config.json"
            with open(wandb_config_file, "w") as f:
                json.dump(
                    {
                        "project": getattr(run, "project", None) or os.environ["WANDB_PROJECT"],
                        "entity": getattr(run, "entity", None) or os.environ.get("WANDB_ENTITY"),
                        "run_id": getattr(run, "id", None) or os.environ.get("WANDB_RUN_ID", ""),
                        "run_name": getattr(run, "name", None) or training_args.run_name,
                        "mode": os.environ["WANDB_MODE"],
                    },
                    f,
                    indent=4,
                )
        if "azure_ml" in report_to:
            print("azure_ml logging is enabled.")
        if "tensorboard" in report_to:
            tensorboard_dir = Path(training_args.output_dir) / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        optimizers=(None, None),
        global_batch_size=None,
    ):
        # Set the gradient accumulation steps if global_batch_size is provided
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        # Create the trainer
        trainer = DualBrainTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            compute_dtype=compute_dtype,
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        # Log dataloader information
        is_stream = isinstance(trainer.train_dataset, torch.utils.data.IterableDataset)
        train_dl_len = "stream (max_steps)" if is_stream else len(trainer.get_train_dataloader())
        train_ds_len = "stream (max_steps)" if is_stream else len(trainer.train_dataset)
        # eval_dl_len = len(trainer.get_eval_dataloader()) # @note (k2): How to manage eval dataloader?

        print(
            f"train dataloader length: {train_dl_len}\n"
            # f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {train_ds_len}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()

        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.training_args.output_dir,
        )


class CriticTrainRunner(TrainRunner):
    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] = None,
        resume_from_checkpoint: bool = False,
    ):
        super().__init__(model, training_args, train_dataset, optimizers, resume_from_checkpoint)


    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        optimizers=(None, None),
        global_batch_size=None,
    ):
        # Set the gradient accumulation steps if global_batch_size is provided
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        # Create the trainer
        trainer = DualBrainRLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            optimizers=optimizers,
            compute_dtype=compute_dtype,
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)
        polyak_update_callback = PolyakUpdateCallback(
            target_model=model.critic_head.target_critic,
            source_model=model.critic_head.critic,
            tau=model.critic_head.config.rl_config["tau"],
        )
        trainer.add_callback(polyak_update_callback)

        # Log dataloader information
        is_stream = isinstance(trainer.train_dataset, torch.utils.data.IterableDataset)
        train_dl_len = "stream (max_steps)" if is_stream else len(trainer.get_train_dataloader())
        train_ds_len = "stream (max_steps)" if is_stream else len(trainer.train_dataset)
        # eval_dl_len = len(trainer.get_eval_dataloader()) # @note (k2): How to manage eval dataloader?

        print(
            f"train dataloader length: {train_dl_len}\n"
            # f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {train_ds_len}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer


class RLTrainRunner(TrainRunner):
    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] = None,
        resume_from_checkpoint: bool = False,
    ):
        super().__init__(model, training_args, train_dataset, optimizers, resume_from_checkpoint)


    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        optimizers=(None, None),
        global_batch_size=None,
    ):
        # Set the gradient accumulation steps if global_batch_size is provided
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        # Create the trainer
        trainer = DualBrainRLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
            optimizers=optimizers,
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)
        polyak_update_callback = PolyakUpdateCallback(
            target_model=model.action_head.target_critic,
            source_model=model.action_head.critic,
            tau=model.action_head.config.rl_config["tau"],
        )
        trainer.add_callback(polyak_update_callback)

        # Log dataloader information
        is_stream = isinstance(trainer.train_dataset, torch.utils.data.IterableDataset)
        train_dl_len = "stream (max_steps)" if is_stream else len(trainer.get_train_dataloader())
        train_ds_len = "stream (max_steps)" if is_stream else len(trainer.train_dataset)
        # eval_dl_len = len(trainer.get_eval_dataloader()) # @note (k2): How to manage eval dataloader?

        print(
            f"train dataloader length: {train_dl_len}\n"
            # f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {train_ds_len}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

