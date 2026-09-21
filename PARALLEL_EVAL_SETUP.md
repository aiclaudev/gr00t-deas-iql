# Parallel simulator evaluation: setting this up on another server

How to stand up parallel RoboCasa and LIBERO evaluation for GR00T N1.5 and N1.7,
and the problems you will hit on the way. Everything here was found by building
it on one shared A100 host; the failures listed are ones that actually occurred,
not hypotheticals.

The tooling lives in four top-level directories that sit beside the vendored
snapshots and never modify them:

| Directory | What it does |
| --- | --- |
| `local_eval/` | RoboCasa evaluation for N1.5 (`deas-gr00t15`) |
| `local_libero/` | LIBERO evaluation for N1.7 (`gr00t17`) |
| `local_train/` | DEAS critic training, throughput benchmark, GPU reservation |
| `local_collect/` | Re-collecting training data from evaluation rollouts |

Each has its own README with commands. This document is about the environment
and the parallelism, which is what does not transfer for free.

## 0. The machine this was built and verified on

Anything below was measured here; treat it as the reference configuration
rather than a requirement. The two islands pin different CUDA builds of torch
on purpose, and both run fine against the same driver.

| | |
| --- | --- |
| GPU | 4x NVIDIA A100 80GB PCIe (81920 MiB each) |
| Driver | 595.91.07 (supports up to CUDA 13.2) |
| CPU / RAM | 64 cores / 1007 GB |
| OS | Ubuntu 22.04.4 LTS, kernel 6.8.0-110-generic, glibc 2.35 |
| Shared | Yes — several users and containers. Check free memory before launching. |

### `deas-rc` (conda) — RoboCasa + GR00T N1.5

| Package | Version |
| --- | --- |
| python | 3.11.10 |
| torch | 2.5.1+**cu124** (cuDNN 9.1.0) |
| flash-attn | 2.7.1.post4 |
| numpy | 1.26.4 |
| transformers | 4.51.3 |
| diffusers | 0.30.2 |
| gymnasium | 1.0.0 |
| mujoco | 3.2.6 |
| robosuite | 1.5.2 |
| robocasa | 0.2.0 |
| robomimic | 0.2.0 (data re-collection only) |
| numba | 0.65.0 |

### `libero_island` (uv venv, uv 0.11.1) — LIBERO + GR00T N1.7

| Package | Version |
| --- | --- |
| python | 3.12.13 |
| torch | 2.9.0+**cu128** (cuDNN 9.10.2) |
| flash-attn | 2.8.3 (official prebuilt cp312 wheel) |
| numpy | 1.26.4 |
| transformers | 4.57.3 |
| diffusers | 0.35.1 |
| peft | 0.17.1 |
| gymnasium | 0.29.1 |
| mujoco | 3.3.1 |
| robosuite | 1.4.0 |
| LIBERO | git checkout, installed editable |
| numba | 0.65.1 |

Two things to carry over rather than re-derive:

- **The CUDA builds differ by design.** N1.5 runs cu124 and N1.7 runs cu128,
  because each pairs with a prebuilt flash-attn wheel for that exact
  python/torch ABI. Do not "unify" them; you will end up building flash-attn
  from source, which takes about an hour per environment.
- **numpy is 1.26.4 in both**, which is what makes the RoboCasa NumPy patch
  (section 2) necessary and what LIBERO's patched requirements pin to.

## 1. Why you need more than one environment

Two constraints force the split, and neither can be worked around:

**Both GR00T versions provide a package named `gr00t`.** The repository README
states it outright: use a separate environment for each; do not install both in
one. N1.5 and N1.7 cannot share an interpreter.

**RoboCasa and LIBERO pin different robosuite versions.** RoboCasa runs
robosuite 1.5.2; LIBERO pins 1.4.0. Their APIs differ enough that one
installation cannot serve both, and mujoco follows along (3.2.6 vs 3.3.1 —
robosuite 1.4.0 calls `mj_fullM(model, dst, M)`, whose signature changed in
mujoco 3.10.0).

So the matrix is (model version) x (simulator), and each cell is its own
environment of roughly 11-12 GB:

| | RoboCasa | LIBERO |
| --- | --- | --- |
| **N1.5** | `deas-rc` (conda) — built | not built |
| **N1.7** | not built | `libero_island` (uv venv) — built |

Budget about 12 GB of disk per cell. If you later need all four, consider the
ZMQ policy-server split instead — `~/Value/Isaac-GR00T` has a working example
where a msgpack serializer lets a torch-free simulator client talk to a policy
server. That turns the matrix from a product into a sum, and the simulator-side
environments shrink to about 2 GB because they need neither torch nor gr00t.

## 2. RoboCasa + N1.5: the `deas-rc` environment

Built by `local_eval/setup_env.sh`. Idempotent; re-running repairs what is
missing.

It **clones an existing GR00T training conda environment** rather than building
from scratch, because that environment already carries a compiled `flash-attn`
wheel for the exact python/torch ABI. This is not optional:
`gr00t/model/backbone/eagle2_hg_model/radio_model.py` imports `flash_attn` at
module level via `configuration_eagle2_5_vl.py`, so the model will not import
without it, and building it from source takes about an hour.

Resulting versions are in the table in section 0.

`robosuite` and `robocasa` are used from source over `PYTHONPATH` rather than
pip-installed, so an existing environment sharing those checkouts keeps working.
`importlib.metadata.version()` still resolves them, because their `.egg-info`
directories sit on `PYTHONPATH` too — the repository's own scripts rely on this.

### Things that will bite you

**RoboCasa hard-asserts NumPy 1.23.x.** The gr00t stack needs 1.26. The
repository ships `deas-gr00t15/scripts/robocasa/robocasa-numpy126.patch`, which
adds an opt-in behind `ROBOCASA_ALLOW_NUMPY_126=1` and leaves the 1.23.x path
untouched, so a co-existing environment on the same checkout is unaffected. Note
the checkout may not be a git repository — apply with `patch -p1`, not
`git apply`.

Resolving this is what makes in-process parallelism possible at all. It is why
this setup does not need the ZMQ split that `~/Value` uses.

**`pynput` does not build.** It pulls `evdev`, whose `ecodes.c` references input
event codes (`KEY_LINK_PHONE`) that older kernel headers do not define, and the
compile fails. `pynput` only backs robosuite's teleoperation devices and demo
scripts, which headless evaluation never imports, so leave it out.

**A cloned conda environment inherits its parent's activation hook.** Ours
prepended a different GR00T checkout to `PYTHONPATH`, which silently shadowed
this repository's `gr00t` package — `import gr00t` resolved elsewhere even with
`PYTHONPATH` set correctly, because the hook ran first. The setup script strips
the `PYTHONPATH` lines from the inherited hook and keeps its CUDA library paths.

**`robomimic` is needed for data re-collection**, not for evaluation. RoboCasa's
state replay uses it to regenerate camera observations. Install with `--no-deps`
so its stale pins stay out.

### Verify

```bash
conda run -n deas-rc python deas-gr00t15/scripts/robocasa/smoke.py \
    --env-name CoffeeSetupMug --n-envs 4 --output /tmp/smoke
```

Expect `status: passed`, three 256x256 camera views with real image variance,
and batched action chunks shaped `(n_envs, action_horizon, ...)`.

## 3. LIBERO + N1.7: the `libero_island` environment

Built by `local_libero/setup_env.sh`, following gr00t17's own
`setup_libero.sh`: a python 3.12 uv venv holding LIBERO, robosuite 1.4.0,
mujoco 3.3.1 and numpy 1.26.4 (full list in section 0), with gr00t17 exposed
through a `.pth` file rather than installed, so the island supplies gr00t's
dependencies itself and nothing is re-resolved.

Two changes from the vendored script: it uses a LIBERO checkout already on disk
instead of a git submodule (the snapshot carries no submodules), and it puts the
virtualenv outside the repository.

### Things that will bite you

**robosuite 1.4.0 writes to a hardcoded `/tmp/robosuite.log`.** Its `macros.py`
defaults `FILE_LOGGING_LEVEL = "DEBUG"` and `DefaultLogger` attaches a
`FileHandler` to that literal path. On a shared host the file usually belongs to
another user, so merely importing robosuite raises `PermissionError`. The escape
hatch is robosuite's own `macros_private.py`, which `__init__.py` imports ahead
of the defaults; `local_libero/_disable_robosuite_file_log.py` writes one with
file logging off.

**LIBERO prompts on first import** about downloading its datasets. Evaluation
needs only the bddl task files, which ship with the checkout. The setup answers
no and creates `~/.libero` itself.

**gr00t17's model dependencies are not in the vendored setup script.** Loading
`Gr00tN1d7` pulls in `diffusers` (`gr00t/model/modules/dit.py`) and `peft`;
`lmdb`, `jsonlines` and `datasets` are imported along the policy path. These
only surface when you first load a real checkpoint.

**flash-attn again, and again do not build it.** gr00t17 pins 2.8.3 and lists an
official prebuilt cp312 wheel URL in its `pyproject.toml` under
`[tool.uv.sources]`; the checkpoint's `conf.yaml` sets `use_flash_attention`.
Install from that URL.

**Skip `torchcodec`.** It is listed as a dependency but only decodes dataset
videos, and it needs a system FFmpeg 4-7 that a modern host may not have.
Simulator evaluation renders its frames directly, and mp4 writing goes through
`imageio-ffmpeg`'s bundled binary, so neither path touches it. `pyarrow` is
needed if you will use the data re-collection tooling.

### Checkpoints trained on another machine

Training records the backbone VLM's location as an absolute path on the training
host, in `processor/processor_config.json` under `processor_kwargs.model_name`.
Loading elsewhere dies in `Qwen3VLProcessor.from_pretrained` with
`Can't load image processor for /home/.../nvidia/Cosmos-Reason2-2B`.

`local_libero/prepare_checkpoint.py` builds a thin directory that symlinks the
weights and carries a corrected `processor/`, taking the replacement backbone
from the checkpoint's own `config.json` `model_name` (usually already a clean
Hub id). Nothing large is duplicated and the download stays untouched.

```bash
local_libero/prepare_checkpoint.py --source <hf snapshot dir> --output ./ckpt
```

N1.5 checkpoints do not need this — the Eagle backbone lives in the weights and
there is no separate processor directory.

### Verify

```bash
source ~/envs/libero_island/.venv/bin/activate
local_libero/list_tasks.py          # expect 130 tasks across 5 suites
```

## 4. Parallelism: what actually helps

Both repositories already contain vectorised evaluation —
`load_robocasa_gym_env(n_envs=N)` for N1.5 and `run_rollout_gymnasium_policy`
for N1.7 — and both step N spawned simulators together so each policy call is
one batched inference. You do not need to write that. What you do need is to
choose N and decide how many evaluations share a GPU.

### Thread limits are not optional

Every simulator worker is a separate process. Without limits each one claims a
thread per core, and with tens of workers the machine thrashes:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

`~/Value`'s launchers set the first four "to reduce CPU oversubscription".
`NUMBA_NUM_THREADS` is the one that is easy to miss and was missing from our
LIBERO environment: numba was installed and defaulting to 64. Both `env.sh`
files now set all five, and share a `NUMBA_CACHE_DIR` so workers do not each
recompile on startup.

### LIBERO: run the suites side by side, on one GPU

Measured with a 3B N1.7 checkpoint, `n_envs=16`, on one A100 80 GB:

| | One suite at a time | Four suites concurrently |
| --- | ---: | ---: |
| GPU utilisation | 44.5% | **88%** |
| GPU memory | 17 GB | 67 GB (about 8 GB per process) |
| CPU load (64 cores) | 14 | 28 |
| Wall time per task | 112 s | 86-156 s |
| Effective throughput | 0.0089 task/s | **0.024 task/s** |

A single suite leaves the GPU idle while its simulators step — utilisation swung
between 0% and 74%. Running the suites side by side fills those gaps: one infers
while another steps. Individual tasks get slower, total throughput roughly
**2.7x**. A full seed of the four standard suites (40 tasks, 2000 episodes) took
**28 minutes** instead of an estimated 1.5 hours.

Note that CPU load stayed at 28 of 64 with 64 worker processes, because
lockstep workers spend much of their time waiting on the batched policy call.
That idle time is exactly what the second suite fills.

Use `local_libero/run_suites_parallel.sh`. Because each process only knows its
own suite, each would write a `summary.json` covering only that suite; the
launcher finishes with `run_suite_gr00t17.py --summarise-only`, which rebuilds a
combined summary without loading a policy.

### RoboCasa: do not copy the LIBERO settings

This is the trap. Four RoboCasa tasks at `n_envs=16` **exhausted an 80 GB A100**:

```
CUDA OOM: 88 MiB free of 79.25 GiB
mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd
```

`0x8cdd` is `GL_FRAMEBUFFER_UNSUPPORTED`, the usual symptom of running out of
GPU memory while allocating an EGL framebuffer.

RoboCasa worker processes each render offscreen through EGL **on the GPU**, and
each loads a full kitchen scene with its meshes and textures. Three cameras
instead of LIBERO's two. Sixty-four EGL contexts on one card is too many.
Memory also grows late: environment creation is lazy, so the first reading looks
comfortable (37 GB) and then climbs as environments reset.

Measure the peak of a single task before choosing `--jobs-per-gpu`. Start at 1.

### GPU memory, roughly

About 8 GB per evaluation process for a 3B policy at `n_envs=16`, plus whatever
the simulator's rendering needs — small for LIBERO, substantial for RoboCasa.

On a shared host, check what else is resident before choosing. A neighbour
holding 60 GB with 0% utilisation still means you only have 18 GB.

## 5. Bugs in this tooling that only appear under parallelism

Worth knowing about if you extend it:

- **Manifest write race.** `local_eval/run_suite.py`'s worker threads all
  refresh `manifest.json`. With a shared scratch filename, two threads race and
  whichever renames second dies with `FileNotFoundError`. Invisible at
  `--jobs-per-gpu 1`. Fixed by giving each writer its own scratch name.
- **Episode overshoot.** `run_rollout_gymnasium_policy` raises `n_episodes` to at
  least `n_envs` and returns whatever the final batch completed, so 50 requested
  episodes came back as 52. The protocol counts exactly N per task; truncate.
- **Import path.** It is `gr00t.eval.sim.env_utils`, not `gr00t.eval.env_utils`.

## 6. Protocol note, if you are comparing against published numbers

The standard LIBERO protocol (OpenVLA and successors) evaluates four suites —
Spatial, Object, Goal, Long (`libero_10`) — at 10 tasks x 50 rollouts = 500
trials per suite, averaged over three random seeds. `libero_90` is excluded from
reported tables; it is a pre-training task pool.

gr00t17's `LiberoEnv` does **not** iterate each task's 50 canonical initial
states. It seeds the reset (`seed + env_index`), and after the first `n_envs`
episodes the vector environment autoresets without an explicit seed. The scale
and spirit match, but the initial-state set does not, so absolute numbers are
not directly comparable to published tables. Relative comparisons between
methods evaluated this same way are fine.

RoboCasa's held-out settings are fixed in `evaluation_protocol()` and ignore the
legacy `--layout`/`--style` flags: object instance split B, layout/style pairs
(1,1) (2,2) (4,4) (6,9) (7,10), 256x256 cameras, terminate on first success,
environment seed `evaluation_seed + env_index`.

**Keep `n_envs` fixed when comparing checkpoints.** Environment seeds are
`seed + env_index`, so changing it changes which scenes you evaluate.
