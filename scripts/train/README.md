# Training

These scripts fine-tune the three policies from the paper on the two LEGS tasks, using the paper settings.
Training runs inside `submodules/Psi0`, which contains ψ0, the openpi fork used for π0.5, and GR00T N1.6.

| Model | Trained parameters | Action chunk | Global batch | Steps | Paper hardware |
|---|---|---|---|---|---|
| ψ0 | action head only (VLM frozen) | 30 | 256 | 20k | 4×H100 |
| π0.5 | full fine-tune from the DROID checkpoint | 16 | 128 | 20k | 4×H100 |
| GR00T N1.6 | projector and action head (backbone frozen, action head re-initialized) | 16 | 24 | 20k | 4×H100 |

All three models use bf16, a learning rate of 1e-4 and 1k warmup steps (GR00T sets a warmup ratio of 0.05, which is also 1k steps).
States and actions are the 18-D end-effector layout shared by all three policies.

| `TASK` | Dataset (Hugging Face, LeRobot v2.1) | π0.5 config | GR00T preset |
|---|---|---|---|
| `locomani` | `hojjunekim/legs_locomani_uni` | `legs_locomani_uni_pi05` | `finetune_legs_locomani_uni_eef_gr00t_n16` |
| `turn` | `hojjunekim/legs_turn_uni` | `legs_turn_uni_pi05` | `finetune_legs_turn_uni_eef_gr00t_n16` |

## 1. Environments

You need [uv](https://docs.astral.sh/uv/), NVIDIA GPUs with a CUDA 12 driver, and the submodules checked out
recursively. The recursive checkout matters because Psi0's uv workspace references `third_party/SIMPLE`.

```bash
# not needed after a recursive clone; -c fetches SIMPLE's SSH-URL submodules over HTTPS (SETUP.md, section 1)
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive
bash scripts/train/setup_envs.sh            # all three; or pass --psi0, --pi05, --groot
```

| Environment | Used for |
|---|---|
| `submodules/Psi0/.venv-psi` | ψ0 (torch 2.7.0, flash-attn 2.7.4.post1) |
| `submodules/Psi0/.venv-openpi` | π0.5 (torch 2.7.1, JAX 0.5.3), and also the dataset export in `scripts/data` |
| `submodules/Psi0/src/gr00t/.venv` | GR00T N1.6 (torch 2.7.0, flash-attn 2.7.4.post1) |

If no prebuilt flash-attn wheel matches your platform, it is compiled from source. That takes a while and needs
`nvcc` from a CUDA 12 toolkit, with `CUDA_HOME` pointing at it. You can safely run the script again: completed
steps are skipped.

## 2. Datasets and base weights

```bash
bash scripts/train/download.sh              # everything; or pass --datasets, --psi0, --pi05, --groot, --task turn, --dataset ID
```

| What | Source | Saved to |
|---|---|---|
| LEGS datasets | `hojjunekim/legs_locomani_uni`, `hojjunekim/legs_turn_uni` | `$HF_LEROBOT_HOME/hojjunekim/<dataset>` |
| ψ0 VLM and action head | `USC-PSI-Lab/psi-model` (`psi0/`) | `$LEGS_CACHE/checkpoints/psi0/` |
| ψ0 processor config | `Qwen/Qwen3-VL-2B-Instruct` (config files only) | Hugging Face cache |
| π0.5 DROID checkpoint (PyTorch) | `USC-PSI-Lab/psi-model` (`openpi/pi05_droid/`) | `$LEGS_CACHE/checkpoints/openpi/pi05_droid/` |
| GR00T N1.6 base model | `nvidia/GR00T-N1.6-3B` | `$LEGS_CACHE/checkpoints/GR00T-N1.6-3B/` |

`LEGS_CACHE` defaults to `~/.cache/legs`. `HF_LEROBOT_HOME` defaults to `$LEGS_CACHE/lerobot`; if you already
set `HF_LEROBOT_HOME`, that value is used instead. On the first run, π0.5 also downloads its PaliGemma tokenizer
and GR00T downloads its Eagle processor, so that run needs network access.

The default datasets are the LEGS (200) sets. To train on another set from the
[dataset table](../../README.md#paper-datasets), fetch it with `--dataset` and pass the same id as `DATASET`,
with `TASK` set to the set's task:

```bash
bash scripts/train/download.sh --dataset hojjunekim/sam_turn_uni
TASK=turn DATASET=hojjunekim/sam_turn_uni bash scripts/train/psi0.sh
```

π0.5 uses the config named after the dataset, `<dataset>_pi05` (here `sam_turn_uni_pi05`), and every set in the
table has one. Serve the run with the same config: the π0.5 server reads normalization stats from the config's
dataset, not from the checkpoint ([`scripts/eval/README.md`, section 1](../eval/README.md#1-serve-a-checkpoint)).
For a dataset of your own, add a config to `submodules/Psi0/src/openpi/training/config.py` (a copy of a paper
config with its `name` and `repo_id` changed) and set `CONFIG` to its name.

## 3. Train

```bash
TASK=locomani bash scripts/train/psi0.sh
TASK=locomani bash scripts/train/pi05.sh
TASK=locomani bash scripts/train/groot.sh
```

Use `TASK=turn` for the turning task. Add `--dry-run` to print the resolved settings, check that every input is
present, and show the exact command without starting anything. Any other arguments are passed through to the
trainer, for example `--train.checkpointing_steps=5000` for ψ0 or `--set save_steps=5000` for GR00T.

Each script keeps the global batch at the paper value and derives the per-GPU batch from the number of GPUs.
The GPU count must divide the global batch:

| GPUs | ψ0 per GPU × accumulation | π0.5 per GPU | GR00T per GPU |
|---|---|---|---|
| 1 | 64 × 4 | 128 | 24 |
| 2 | 64 × 2 | 64 | 12 |
| 4 | 64 × 1 | 32 | 6 |
| 8 | 32 × 1 | 16 | 3 |

The openpi trainer has no gradient accumulation, so π0.5 on fewer than 4 GPUs needs much more memory per GPU.
If you lower the batch with `--batch_size=<global>`, the run no longer matches the paper. For ψ0, a smaller
`PER_GPU_BATCH` adds gradient accumulation instead. For GR00T, `GRAD_ACCUM` does the same.

| Variable | Default | Meaning |
|---|---|---|
| `TASK` | `locomani` | `locomani` or `turn` |
| `NUM_GPUS` | all visible GPUs | number of GPUs; a subset of `CUDA_VISIBLE_DEVICES` |
| `STEPS` | `20000` | training steps (the π0.5 cosine schedule decays over the same number) |
| `OUTPUT_DIR` | `submodules/Psi0/.runs` | root of the run directories |
| `EXP` | `<model>-<dataset>-<scope>-<steps>` | run name; re-running with the same name resumes |
| `DATASET` | the LEGS (200) set of `TASK` | Hugging Face id of the training set |
| `DATASET_DIR` | `$HF_LEROBOT_HOME/<dataset>` | local dataset directory |
| `LEGS_CACHE` | `~/.cache/legs` | where `download.sh` puts weights (and datasets) |
| `WANDB_PROJECT` | unset | enables Weights & Biases logging with your own `wandb login`; ψ0 and π0.5 log to this project, GR00T always logs to `finetune-gr00t-n1d6` |
| `PER_GPU_BATCH` | `min(64, 256 / GPUs)` | ψ0 only |
| `SAVE_INTERVAL` | `2500` | π0.5 only, checkpoint interval |
| `GRAD_ACCUM` | `1` | GR00T only |
| `CONFIG`, `PRESET` | per task | π0.5 config and GR00T preset |

On 4×H100, π0.5 takes about 2 s per step, which is about 11 h for 20k steps.

## Outputs and resuming

```
submodules/Psi0/.runs/
  finetune/<EXP>.real.flow1000.cosine.lr1.0e-04.b256.gpus<N>.<timestamp>/checkpoints/ckpt_<step>/   ψ0
  openpi-05/<CONFIG>/<EXP>/<step>/                                                                  π0.5
  gr00t-n16/<EXP>/checkpoint-<step>/                                                                GR00T N1.6
```

If a run is interrupted, start it again with the same `TASK` and `EXP` and it continues from its last checkpoint.

## Notes

- The first run of each model may write to the dataset directory:
  - ψ0 adds the 18-D entries to `meta/stats.json` if they are missing, and keeps the original as `stats.json.orig`.
  - π0.5 writes `norm_stats.json` if it is missing, and the trainer rewrites it there at every checkpoint save.
    The π0.5 server reads it from the dataset directory, not from the checkpoint.
  - GR00T copies `meta/modality_g1_eef_uni.json` over `meta/modality.json` (the original is kept as
    `meta/modality_sonic.json`) and writes `meta/relative_stats.json`.

  The published datasets already include the ψ0 and π0.5 files. ψ0 and π0.5 do not read `meta/modality.json`.
- The Psi0 entry points need a non-empty `submodules/Psi0/.env`. If it is missing, the scripts create one that sets
  `PSI_HOME`, and any variables exported in your shell take precedence over it.
- Keep dataset, cache and output paths free of spaces, because the Psi0 launch scripts split their arguments on
  whitespace.
