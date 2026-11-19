# Raw RTG plus return prompt experiment

Authorized experiment dated 2026-10-16. Old DT/PDT code, checkpoints, trajectories and active B2000 PPO training are unchanged.

## Scope

Input at every timestep is `[raw RTG, return prompt, local state, local action]`.
There is no scenario/AP-count/area token. The original per-AP training structure is retained, so this is not a full reproduction of the paper's joint multi-BS architecture.

The seven physical 70% datasets contain 332 trajectories: A/B/C 35 each, B4 140, B36 17, B500 35, B2000 35. All 10080 steps are used. No second seed truncation, no new trajectory collection, no evaluation data used for statistics.

## Explicit bootstrap implementation choice

Compute gamma=1 undiscounted RTG for each AP, in raw reward units (rtg_scale=1). At each absolute timestep, pool all seven training datasets. Each trajectory has equal weight and APs have equal weight within a trajectory. Resample whole trajectories with replacement in 256 bootstrap replicates, seed 1; a resampled trajectory's full time series is retained together.

For each replicate compute the RTG mean and population variance using per-trajectory AP first and second moments. Final mu is the mean of replicate means; final sigma is the square root of the mean of replicate population variances, floored at 1e-6. Sigma is not the standard error of bootstrap means. This is recorded because the paper does not specify this estimator precisely. It is not per-task standardization.

Training prompt is `sigmoid((raw_RTG - mu[t]) / sigma[t])`, one scalar per timestep/AP. Original RTG remains a separate input. Statistics and the empirical training maximum prompt are saved in the checkpoint and return_prompt_stats.npz.

For a future paper-style new-task evaluation: set each timestep's prompt to the saved maximum training prompt; update the original target RTG with `RTG -= actual raw reward`. Do not subtract raw reward from a standardized value. No new-task test results are used to fit mu or sigma. Such simulations have not yet been started by this training command.

## Training

3 blocks, dimension 128, 1 head, dropout 0.1, context 8, batch 256, AdamW lr 5e-4 and weight decay 1e-4, seed 1, 1800 rounds x 150 updates = 270000 updates. Same loss and gradient clipping as baseline. CUDA 0, 4 dataloader workers, 2 CPU math threads to limit contention with existing PPO training. No W&B, no previous checkpoint loaded.

Output: `/workspace/Cell-Free-DT-new/dt_results/PDT_return_prompt_seven_ctx8/`.
Supervisor: `new_PDT_return_prompt_seven_ctx8`.
Log: `/tmp/new_PDT_return_prompt_seven_ctx8.log`.

The model uses four tokens per timestep instead of baseline's three plus an episode-level deployment token; compute cost is therefore not identical despite identical updates and hidden dimensions. Different input-group correlations and narrow expert data can still limit prompt usefulness. Loss alone does not establish deployment performance.

Tests: output dimensions, causal mask/future-input nonleakage, own-target-action nonleakage, finite forward/backward, gradients into both RTG and prompt embeddings. Training refuses to overwrite an existing experiment output directory.
