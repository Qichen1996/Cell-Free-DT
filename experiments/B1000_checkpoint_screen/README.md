# B1000 multi-checkpoint screening

Only screening is authorized here. No new PPO training and no trajectory PKL collection.

Source run: mfxf31yk. Checkpoint suffixes: `_eps40`, `_eps120`, `_eps220`. These are early/middle/reference candidates, not assumed quality labels. Three independent tasks, seeds1–3, each10080steps. All source models are preserved.

Environment: B1000x1000m,16AP,spacing250m,neighbor0.25km,cluster0.90,min1AP. Deterministic original MappoPolicy, original environment and actor architecture.

Source training weights are80/0.2/0.2/1.8. All screening uses the same evaluation reward40/0.5/0.5/3, matching the retained uniform trajectory convention. Model SHA256, exact checkpoint suffix, environment and weights are recorded per seed.

Results: `analysis/sim_stats/B1000_checkpoint_screen/cp040`, `cp120`, `cp220`, each containingB/SEED1..3. Five CSV types and evaluation.json are retained; existing targets are never overwritten. A2GiB disk reserve is checked before each seed and at each simulated day. Failures stop without automatic retries, restarts or deletion. Logs are bounded with supervisor rotation.

After completion, compare pooled stringent/sensitive/tolerant packet-drop percentages and mean reward.csv pc_kw. Verify all nine seeds and10080reward rows before reporting a final comparison. More data collection and any cleanup require separate authorization.
