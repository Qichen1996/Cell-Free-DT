# B1000 prompt diagnosis

This is an isolated diagnostic experiment. Original models, training code and results are not changed. No retraining is performed.

## Fixed controls

- B traffic, area 1000 m, 16 APs, spacing 250 m, neighbor distance 0.25 km, cluster 0.90, minimum 1 AP.
- Reward weights 40 / 0.5 / 0.5 / 3, context 8, original raw gamma=1 RTG, no state normalization.
- Closed-loop simulations all use RTG -523.07568359375, seeds 1–3, 10080 steps each.
- DT and deployment-prompt PDT use their existing trained checkpoints. They are not newly trained ablation controls.

## Offline test

13,440 AP observations from B training seeds 1–5: all 16 APs, t=30+60k throughout the week, with 8-step expert state/action histories. True future RTG is a teacher-forced reproduction diagnostic, not an available deployment input. All selected seeds were included in training, so matching accuracy is not a held-out generalization score.

Compare true demonstration RTG against deployment RTG propagated along the same demonstration rewards. New PDT uses mapped training prompt, fixed historical global maximum, or B-training prompt quantiles. Present target-action nonleakage is tested for all three models. The actual source expert mfxf31yk_eps220 must reproduce the stored action labels exactly.

Prompt quartiles are estimated from uniformly spaced hourly samples across all 35 retained B training seeds and all APs, not from evaluation outcomes. Values: q25=0.7320761382579803, q50=0.7433800101280212, q75=0.7476419061422348. Historical global maximum is 0.8126307725906372. The sampled B training maximum is 0.7750362753868103, not a proven full-dataset maximum.

## Closed-loop variants

| Variant | Model | Prompt |
|---|---|---|
| dt | DT_seven_70_ctx8 | No prompt |
| deployment | PDT_seven_70_ctx8 | [B,16,1000] |
| q25 | PDT_return_prompt_seven_ctx8 | Fixed 0.7320761382579803 |
| q50 | PDT_return_prompt_seven_ctx8 | Fixed 0.7433800101280212 |
| q75 | PDT_return_prompt_seven_ctx8 | Fixed 0.7476419061422348 |
| mapped | PDT_return_prompt_seven_ctx8 | sigmoid((current raw RTG - training mu_t)/training sigma_t) |
| max | PDT_return_prompt_seven_ctx8 | Fixed 0.8126307725906372; reuse existing matching B seed1–3 results |

New outputs are `analysis/sim_stats/PDTdiag_B_{variant}/B/SEED1..3`. Existing max-prompt results are reused, not rerun or overwritten. Each variant records exact parameters and model SHA256 in evaluation.json. Rollout metadata is written only after five CSV files exist and reward contains 10080 rows.

Drop is computed from summed dropped/completed packet counts by delay_budget 0.05/0.15/0.3. Power is the mean of reward.csv pc_kw. Individual seed results remain available; three seeds do not establish statistical significance.

Offline matching alone cannot prove closed-loop distribution shift or explain its cause. Performance on new states must be interpreted separately. No diagnostic variant replaces the selected formal results without user authorization.
