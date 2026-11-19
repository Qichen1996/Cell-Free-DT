# B1000 multi-checkpoint controlled comparison

The managed workflow waits for `new_B1000_multicp_traj35` to finish and verifies the final PKL before training. No existing datasets, models or simulation results are replaced or deleted.

- Dataset: 21 trajectories from checkpoint220, 7 from40 and7 from120, all complete10080-step B1000 trajectories. Other six datasets are unchanged: total332 trajectories, seed_fraction1.
- Architecture: original deployment-prompt PDT and ordinary no-prompt DT. This experiment does not introduce the separate return-prompt architecture.
- Both models: seed1, context8, 3blocks,128dimensions,1head,dropout0.1,batch256,lr0.0005,decay0.0001,1800rounds×150updates. Trained sequentially on cuda:0 to limit memory pressure.
- New models: `dt_results/PDT_seven_multicp70_ctx8/dt_8_1.pt` and `dt_results/DT_seven_multicp70_ctx8/dt_8_1.pt`.
- Evaluation: B1000,16AP,250m spacing,0.25km neighbor distance,cluster0.90,min1AP,weights40/0.5/0.5/3,seed1–5,10080steps.
- Actual RTG: **-523.07568359375** for both new models and both original-model evaluation controls. Existing DT formal B results at a different RTG are not used as a matched control.
- Results: `analysis/sim_stats/B1000_multicp_comparison/{PDT_multicp,DT_multicp,PDT_original,DT_original}/B/SEED1` through `SEED5`.
- Source PPO reference is only the trajectory-generating `mfxf31yk` checkpoint220, using matching seeds/environment; not a different PPO checkpoint or weights model. Existing checkpoint-screen CSVs cover only3seeds and are not a5seed control.
- Plan and all seven input SHA256s are recorded in `experiments/B1000_multicp_comparison/plan.json` after collection verification.
- Supervisor: `new_B1000_multicp_workflow`. Log: `/tmp/new_B1000_multicp_workflow.log`. Stops on failure or insufficient disk; never automatically restarts or cleans up.
- Seven-environment expansion is deferred until the B1000 comparison has been reviewed for improvement. No automatic RTG sweep or additional training budget.

Report drop by pooling per-class dropped_packets/completed_packets across matching seeds (delay_budget0.05/0.15/0.3); power is the mean of reward.csv pc_kw, not net_stats.avg_pc. Training and evaluation completion must be verified before reporting results.
