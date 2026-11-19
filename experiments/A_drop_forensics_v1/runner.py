# Forensics: re-run scenario A original PPO (dcoph43i eps480) seeds 3-5 (the weeks with stringent drops), verify obs == A.pkl episode seed-1,
# log every stringent packet outcome with the UE and ALL AP states at that instant, plus per-step AP states.
import os, sys, json, shutil
os.environ.update(DEBUG='0', TRAIN='0', EVAL='1', OMP_NUM_THREADS='1')
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R / 'experiments/B1000_checkpoint_screen'))
import screen as s
import numpy as np
from network.user_equipment import UserEquipment
O = R / 'analysis/sim_stats/A_drop_forensics_v1'
def snap(bss):
    return [dict(id=b.id, sleep=int(b.sleep), next_sleep=int(b._next_sleep), ants=int(b.num_ant), conn=int(b.conn_mode),
                 wake_left_ms=round(1000 * float(b.wakeup_time), 3), serving=len(b.ues), queue=len(b.queue), covered=len(b.covered_ues)) for b in bss]
@s.torch.no_grad()
def run(seed):
    s.torch.set_num_threads(1)
    plan = json.loads((R / 'experiments/B1000_checkpoint_screen/abc_first5_plan.json').read_text()); case = plan['A']; src = R / case['source']
    cfg = s.get_config().parse_args([])
    for k, v in s.yaml.safe_load((src / 'config.yaml').read_text()).items():
        if hasattr(cfg, k) and isinstance(v, dict) and 'value' in v: setattr(cfg, k, v['value'])
    cfg.model_version = case['suffix']; cfg.use_wandb = False
    base = O / f'SEED{seed}'; assert not base.exists()
    env = s.make_env(dict(name='A', scenario='A', area=1000, grid=4, cluster=.98), seed, str(base / '_sim'))
    policy = s.MappoPolicy(cfg, env.observation_space[0], env.cent_observation_space[0], env.action_space[0], model_dir=str(src))
    obs, _, _ = env.reset(); policy.prep_rollout()
    from trajectory_io import load_trajectory_pickle
    ref = load_trajectory_pickle(R / 'analysis/trajectory/uniform/A.pkl.zst')[seed - 1]['observations']
    bss = list(env.net.bss.values()); pos = [list(map(float, b.pos[:2])) for b in bss]
    ev = []; cur = {'t': -1}; orig = UserEquipment.finish_packet
    def fp(self, success):
        if self.service == 0:
            ev.append(dict(step=cur['t'], ue=str(self.id), success=bool(success), delay_ms=round(1000 * float(self.delay), 3), pos=list(map(float, self.pos[:2])),
                           serving=[b.id for b in self.bss], cover=sorted(b.id for b in self._cover_bss), status=str(self.status),
                           pkt_index=50 - int(self.packets_remaining), aps=snap(bss) if not success else None))
        return orig(self, success)
    UserEquipment.finish_packet = fp
    st = np.zeros((10080, 16, 5), np.int16); mo = 0.
    for t in range(10080):
        mo = max(mo, float(np.abs(np.asarray(obs) - ref[t]).max())); cur['t'] = t
        obs, _, _, done, _, _ = env.step(policy.act(obs, deterministic=True))
        for k, b in enumerate(bss): st[t, k] = [b.sleep, b._next_sleep, b.num_ant, len(b.ues), len(b.covered_ues)]
        if (t + 1) % 1440 == 0: print('DAY', seed, (t + 1) // 1440, mo, flush=True); assert shutil.disk_usage(R).free > 1024**3
    env.close(); UserEquipment.finish_packet = orig
    np.savez_compressed(base / 'ap_state.npz', state=st, cols=np.array(['sleep', 'next_sleep', 'ants', 'serving', 'covered']))
    sim = next((base / '_sim').rglob('ue_stats.csv')).parent
    for f in ['ue_stats.csv', 'bs_stats.csv', 'reward.csv']: shutil.copy(sim / f, base / f)
    shutil.rmtree(base / '_sim')
    v = dict(seed=seed, max_obs_diff_vs_pkl=mo, str_packets=len(ev), str_drops=sum(not e['success'] for e in ev), ap_pos=pos)
    (base / 'events.json').write_text(json.dumps(dict(meta=v, events=ev))); print('DONE', json.dumps(v), flush=True)
if __name__ == '__main__': run(int(sys.argv[1]))
