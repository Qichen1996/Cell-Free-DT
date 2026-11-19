# Read-only diagnostic (no training). B scene, one env seed, one PC policy. At every step BOTH the original PPO actor (deterministic)
# and the weight-normalized PDT (prompt pc/3.3 for all APs, mean RTG, context 8 of the DRIVER's own history) choose actions;
# the DRIVER's action is executed. driver=PPO -> PDT decision on PPO's states (decision bias); driver=PDT -> PDT's own state distribution (drift).
# Logs: both actions, AP sleep/ants per step, every stringent packet outcome (step, covering APs, pkt index). Sim trajectory.csv deleted.
import os, sys, json, importlib.util, shutil
os.environ.update(DEBUG='0', TRAIN='0', EVAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R))
def load(n, p): sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
col = load('col', R / 'experiments/tradeoff_selected_single_collect200_v1/runner.py')
import numpy as np, torch
sys.path.insert(0, str(R / 'experiments/selected_seven_pc33_v1')); import train_mixed as training
from network.user_equipment import UserEquipment
O = R / 'analysis/sim_stats/PDT_pc18_dual_v1'
M = R / 'dt_results/selected_seven_Bsingle200_v1/PDT/model.pt'; H = '00c227342a337f9239a3572e387b1cd7696e084cfede9acc443d04bd6a190514'
LAB = json.loads((R / 'experiments/PDT_single200_uniformprompt_RTG_v1/labels.json').read_text())
@torch.no_grad()
def run(tag, driver, seed):
    s = next(x for x in col.specs() if x['tag'] == tag); assert col.sha(M) == H
    base = O / driver / tag / f'SEED{seed}'; assert not base.exists(); stats = base / '_sim'
    env, obs, policy, par = col.setup(s, 'B', seed, stats); n = 16
    model = training.Model(True); ck = torch.load(M, map_location='cpu', weights_only=False); model.load_state_dict(ck['model_state_dict'], strict=True); model.eval()
    p = s['pc'] / 3.3; target = LAB[tag]['mean']
    from trajectory_io import load_trajectory_pickle
    ref = load_trajectory_pickle(next(f['path'] for f in json.loads((R / 'experiments/tradeoff_selected_single_collect200_v1/dataset-manifest.json').read_text())['files'] if f['model']['tag'] == tag and f['seed'] == seed))[0]['observations']
    bss = list(env.net.bss.values()); EV = []; orig = UserEquipment.finish_packet
    def fp(self, success):
        if self.service == 0:
            EV.append(dict(step=int(round(self.net._time / 0.02 - 0.5)), success=bool(success), pkt_index=50 - int(self.packets_remaining),
                           serving=[b.id for b in self.bss], cover=sorted(b.id for b in self._cover_bss)))
        return orig(self, success)
    UserEquipment.finish_packet = fp
    ts = torch.arange(10080).repeat(n, 1); S = torch.zeros(n, 10080, 110); A = torch.zeros(n, 10080, 2); G = torch.zeros(n, 10080, 1)
    running = torch.full((n, 1), target); prompt = torch.full((n, 1), p)
    a_ppo = np.zeros((10080, n, 2), np.int8); a_pdt = np.zeros((10080, n, 2), np.int8); st = np.zeros((10080, n, 3), np.int8); mo = 0.
    for t in range(10080):
        if driver == 'PPO': mo = max(mo, float(np.abs(np.asarray(obs) - ref[t]).max()))
        S[:, t] = torch.as_tensor(np.asarray(obs), dtype=torch.float32); G[:, t] = running
        l, r = max(0, t - 7), max(8, t + 1); h = model(ts[:, l:r], S[:, l:r], A[:, l:r], G[:, l:r], prompt); k = t if t < 8 else -1
        dt = torch.stack([x[:, k].argmax(-1) for x in h], -1).numpy(); dp = np.asarray(policy.act(obs, deterministic=True)).astype(np.int64)
        a_ppo[t] = dp; a_pdt[t] = dt; act = dp if driver == 'PPO' else dt
        A[:, t] = torch.as_tensor(act, dtype=torch.float32)
        obs, _, rew, done, _, _ = env.step(act); running -= torch.as_tensor(np.asarray(rew).reshape(n, 1), dtype=torch.float32)
        for j, b in enumerate(bss): st[t, j] = [b.sleep, b.num_ant // 4 if b.num_ant > 127 else b.num_ant, len(b.ues)]
        if (t + 1) % 1440 == 0: print('DAY', driver, tag, seed, (t + 1) // 1440, mo, flush=True); assert shutil.disk_usage(R).free > 0.8 * 1024**3
    env.close(); UserEquipment.finish_packet = orig
    np.savez_compressed(base / 'dual.npz', a_ppo=a_ppo, a_pdt=a_pdt, state=st)
    sim = next(stats.rglob('reward.csv')).parent
    for f in ['reward.csv', 'ue_stats.csv', 'bs_stats.csv']: shutil.copy(sim / f, base / f)
    shutil.rmtree(stats)
    meta = dict(tag=tag, driver=driver, seed=seed, prompt=p, rtg=target, model_sha256=H, max_obs_diff_vs_pkl=mo if driver == 'PPO' else None,
                str_packets=len(EV), str_drops=sum(not e['success'] for e in EV))
    (base / 'events.json').write_text(json.dumps(dict(meta=meta, events=EV))); print('DONE', json.dumps(meta), flush=True)
if __name__ == '__main__':
    if sys.argv[1] == 'one': run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        import subprocess, concurrent.futures
        B = Path(__file__).parent; (B / 'dual_logs').mkdir(exist_ok=True)
        tasks = [(t, d, s) for t in ['pc1.5_trainseed3_cp240', 'pc1.8_trainseed5_cpfinal', 'pc2.1_trainseed5_cpfinal'] for d in ['PPO', 'PDT'] for s in [1, 2, 3]]
        def job(x):
            with (B / 'dual_logs' / f'{x[0]}-{x[1]}-{x[2]}.log').open('x') as f: rc = subprocess.call([sys.executable, '-u', __file__, 'one', *map(str, x)], cwd=R, stdout=f, stderr=subprocess.STDOUT)
            print('JOB', x, rc, flush=True); return rc
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p: rcs = list(p.map(job, tasks))
        print('ALL_DUAL_DONE', rcs, flush=True)
